#include "engine/models/qwen3_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"
#include "engine/models/qwen3_tts/prompt_tts_voice_design.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 8192;
constexpr int64_t kDefaultMaxBatchSize = 1;

// ggml's CUDA vector-matmul kernels cover a batch of at most 8 -- see
// MMVQ_MAX_BATCH_SIZE in ggml-cuda/mmvq.cuh and MMVF_MAX_BATCH_SIZE in
// mmvf.cuh. Past that the backend switches to the tiled path built for
// prefill-sized batches, which for one token per sequence means
// re-materializing the weights every step; going above 8 is never faster.
constexpr int64_t kMaxCudaBatchSize = 8;

std::shared_ptr<const Qwen3TTSAssets> require_assets(std::shared_ptr<const Qwen3TTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 TTS session requires assets");
    }
    return assets;
}

Qwen3TTSGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const Qwen3TTSConfig & config) {
    Qwen3TTSGenerationOptions options;
    options.max_new_tokens = config.max_new_tokens;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("Qwen3 TTS max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::find_option(
            request.options,
            {"subtalker_do_sample"})) {
        options.subtalker_do_sample = runtime::parse_bool_option(*value, "subtalker_do_sample");
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"subtalker_temperature"})) {
        options.subtalker_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(
            request.options,
            {"subtalker_top_k"})) {
        options.subtalker_top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(
            request.options,
            {"subtalker_top_p"})) {
        options.subtalker_top_p = *value;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    return options;
}

std::string ascii_lower(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

core::BackendConfig voice_prompt_backend_config(const runtime::SessionOptions & options) {
    core::BackendConfig config = options.backend;
    // Voice-clone prompt codes are discrete argmax outputs; keep this stage on CPU so CUDA TF32 math
    // cannot change the reference prompt that the main talker conditions on.
    config.type = core::BackendType::Cpu;
    return config;
}

bool custom_voice_icl_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.custom_voice_icl"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.custom_voice_icl");
    }
    return false;
}

bool register_primer_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.register_primer"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.register_primer");
    }
    return false;
}

std::string primer_text_from_options(const runtime::SessionOptions & options) {
    // The comma matters: a sentence-final carrier ("Alright.") invites the
    // model to emit EOS right after the forced frames; continuing prosody
    // keeps the generation going into the chunk text.
    return runtime::find_option(options.options, {"qwen3_tts.primer_text"}).value_or("Alright,");
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.mem_saver");
    }
    return false;
}

Qwen3TTSPerfMode perf_mode_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.perf_mode"})) {
        if (*value == "off" || *value == "standard") {
            return Qwen3TTSPerfMode::Standard;
        }
        if (*value == "flash_attention") {
            return Qwen3TTSPerfMode::FlashAttention;
        }
        throw std::runtime_error("Invalid qwen3_tts.perf_mode: " + *value);
    }
    return Qwen3TTSPerfMode::Standard;
}

bool source_contains_q8_tensor(const assets::TensorSource & source) {
    for (const auto & tensor : source.tensors()) {
        if (assets::tensor_storage_type_for_dtype(tensor.dtype) == assets::TensorStorageType::Q8_0) {
            return true;
        }
    }
    return false;
}

int64_t resolve_max_batch_size(const runtime::SessionOptions & options) {
    const int64_t size = runtime::parse_i64_option(
        options.options,
        {"qwen3_tts.max_batch", "max_batch"})
        .value_or(kDefaultMaxBatchSize);
    if (size < 1) {
        throw std::runtime_error("qwen3_tts.max_batch must be at least 1");
    }
    return size;
}

std::size_t voice_prompt_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"qwen3_tts.voice_prompt_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

std::size_t speaker_embedding_cache_slots_from_options(const runtime::SessionOptions & options) {
    // An entry is one hidden_size embedding (~8 KB), so a generous default is
    // nearly free and spares the CPU-side speaker encoder pass (~hundreds of
    // milliseconds) for every reference clip the session has already seen.
    constexpr int64_t kDefaultCacheSlots = 1024;
    const int64_t slots = runtime::parse_i64_option(options.options, {"qwen3_tts.speaker_embedding_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("qwen3_tts.speaker_embedding_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("qwen3_tts.speaker_embedding_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

void validate_talker_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error("Qwen3 TTS talker_weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, and f16");
}

}  // namespace

bool Qwen3TTSSession::VoicePromptCacheKeyEqual::operator()(
    const VoicePromptCacheKey & lhs,
    const VoicePromptCacheKey & rhs) const noexcept {
    return lhs.reference_text == rhs.reference_text &&
        lhs.mode == rhs.mode &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

bool Qwen3TTSSession::SpeakerEmbeddingCacheKeyEqual::operator()(
    const SpeakerEmbeddingCacheKey & lhs,
    const SpeakerEmbeddingCacheKey & rhs) const noexcept {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

Qwen3TTSSession::Qwen3TTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3TTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      mem_saver_(mem_saver_from_options(options)),
      custom_voice_icl_(custom_voice_icl_from_options(options)),
      register_primer_(register_primer_from_options(options)),
      primer_text_(primer_text_from_options(options)),
      perf_mode_(perf_mode_from_options(options)),
      text_tokenizer_(assets_),
      talker_(assets_->config.talker),
      voice_prompt_context_(voice_prompt_backend_config(options)),
      voice_prompt_cache_(voice_prompt_cache_slots_from_options(options)),
      speaker_embedding_cache_(speaker_embedding_cache_slots_from_options(options)) {
    talker_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_graph_arena_mb"}, talker_graph_arena_bytes_);
    speech_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_encoder_graph_arena_mb"}, speech_encoder_graph_arena_bytes_);
    speech_decoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_graph_arena_mb"}, speech_decoder_graph_arena_bytes_);
    speaker_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speaker_encoder_graph_arena_mb"}, speaker_encoder_graph_arena_bytes_);
    talker_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_constant_context_mb"}, talker_constant_context_bytes_);
    code_predictor_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.code_predictor_constant_context_mb"}, code_predictor_constant_context_bytes_);
    speech_decoder_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_constant_context_mb"}, speech_decoder_constant_context_bytes_);
    if (const auto it = options.options.find("qwen3_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "qwen3_tts.weight_type");
        validate_talker_weight_storage(storage_type);
        talker_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("qwen3_tts.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "qwen3_tts.conv_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.talker_weight_type"); it != options.options.end()) {
        talker_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_talker_weight_storage(talker_weight_storage_type_);
    }
    if (const auto it = options.options.find("qwen3_tts.speech_encoder_weight_type"); it != options.options.end()) {
        speech_encoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_encoder_weight_storage_type_, "qwen3_tts.speech_encoder_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.speech_decoder_weight_type"); it != options.options.end()) {
        speech_decoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_decoder_weight_storage_type_, "qwen3_tts.speech_decoder_weight_type");
    }
    if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention &&
        (!source_contains_q8_tensor(*assets_->model_weights) ||
         !source_contains_q8_tensor(*assets_->speech_tokenizer_weights))) {
        throw std::runtime_error("qwen3_tts.perf_mode=flash_attention is supported only with Q8_0 GGUF weights");
    }
    max_batch_size_ = resolve_max_batch_size(options);
    for (const auto & [key, _] : options.options) {
        if (key.rfind("qwen3_tts.", 0) == 0 &&
            key != "qwen3_tts.max_batch" &&
            key != "qwen3_tts.talker_graph_arena_mb" &&
            key != "qwen3_tts.speech_encoder_graph_arena_mb" &&
            key != "qwen3_tts.speech_decoder_graph_arena_mb" &&
            key != "qwen3_tts.speaker_encoder_graph_arena_mb" &&
            key != "qwen3_tts.talker_constant_context_mb" &&
            key != "qwen3_tts.code_predictor_constant_context_mb" &&
            key != "qwen3_tts.speech_decoder_constant_context_mb" &&
            key != "qwen3_tts.weight_type" &&
            key != "qwen3_tts.conv_weight_type" &&
            key != "qwen3_tts.talker_weight_type" &&
            key != "qwen3_tts.speech_encoder_weight_type" &&
            key != "qwen3_tts.speech_decoder_weight_type" &&
            key != "qwen3_tts.voice_prompt_cache_slots" &&
            key != "qwen3_tts.speaker_embedding_cache_slots" &&
            key != "qwen3_tts.custom_voice_icl" &&
            key != "qwen3_tts.register_primer" &&
            key != "qwen3_tts.primer_text" &&
            key != "qwen3_tts.perf_mode" &&
            key != "qwen3_tts.mem_saver") {
            throw std::runtime_error("unknown Qwen3 TTS session option: " + key);
        }
    }
    talker_weights_ = talker_.create_weights_runtime(
        assets_,
        options.backend.type,
        options.backend.device,
        std::max(1, options.backend.threads),
        talker_graph_arena_bytes_,
        talker_constant_context_bytes_,
        code_predictor_constant_context_bytes_,
        talker_weight_storage_type_,
        perf_mode_);
    talker_step_ = talker_.create_step_runtime(
        talker_weights_,
        assets_->config.talker.max_position_embeddings,
        assets_->config.max_new_tokens);
    // Clamp against the resolved backend, not the requested one: a session
    // created with BestAvailable may still land on CUDA.
    if (talker_step_->backend_type() == core::BackendType::Cuda && max_batch_size_ > kMaxCudaBatchSize) {
        debug::log_message(
            debug::LogLevel::Warning,
            "qwen3_tts",
            "qwen3_tts.max_batch=" + std::to_string(max_batch_size_) +
                " exceeds the CUDA vector-matmul batch limit of " +
                std::to_string(kMaxCudaBatchSize) +
                "; clamping, because larger batches fall back to a much slower kernel path");
        max_batch_size_ = kMaxCudaBatchSize;
    }
    speech_decoder_ = std::make_unique<Qwen3SpeechTokenizerDecoderRuntime>(
        assets_,
        execution_context(),
        speech_decoder_graph_arena_bytes_,
        speech_decoder_constant_context_bytes_,
        speech_decoder_weight_storage_type_,
        conv_weight_storage_type_,
        perf_mode_);
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 TTS currently supports offline sessions");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 base TTS model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign && task_.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Qwen3 voice design model only supports the VoiceDesign task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 custom voice model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base ||
        (assets_->config.variant == Qwen3TTSVariant::CustomVoice && custom_voice_icl_)) {
        try {
            speech_encoder_ = std::make_unique<Qwen3SpeechTokenizerEncoderRuntime>(
                assets_,
                voice_prompt_context_,
                speech_encoder_graph_arena_bytes_,
                speech_encoder_weight_storage_type_,
                conv_weight_storage_type_,
                perf_mode_);
        } catch (const std::exception & ex) {
            if (assets_->config.variant == Qwen3TTSVariant::Base) {
                throw;
            }
            // ICL cloning was requested but this CustomVoice package lacks the
            // speech-tokenizer encoder; keep the embedding path working and
            // make the reason visible.
            std::fprintf(
                stderr,
                "[qwen3_tts] speech tokenizer encoder unavailable, custom_voice_icl disabled: %s\n",
                ex.what());
            speech_encoder_.reset();
        }
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base
        || assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        // CustomVoice-Checkpoints bringen den Sprecher-Encoder normalerweise
        // nicht mit. Fehlt er, bleibt der Zeiger leer und nur das Klonen einer
        // eigenen Stimme entfaellt - die mitgelieferten Sprecher gehen weiter.
        try {
            speaker_encoder_ = std::make_unique<Qwen3SpeakerEncoderRuntime>(
                assets_,
                voice_prompt_context_,
                speaker_encoder_graph_arena_bytes_,
                conv_weight_storage_type_);
        } catch (const std::exception & ex) {
            if (assets_->config.variant == Qwen3TTSVariant::Base) {
                throw;
            }
            // Grund sichtbar machen: sonst sieht ein fehlender Encoder genauso
            // aus wie ein falsch benannter oder unvollstaendig kopierter.
            std::fprintf(
                stderr,
                "[qwen3_tts] speaker encoder unavailable for custom voice: %s\n",
                ex.what());
            speaker_encoder_.reset();
        }
    }
}

std::string Qwen3TTSSession::family() const {
    return "qwen3_tts";
}

runtime::VoiceTaskKind Qwen3TTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3TTSSession::run_mode() const {
    return task_.mode;
}

void Qwen3TTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult Qwen3TTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 TTS run");
    const auto wall_start = Clock::now();
    auto release_talker_cached_step_graph = [&]() {
        if (mem_saver_) {
            const auto release_start = Clock::now();
            const int64_t released_steps = talker_step_->release_cached_step_graph();
            debug::timing_log_scalar(
                "qwen3_tts.talker.cached_step_release_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
            debug::timing_log_scalar("qwen3_tts.talker.cached_step_released_steps", released_steps);
        }
    };
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3TTSVoiceDesignPromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            const auto prefill_start = Clock::now();
            const auto prefill = prompt_builder.build_prefill(qwen_request);
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            const auto talker_start = Clock::now();
            const auto codes = talker_step_->generate(
                prefill,
                qwen_request.generation,
                qwen_request.generation.repetition_penalty);
            talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
            const auto decoder_start = Clock::now();
            runtime::append_audio_buffer(
                merged_audio,
                speech_decoder_->decode(codes.generated_codes));
            decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
        }
        release_talker_cached_step_graph();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.voice_design_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    const bool custom_voice_icl_active =
        assets_->config.variant == Qwen3TTSVariant::CustomVoice &&
        custom_voice_icl_ && speech_encoder_ != nullptr && speaker_encoder_ != nullptr &&
        make_request(chunk_requests.front()).voice_clone.has_value();
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && !custom_voice_icl_active) {
        Qwen3TTSCustomVoicePromptBuilder prompt_builder(
            text_tokenizer_,
            speaker_encoder_.get(),
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        // A cloned voice resolves through the embedding cache once per
        // request; every chunk reuses the vector instead of re-running the
        // CPU-side speaker encoder.
        std::optional<Qwen3SpeakerEmbedding> cloned_embedding;
        {
            const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
            if (first_request.custom_voice.has_value() &&
                first_request.custom_voice->reference_audio.has_value() &&
                speaker_encoder_ != nullptr) {
                cloned_embedding = resolve_custom_voice_embedding(*first_request.custom_voice->reference_audio);
            }
        }
        // Register priming: generate one canonical carrier take per voice,
        // instruct, and language, then force every chunk to start with it.
        const Qwen3SpeechCodes * primer = nullptr;
        if (register_primer_) {
            const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
            std::string primer_key = primer_text_ + '\x1f' + first_request.language + '\x1f';
            if (first_request.custom_voice.has_value()) {
                primer_key += first_request.custom_voice->speaker + '\x1f' +
                    first_request.custom_voice->instruct + '\x1f';
            }
            if (cloned_embedding.has_value()) {
                uint64_t hash = 1469598103934665603ull;
                hash = fnv1a_mix(
                    hash,
                    cloned_embedding->values.data(),
                    cloned_embedding->values.size() * sizeof(float));
                primer_key += std::to_string(hash);
            }
            auto cached = primer_cache_.find(primer_key);
            if (cached == primer_cache_.end()) {
                const auto carrier_start = Clock::now();
                Qwen3TTSRequest carrier_request = first_request;
                carrier_request.text = primer_text_;
                // A fixed seed makes the carrier take canonical: every session
                // regenerates the same acoustic starting state for this voice.
                carrier_request.generation.seed = 424242;
                carrier_request.generation.max_new_tokens =
                    std::min<int64_t>(carrier_request.generation.max_new_tokens, 256);
                auto carrier_prefill = prompt_builder.build_prefill(
                    carrier_request,
                    cloned_embedding.has_value() ? &*cloned_embedding : nullptr);
                auto carrier_codes = talker_step_->generate(
                    carrier_prefill,
                    carrier_request.generation,
                    carrier_request.generation.repetition_penalty);
                debug::timing_log_scalar(
                    "qwen3_tts.primer.carrier_ms",
                    engine::debug::elapsed_ms(carrier_start, Clock::now()));
                if (carrier_codes.generated_codes.frames >= 3) {
                    cached = primer_cache_.emplace(primer_key, std::move(carrier_codes.generated_codes)).first;
                } else {
                    std::fprintf(
                        stderr,
                        "[qwen3_tts] register primer disabled: carrier take produced %lld frames\n",
                        static_cast<long long>(carrier_codes.generated_codes.frames));
                }
            }
            if (cached != primer_cache_.end()) {
                primer = &cached->second;
                debug::trace_log_scalar("qwen3_tts.primer.frames", primer->frames);
            }
        }
        for (const auto & chunk_request : chunk_requests) {
            Qwen3TTSRequest qwen_request = make_request(chunk_request);
            if (primer != nullptr) {
                qwen_request.text = primer_text_ + " " + qwen_request.text;
            }
            const auto prefill_start = Clock::now();
            auto prefill = prompt_builder.build_prefill(
                qwen_request,
                cloned_embedding.has_value() ? &*cloned_embedding : nullptr);
            if (primer != nullptr) {
                prefill.primer_codes = *primer;
            }
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            // Sprecher-Embedding auf Wunsch herausschreiben - dasselbe Ventil
            // wie im Base-Klonpfad, damit ein CustomVoice-Setup ohne geladenes
            // Base-Modell auskommt.
            if (const auto path = runtime::find_option(
                    chunk_request.options,
                    {"speaker_embedding_out"})) {
                if (!prefill.speaker_embedding.has_value()) {
                    throw std::runtime_error(
                        "speaker_embedding_out requires reference audio (voice_ref)");
                }
                write_speaker_embedding(*path, *prefill.speaker_embedding);
            }
            const auto talker_start = Clock::now();
            const auto codes = talker_step_->generate(
                prefill,
                qwen_request.generation,
                qwen_request.generation.repetition_penalty);
            talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
            const auto decoder_start = Clock::now();
            if (primer != nullptr && codes.generated_codes.frames > primer->frames) {
                // The forced carrier frames are decoded for acoustic context
                // and trimmed off, exactly like an ICL reference.
                const int64_t groups = codes.generated_codes.code_groups;
                Qwen3SpeechCodes primer_part;
                primer_part.code_groups = groups;
                primer_part.frames = primer->frames;
                primer_part.codes.assign(
                    codes.generated_codes.codes.begin(),
                    codes.generated_codes.codes.begin() +
                        static_cast<std::ptrdiff_t>(primer->frames * groups));
                Qwen3SpeechCodes rest;
                rest.code_groups = groups;
                rest.frames = codes.generated_codes.frames - primer->frames;
                rest.codes.assign(
                    codes.generated_codes.codes.begin() +
                        static_cast<std::ptrdiff_t>(primer->frames * groups),
                    codes.generated_codes.codes.end());
                runtime::append_audio_buffer(
                    merged_audio,
                    speech_decoder_->decode_and_trim_reference(primer_part, rest));
            } else {
                runtime::append_audio_buffer(
                    merged_audio,
                    speech_decoder_->decode(codes.generated_codes));
            }
            decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
        }
        release_talker_cached_step_graph();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.custom_voice_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error("Qwen3 base TTS requires voice clone reference audio");
    }
    if (speech_encoder_ == nullptr || speaker_encoder_ == nullptr) {
        throw std::runtime_error("Qwen3 base TTS session is missing voice clone runtimes");
    }
    Qwen3TTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        *speech_encoder_,
        *speaker_encoder_,
        assets_->config.talker.max_position_embeddings);
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double talker_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const Qwen3TTSRequest qwen_request = make_request(chunk_request);
        const auto prompt_start = Clock::now();
        const auto & voice_prompt = resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder);
        // Sprecher-Embedding auf Wunsch herausschreiben. Es entsteht hier
        // ohnehin; ohne diesen Ausgang gaebe es keinen Weg, den Vektor einer
        // Stimme aufzubewahren - CustomVoice erwartet genau ihn.
        if (const auto path = runtime::find_option(
                chunk_request.options,
                {"speaker_embedding_out"})) {
            write_speaker_embedding(*path, voice_prompt.speaker_embedding);
        }
        prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
        const auto prefill_start = Clock::now();
        auto prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        // CustomVoice requests routed through the ICL path keep their style
        // instruction; the talker prepends it in front of the ICL layout.
        if (qwen_request.custom_voice.has_value() && !qwen_request.custom_voice->instruct.empty()) {
            prefill.instruct_ids =
                text_tokenizer_.encode(text_tokenizer_.build_instruct_prompt(qwen_request.custom_voice->instruct));
        }
        prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        if (!voice_prompt.reference_codes.has_value()) {
            throw std::runtime_error("Qwen3 base TTS talker currently requires ICL reference codes");
        }
        const auto talker_start = Clock::now();
        const auto codes = talker_step_->generate(
            prefill,
            qwen_request.generation,
            qwen_request.generation.repetition_penalty);
        talker_ms += engine::debug::elapsed_ms(talker_start, Clock::now());
        const auto decoder_start = Clock::now();
        runtime::append_audio_buffer(
            merged_audio,
            speech_decoder_->decode_and_trim_reference(
                *voice_prompt.reference_codes,
                codes.generated_codes));
        decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    }
    release_talker_cached_step_graph();
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("qwen3_tts.voice_prompt_ms", prompt_ms);
    debug::timing_log_scalar("qwen3_tts.prefill_build_ms", prefill_ms);
    debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

int64_t Qwen3TTSSession::max_batch_size() const {
    return max_batch_size_;
}

std::vector<runtime::BatchedTaskResult> Qwen3TTSSession::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    require_prepared("Qwen3 TTS run_batch");
    if (requests.empty()) {
        throw std::runtime_error("Qwen3 TTS run_batch requires at least one request");
    }
    if (max_batch_size_ <= 1) {
        // Batching disabled. Run each request on its own and translate a throw
        // into the same per-request error the batched path reports, so callers
        // only handle one shape of failure. A single request stays on the
        // batched path below: its text chunks are independent generations and
        // batch against each other.
        std::vector<runtime::BatchedTaskResult> out;
        out.reserve(requests.size());
        for (const auto & request : requests) {
            runtime::BatchedTaskResult entry;
            try {
                entry.result = run(request);
            } catch (const std::exception & ex) {
                entry.error = ex.what();
            }
            out.push_back(std::move(entry));
        }
        return out;
    }

    const auto wall_start = Clock::now();

    // Long text is chunked before generation, so the unit of batching is a
    // chunk, not a request. Chunks of one request are independent generations
    // and only have to be concatenated in order afterwards.
    struct ChunkItem {
        size_t request_index = 0;
        size_t chunk_index = 0;
        size_t text_length = 0;
        Qwen3TalkerBatchItem item;
    };
    std::vector<ChunkItem> items;
    std::vector<size_t> chunk_counts(requests.size(), 0);
    // A request whose prompt cannot be built (bad speaker, token limit, missing
    // reference) fails alone; its neighbours still run.
    std::vector<std::string> request_errors(requests.size());
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        try {
            auto batch_items = build_batch_items(requests[request_index]);
            chunk_counts[request_index] = batch_items.size();
            for (size_t chunk_index = 0; chunk_index < batch_items.size(); ++chunk_index) {
                ChunkItem item;
                item.request_index = request_index;
                item.chunk_index = chunk_index;
                item.text_length = batch_items[chunk_index].prefill.input_ids.size();
                item.item = std::move(batch_items[chunk_index]);
                items.push_back(std::move(item));
            }
        } catch (const std::exception & ex) {
            request_errors[request_index] = ex.what();
        }
    }

    // Slots run in lockstep until the longest one finishes, so batching
    // similar lengths together is what keeps the padding cheap. Prompts do not
    // share state across slots, so length is the only sort key.
    std::vector<size_t> order(items.size());
    for (size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return items[lhs].text_length < items[rhs].text_length;
    });

    debug::trace_log_scalar("qwen3_tts.run_batch.requests", static_cast<int64_t>(requests.size()));
    debug::trace_log_scalar("qwen3_tts.run_batch.chunks", static_cast<int64_t>(items.size()));
    debug::trace_log_scalar("qwen3_tts.run_batch.max_batch", max_batch_size_);

    std::vector<std::vector<runtime::AudioBuffer>> chunk_audio(requests.size());
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        chunk_audio[request_index].resize(chunk_counts[request_index]);
    }

    // Split into equally sized batches rather than filling each to the limit.
    // A trailing remainder batch is the expensive case: cost per step grows far
    // more slowly than the batch, so 5+5 beats 8+2 for ten chunks.
    if (!order.empty()) {
        const size_t batch_limit = static_cast<size_t>(max_batch_size_);
        const size_t batch_count = (order.size() + batch_limit - 1) / batch_limit;
        const size_t base_size = order.size() / batch_count;
        const size_t remainder = order.size() % batch_count;
        for (size_t index = 0, batch_index = 0; index < order.size(); ++batch_index) {
            const size_t start = index;
            const size_t end = start + base_size + (batch_index < remainder ? 1 : 0);
            index = end;
            std::vector<Qwen3TalkerBatchItem> batch;
            batch.reserve(end - start);
            for (size_t slot = start; slot < end; ++slot) {
                batch.push_back(items[order[slot]].item);
            }
            std::vector<Qwen3TalkerCodes> batch_codes;
            try {
                batch_codes = talker_step_->generate_batch(batch);
            } catch (const std::exception & ex) {
                // Backstop for per-item validation that only the talker can do
                // (assembled prompt over capacity, say). The failure is
                // attributed to every request in this sub-batch, but other
                // sub-batches keep their results.
                for (size_t slot = start; slot < end; ++slot) {
                    auto & error_slot = request_errors[items[order[slot]].request_index];
                    if (error_slot.empty()) {
                        error_slot = ex.what();
                    }
                }
                continue;
            }
            if (batch_codes.size() != batch.size()) {
                throw std::runtime_error("Qwen3 TTS batch generation returned the wrong result count");
            }
            for (size_t slot = start; slot < end; ++slot) {
                const auto & item = items[order[slot]];
                auto & error_slot = request_errors[item.request_index];
                if (!error_slot.empty()) {
                    // The request already failed on an earlier chunk; decoding
                    // more of its audio would be thrown away at assembly.
                    continue;
                }
                try {
                    chunk_audio[item.request_index][item.chunk_index] =
                        decode_batch_codes(item.item, batch_codes[slot - start]);
                } catch (const std::exception & ex) {
                    error_slot = "chunk " + std::to_string(item.chunk_index + 1) + " of " +
                        std::to_string(chunk_counts[item.request_index]) + ": " + ex.what();
                }
            }
        }
    }

    std::vector<runtime::BatchedTaskResult> out;
    out.reserve(requests.size());
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        runtime::BatchedTaskResult entry;
        if (!request_errors[request_index].empty()) {
            entry.error = std::move(request_errors[request_index]);
            out.push_back(std::move(entry));
            continue;
        }
        runtime::AudioBuffer merged;
        for (auto & chunk : chunk_audio[request_index]) {
            runtime::append_audio_buffer(merged, std::move(chunk));
        }
        entry.result.audio_output = std::move(merged);
        out.push_back(std::move(entry));
    }
    if (mem_saver_) {
        // A size-1 sub-batch delegates to the single-sequence path, which
        // retains its cached step graph; honor mem_saver here like run() does.
        talker_step_->release_cached_step_graph();
    }
    debug::timing_log_scalar("session.batch_wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return out;
}

void Qwen3TTSSession::write_speaker_embedding(
    const std::string & path,
    const Qwen3SpeakerEmbedding & embedding) const {
    std::FILE * out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) {
        throw std::runtime_error("cannot write speaker_embedding_out: " + path);
    }
    std::fwrite(embedding.values.data(), sizeof(float), embedding.values.size(), out);
    std::fclose(out);
}

// Returns the cloned-voice embedding for a reference clip, encoding it at most
// once per distinct clip. The encoder runs on the CPU and costs hundreds of
// milliseconds, so on a server that keeps reusing the same voices this cache
// turns the dominant per-request cost into a hash lookup. Returned by value:
// an embedding is ~8 KB, and a copy dodges every eviction-lifetime question.
Qwen3SpeakerEmbedding Qwen3TTSSession::resolve_custom_voice_embedding(
    const runtime::AudioBuffer & reference_audio) {
    if (speaker_encoder_ == nullptr) {
        throw std::runtime_error(
            "Qwen3 custom voice reference audio needs speaker encoder weights - "
            "this checkpoint has none");
    }
    SpeakerEmbeddingCacheKey key;
    key.sample_rate = reference_audio.sample_rate;
    key.channels = reference_audio.channels;
    key.sample_count = static_cast<uint64_t>(reference_audio.samples.size());
    key.sample_hash = hash_audio_samples(reference_audio);
    if (const auto * cached = speaker_embedding_cache_.find(key)) {
        debug::trace_log_scalar("qwen3_tts.speaker_embedding_cache.hit", 1);
        debug::trace_log_scalar(
            "qwen3_tts.speaker_embedding_cache.entries",
            static_cast<int64_t>(speaker_embedding_cache_.size()));
        return cached->embedding;
    }
    debug::trace_log_scalar("qwen3_tts.speaker_embedding_cache.hit", 0);
    const auto encode_start = Clock::now();
    SpeakerEmbeddingCacheEntry entry;
    entry.embedding = speaker_encoder_->encode(reference_audio);
    debug::timing_log_scalar(
        "qwen3_tts.speaker_embedding_encode_ms",
        engine::debug::elapsed_ms(encode_start, Clock::now()));
    auto embedding = entry.embedding;
    speaker_embedding_cache_.put(std::move(key), std::move(entry));
    debug::trace_log_scalar(
        "qwen3_tts.speaker_embedding_cache.entries",
        static_cast<int64_t>(speaker_embedding_cache_.size()));
    return embedding;
}

// Mirrors the per-item validation that build_prompt_state and generate_batch
// perform later, so a bad speaker, language, or token budget fails its own
// request during item building instead of throwing out of the shared batched
// generation and taking its batch neighbours down with it.
void Qwen3TTSSession::validate_batch_item(
    const Qwen3TalkerPrefill & prefill,
    const Qwen3TTSGenerationOptions & options) const {
    const auto & config = assets_->config.talker;
    if (options.max_new_tokens > assets_->config.max_new_tokens) {
        throw std::runtime_error("Qwen3 TTS max_tokens exceeds the model generation capacity");
    }
    std::string language = ascii_lower(prefill.language);
    if (prefill.prompt_mode == Qwen3TalkerPromptMode::CustomVoice && !prefill.speaker_embedding.has_value()) {
        const std::string speaker = ascii_lower(prefill.speaker);
        const auto speaker_it = config.speaker_id.find(speaker);
        if (speaker_it == config.speaker_id.end()) {
            throw std::runtime_error("Qwen3 custom voice unsupported speaker: " + prefill.speaker);
        }
        const auto dialect_it = config.speaker_dialect.find(speaker);
        if (dialect_it == config.speaker_dialect.end()) {
            throw std::runtime_error("Qwen3 custom voice missing dialect entry for speaker: " + prefill.speaker);
        }
        if ((language == "chinese" || language == "auto") && dialect_it->second.has_value()) {
            language = *dialect_it->second;
        }
    }
    if (language != "auto" && config.codec_language_id.find(language) == config.codec_language_id.end()) {
        throw std::runtime_error("Qwen3 talker unsupported language: " + prefill.language);
    }
}

std::vector<Qwen3TalkerBatchItem> Qwen3TTSSession::build_batch_items(const runtime::TaskRequest & request) {
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    const auto embedding_out_path = runtime::find_option(request.options, {"speaker_embedding_out"});
    std::vector<Qwen3TalkerBatchItem> items;
    items.reserve(chunk_requests.size());
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3TTSVoiceDesignPromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            Qwen3TalkerBatchItem item;
            item.prefill = prompt_builder.build_prefill(qwen_request);
            item.options = qwen_request.generation;
            item.repetition_penalty = qwen_request.generation.repetition_penalty;
            validate_batch_item(item.prefill, item.options);
            items.push_back(std::move(item));
        }
        return items;
    }
    const bool custom_voice_icl_active =
        assets_->config.variant == Qwen3TTSVariant::CustomVoice &&
        custom_voice_icl_ && speech_encoder_ != nullptr && speaker_encoder_ != nullptr &&
        make_request(chunk_requests.front()).voice_clone.has_value();
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && !custom_voice_icl_active) {
        Qwen3TTSCustomVoicePromptBuilder prompt_builder(
            text_tokenizer_,
            speaker_encoder_.get(),
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        // A cloned voice resolves through the embedding cache once per
        // request; every chunk reuses the vector, and a voice the session has
        // seen before skips the speaker encoder entirely.
        std::optional<Qwen3SpeakerEmbedding> cloned_embedding;
        {
            const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
            if (first_request.custom_voice.has_value() &&
                first_request.custom_voice->reference_audio.has_value() &&
                speaker_encoder_ != nullptr) {
                cloned_embedding = resolve_custom_voice_embedding(*first_request.custom_voice->reference_audio);
            }
        }
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            Qwen3TalkerBatchItem item;
            item.prefill = prompt_builder.build_prefill(
                qwen_request,
                cloned_embedding.has_value() ? &*cloned_embedding : nullptr);
            if (items.empty() && embedding_out_path.has_value()) {
                if (!item.prefill.speaker_embedding.has_value()) {
                    throw std::runtime_error(
                        "speaker_embedding_out requires reference audio (voice_ref)");
                }
                write_speaker_embedding(*embedding_out_path, *item.prefill.speaker_embedding);
            }
            item.options = qwen_request.generation;
            item.repetition_penalty = qwen_request.generation.repetition_penalty;
            validate_batch_item(item.prefill, item.options);
            items.push_back(std::move(item));
        }
        return items;
    }
    const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error("Qwen3 base TTS requires voice clone reference audio");
    }
    if (speech_encoder_ == nullptr || speaker_encoder_ == nullptr) {
        throw std::runtime_error("Qwen3 base TTS session is missing voice clone runtimes");
    }
    Qwen3TTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        *speech_encoder_,
        *speaker_encoder_,
        assets_->config.talker.max_position_embeddings);
    // Resolved once per request and consumed before the next request resolves;
    // the returned reference points into a slot cache a later resolve may evict.
    const auto & voice_prompt = resolve_voice_prompt(*first_request.voice_clone, prompt_builder);
    if (embedding_out_path.has_value()) {
        write_speaker_embedding(*embedding_out_path, voice_prompt.speaker_embedding);
    }
    if (!voice_prompt.reference_codes.has_value()) {
        throw std::runtime_error("Qwen3 base TTS talker currently requires ICL reference codes");
    }
    for (const auto & chunk_request : chunk_requests) {
        const Qwen3TTSRequest qwen_request = make_request(chunk_request);
        Qwen3TalkerBatchItem item;
        item.prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        if (qwen_request.custom_voice.has_value() && !qwen_request.custom_voice->instruct.empty()) {
            item.prefill.instruct_ids =
                text_tokenizer_.encode(text_tokenizer_.build_instruct_prompt(qwen_request.custom_voice->instruct));
        }
        item.options = qwen_request.generation;
        item.repetition_penalty = qwen_request.generation.repetition_penalty;
        validate_batch_item(item.prefill, item.options);
        items.push_back(std::move(item));
    }
    return items;
}

runtime::AudioBuffer Qwen3TTSSession::decode_batch_codes(
    const Qwen3TalkerBatchItem & item,
    const Qwen3TalkerCodes & codes) {
    // ICL prompts (Base always, CustomVoice with custom_voice_icl) decode the
    // reference together with the generation and trim it off afterwards.
    if (item.prefill.reference_codes.has_value()) {
        return speech_decoder_->decode_and_trim_reference(
            *item.prefill.reference_codes,
            codes.generated_codes);
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        throw std::runtime_error("Qwen3 base TTS batch decode requires reference codes");
    }
    return speech_decoder_->decode(codes.generated_codes);
}

const Qwen3VoiceClonePrompt & Qwen3TTSSession::resolve_voice_prompt(
    const Qwen3VoiceCloneInput & input,
    const Qwen3TTSVoiceClonePromptBuilder & prompt_builder) {
    const uint64_t sample_count = static_cast<uint64_t>(input.reference_audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(input.reference_audio);
    VoicePromptCacheKey key;
    key.reference_text = input.reference_text;
    key.mode = input.mode;
    key.sample_rate = input.reference_audio.sample_rate;
    key.channels = input.reference_audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    if (auto * cached = voice_prompt_cache_.find(key)) {
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 1);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return cached->prompt;
    }

    VoicePromptCacheEntry entry;
    entry.prompt = prompt_builder.build_voice_prompt(input);
    if (voice_prompt_cache_.capacity() == 0) {
        uncached_voice_prompt_ = std::move(entry);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return uncached_voice_prompt_->prompt;
    }
    const bool will_evict = voice_prompt_cache_.size() >= voice_prompt_cache_.capacity();
    voice_prompt_cache_.put(std::move(key), std::move(entry));
    auto * cached = voice_prompt_cache_.find(VoicePromptCacheKey{
        input.reference_text,
        input.mode,
        input.reference_audio.sample_rate,
        input.reference_audio.channels,
        sample_count,
        sample_hash,
    });
    if (cached == nullptr) {
        throw std::runtime_error("Qwen3 TTS voice prompt cache insert failed");
    }
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", will_evict ? 1 : 0);
    return cached->prompt;
}

Qwen3TTSRequest Qwen3TTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Qwen3 TTS requires text input");
    }
    Qwen3TTSRequest out;
    out.text = request.text_input->text;
    out.language = !request.text_input->language.empty() ? request.text_input->language : "Auto";
    out.generation = generation_options_from_request(request, assets_->config);
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        const runtime::AudioBuffer * reference_audio = nullptr;
        if (request.voice.has_value()
            && request.voice->speaker.has_value()
            && request.voice->speaker->audio.has_value()) {
            reference_audio = &*request.voice->speaker->audio;
        } else if (request.audio_input.has_value()) {
            reference_audio = &*request.audio_input;
        }
        if (reference_audio != nullptr) {
            Qwen3VoiceCloneInput voice_clone;
            voice_clone.reference_audio = *reference_audio;
            if (const auto reference_text = runtime::find_option(
                    request.options,
                    {"reference_text"})) {
                voice_clone.reference_text = *reference_text;
            }
            bool x_vector_only = false;
            if (const auto value = runtime::find_option(
                    request.options,
                    {"x_vector_only_mode"})) {
                x_vector_only = runtime::parse_bool_option(*value, "x_vector_only_mode");
            }
            voice_clone.mode = x_vector_only
                ? Qwen3VoiceCloneMode::SpeakerEmbeddingOnly
                : Qwen3VoiceCloneMode::Icl;
            out.voice_clone = std::move(voice_clone);
        }
    } else if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3VoiceDesignInput voice_design;
        voice_design.instruct = runtime::find_option(
            request.options,
            {"instruct"})
            .value_or("");
        if (voice_design.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                voice_design.instruct = tag->second;
            }
        }
        out.voice_design = std::move(voice_design);
    } else if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        Qwen3CustomVoiceInput custom_voice;
        custom_voice.speaker = runtime::find_option(request.options, {"speaker"}).value_or("");
        if (custom_voice.speaker.empty() && request.voice.has_value() && request.voice->speaker.has_value()) {
            custom_voice.speaker = request.voice->speaker->cached_voice_id.value_or("");
        }
        if (request.voice.has_value() && request.voice->speaker.has_value()
            && request.voice->speaker->audio.has_value()) {
            custom_voice.reference_audio = *request.voice->speaker->audio;
        }
        custom_voice.instruct = runtime::find_option(
            request.options,
            {"instruct"})
            .value_or("");
        if (custom_voice.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                custom_voice.instruct = tag->second;
            }
        }
        // Experimental ICL cloning: with reference audio plus transcript the
        // request can run through the Base voice-clone prompt, which anchors
        // register and prosody on the actual reference codes instead of the
        // embedding row alone.
        if (custom_voice_icl_ && custom_voice.reference_audio.has_value()) {
            if (const auto reference_text = runtime::find_option(request.options, {"reference_text"})) {
                Qwen3VoiceCloneInput voice_clone;
                voice_clone.reference_audio = *custom_voice.reference_audio;
                voice_clone.reference_text = *reference_text;
                voice_clone.mode = Qwen3VoiceCloneMode::Icl;
                out.voice_clone = std::move(voice_clone);
            }
        }
        out.custom_voice = std::move(custom_voice);
    }
    return out;
}

}  // namespace engine::models::qwen3_tts
