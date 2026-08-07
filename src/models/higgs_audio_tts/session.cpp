#include "engine/models/higgs_audio_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kDefaultTextChunkSize = 1024;
constexpr int64_t kDefaultReferenceCacheSlots = 1;
constexpr int64_t kDefaultMaxBatchSize = 1;

// ggml's CUDA vector-matmul kernels cover a batch of at most 8 -- see
// MMVQ_MAX_BATCH_SIZE in ggml-cuda/mmvq.cuh and MMVF_MAX_BATCH_SIZE in
// mmvf.cuh, which apply to quantized and float weights respectively. Past that
// the backend switches to the tiled path built for prefill-sized batches, which
// for one token per sequence means re-materializing the weights every step.
// Measured on an RTX 4000 Ada with the Q8 package: 2.6 ms per sequence-token at
// batch 8, 11.2 ms at batch 9. Going above 8 is never faster, so clamp.
constexpr int64_t kMaxCudaBatchSize = 8;

void validate_matmul_weight_storage(assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
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

std::size_t resolve_reference_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"higgs_audio_tts.reference_cache_slots", "reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("higgs_audio_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("higgs_audio_tts.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

int64_t resolve_max_batch_size(const runtime::SessionOptions & options) {
    const int64_t size = runtime::parse_i64_option(
        options.options,
        {"higgs_audio_tts.max_batch", "max_batch"})
        .value_or(kDefaultMaxBatchSize);
    if (size < 1) {
        throw std::runtime_error("higgs_audio_tts.max_batch must be at least 1");
    }
    return size;
}

const runtime::AudioBuffer * find_reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value()
        && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    if (request.audio_input.has_value()) {
        return &*request.audio_input;
    }
    return nullptr;
}

HiggsGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const HiggsConfig & config) {
    HiggsGenerationOptions options;
    options.max_tokens = 2048;
    options.temperature = 0.8F;
    options.top_p = 0.8F;
    options.top_k = 30;
    options.repetition_penalty = 1.1F;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("Higgs TTS max_tokens must be non-negative");
        }
        if (*value > 0) {
            options.max_tokens = *value;
        }
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        options.seed = *value;
    }
    if (options.max_tokens > config.text.max_position_embeddings) {
        throw std::runtime_error("Higgs TTS max_tokens exceeds model max_position_embeddings");
    }
    if (!(options.repetition_penalty > 0.0F) || !std::isfinite(options.repetition_penalty)) {
        throw std::runtime_error("Higgs TTS repetition_penalty must be finite and positive");
    }
    return options;
}

}  // namespace

HiggsTTSSession::HiggsTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HiggsAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      reference_cache_(resolve_reference_cache_slots(this->options())),
      max_batch_size_(resolve_max_batch_size(this->options())) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs TTS session requires assets");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Higgs TTS currently supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Higgs TTS only supports the Tts task");
    }
    ar_weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.ar_weight_context_mb"}, ar_weight_context_bytes_);
    codec_weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_weight_context_mb"}, codec_weight_context_bytes_);
    ar_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.ar_decode_graph_arena_mb"}, ar_decode_graph_arena_bytes_);
    codec_decode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_decode_graph_arena_mb"}, codec_decode_graph_arena_bytes_);
    codec_encode_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"higgs_audio_tts.codec_encode_graph_arena_mb"}, codec_encode_graph_arena_bytes_);

    if (const auto it = options.options.find("higgs_audio_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "higgs_audio_tts.weight_type");
        ar_weight_storage_type_ = storage_type;
        codec_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("higgs_audio_tts.ar_weight_type"); it != options.options.end()) {
        ar_weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(ar_weight_storage_type_, "higgs_audio_tts.ar_weight_type");
    }
    if (const auto it = options.options.find("higgs_audio_tts.codec_weight_type"); it != options.options.end()) {
        codec_weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(codec_weight_storage_type_, "higgs_audio_tts.codec_weight_type");
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("higgs_audio_tts.", 0) == 0 &&
            key != "higgs_audio_tts.ar_weight_context_mb" &&
            key != "higgs_audio_tts.codec_weight_context_mb" &&
            key != "higgs_audio_tts.ar_decode_graph_arena_mb" &&
            key != "higgs_audio_tts.codec_decode_graph_arena_mb" &&
            key != "higgs_audio_tts.codec_encode_graph_arena_mb" &&
            key != "higgs_audio_tts.reference_cache_slots" &&
            key != "higgs_audio_tts.max_batch" &&
            key != "higgs_audio_tts.weight_type" &&
            key != "higgs_audio_tts.ar_weight_type" &&
            key != "higgs_audio_tts.codec_weight_type") {
            throw std::runtime_error("unknown Higgs TTS session option: " + key);
        }
    }

    ar_ = std::make_shared<HiggsARRuntime>(
        assets_,
        execution_context(),
        ar_weight_context_bytes_,
        ar_weight_storage_type_);
    codec_ = std::make_shared<HiggsCodecRuntime>(
        assets_,
        execution_context(),
        codec_weight_context_bytes_,
        codec_decode_graph_arena_bytes_,
        codec_encode_graph_arena_bytes_,
        codec_weight_storage_type_);
    generator_ = std::make_unique<HiggsGenerator>(
        assets_,
        ar_,
        codec_,
        ar_decode_graph_arena_bytes_);

    if (ar_->backend_type() == core::BackendType::Cuda && max_batch_size_ > kMaxCudaBatchSize) {
        debug::log_message(
            debug::LogLevel::Warning,
            "higgs_audio_tts",
            "higgs_audio_tts.max_batch=" + std::to_string(max_batch_size_) +
                " exceeds the CUDA vector-matmul batch limit of " +
                std::to_string(kMaxCudaBatchSize) +
                "; clamping, because larger batches fall back to a much slower kernel path");
        max_batch_size_ = kMaxCudaBatchSize;
    }
}

std::string HiggsTTSSession::family() const {
    return "higgs_audio_tts";
}

runtime::VoiceTaskKind HiggsTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HiggsTTSSession::run_mode() const {
    return task_.mode;
}

void HiggsTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.text.has_value()) {
        runtime::TaskRequest task_request;
        task_request.text_input = request.text;
        task_request.voice = request.voice;
        task_request.options = request.options;
        const auto generation_request = make_generation_request(task_request);
        generator_->prepare(generation_request);
    }
    mark_prepared();
}

runtime::TaskResult HiggsTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Higgs TTS run");
    const auto wall_start = Clock::now();
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
    const std::string reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
    const auto * reference_audio = find_reference_audio(request);
    const HiggsCodecEncodeOutput * reference_codes =
        reference_audio != nullptr ? &resolve_reference_codes(*reference_audio, reference_text) : nullptr;
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_size", text_chunk_size);
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    debug::trace_log_scalar("higgs_audio_tts.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));

    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const auto generation_request = make_generation_request(chunk_request, reference_codes);
        auto result = generator_->generate(generation_request);
        runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{
            result.audio.sample_rate,
            result.audio.channels,
            std::move(result.audio.values),
        });
    }

    runtime::TaskResult out;
    out.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return out;
}

int64_t HiggsTTSSession::max_batch_size() const {
    return max_batch_size_;
}

std::vector<runtime::TaskResult> HiggsTTSSession::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    require_prepared("Higgs TTS run_batch");
    if (requests.empty()) {
        throw std::runtime_error("Higgs TTS run_batch requires at least one request");
    }
    if (max_batch_size_ <= 1 || requests.size() == 1) {
        std::vector<runtime::TaskResult> out;
        out.reserve(requests.size());
        for (const auto & request : requests) {
            out.push_back(run(request));
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
        uint64_t voice_key = 0;
        size_t text_length = 0;
        HiggsGenerationRequest generation;
    };
    std::vector<ChunkItem> items;
    std::vector<size_t> chunk_counts(requests.size(), 0);
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const auto & request = requests[request_index];
        const int64_t text_chunk_size =
            engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
        const auto text_chunk_mode =
            engine::text::parse_text_chunk_mode_override(request.options)
                .value_or(engine::text::TextChunkMode::Default);
        const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
        const std::string reference_text =
            runtime::find_option(request.options, {"reference_text"}).value_or("");
        const auto * reference_audio = find_reference_audio(request);
        // Resolved here and consumed immediately: the returned reference points
        // into a slot cache that a later resolve may evict.
        const HiggsCodecEncodeOutput * reference_codes =
            reference_audio != nullptr ? &resolve_reference_codes(*reference_audio, reference_text) : nullptr;
        uint64_t voice_key = 1469598103934665603ull;
        voice_key = fnv1a_mix(voice_key, reference_text.data(), reference_text.size());
        if (reference_codes != nullptr && !reference_codes->codes.empty()) {
            voice_key = fnv1a_mix(
                voice_key,
                reference_codes->codes.data(),
                reference_codes->codes.size() * sizeof(int32_t));
        }
        chunk_counts[request_index] = chunk_requests.size();
        for (size_t chunk_index = 0; chunk_index < chunk_requests.size(); ++chunk_index) {
            ChunkItem item;
            item.request_index = request_index;
            item.chunk_index = chunk_index;
            item.voice_key = voice_key;
            item.generation = make_generation_request(chunk_requests[chunk_index], reference_codes);
            item.text_length = item.generation.text.size();
            items.push_back(std::move(item));
        }
    }

    // Group by voice so every slot in a batch shares a reference prompt, then
    // by length: slots run in lockstep until the longest one finishes, so
    // batching similar lengths together is what keeps the padding cheap.
    std::vector<size_t> order(items.size());
    for (size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (items[lhs].voice_key != items[rhs].voice_key) {
            return items[lhs].voice_key < items[rhs].voice_key;
        }
        return items[lhs].text_length < items[rhs].text_length;
    });

    debug::trace_log_scalar("higgs_audio_tts.run_batch.requests", static_cast<int64_t>(requests.size()));
    debug::trace_log_scalar("higgs_audio_tts.run_batch.chunks", static_cast<int64_t>(items.size()));
    debug::trace_log_scalar("higgs_audio_tts.run_batch.max_batch", max_batch_size_);

    std::vector<std::vector<runtime::AudioBuffer>> chunk_audio(requests.size());
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        chunk_audio[request_index].resize(chunk_counts[request_index]);
    }

    // Split into equally sized batches rather than filling each to the limit.
    // A trailing remainder batch is the expensive case: cost per step grows far
    // more slowly than the batch, so 5+5 beats 8+2 for ten requests.
    const size_t batch_limit = static_cast<size_t>(max_batch_size_);
    const size_t batch_count = (order.size() + batch_limit - 1) / batch_limit;
    const size_t base_size = order.size() / batch_count;
    const size_t remainder = order.size() % batch_count;
    for (size_t index = 0, batch_index = 0; index < order.size(); ++batch_index) {
        const size_t start = index;
        const size_t end = start + base_size + (batch_index < remainder ? 1 : 0);
        index = end;
        std::vector<HiggsGenerationRequest> batch;
        batch.reserve(end - start);
        for (size_t index = start; index < end; ++index) {
            batch.push_back(items[order[index]].generation);
        }
        auto batch_results = generator_->generate_batch(batch);
        if (batch_results.size() != batch.size()) {
            throw std::runtime_error("Higgs TTS batch generation returned the wrong result count");
        }
        for (size_t index = start; index < end; ++index) {
            auto & result = batch_results[index - start];
            const auto & item = items[order[index]];
            chunk_audio[item.request_index][item.chunk_index] = runtime::AudioBuffer{
                result.audio.sample_rate,
                result.audio.channels,
                std::move(result.audio.values),
            };
        }
    }

    std::vector<runtime::TaskResult> out;
    out.reserve(requests.size());
    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        runtime::AudioBuffer merged;
        for (auto & chunk : chunk_audio[request_index]) {
            runtime::append_audio_buffer(merged, std::move(chunk));
        }
        runtime::TaskResult result;
        result.audio_output = std::move(merged);
        out.push_back(std::move(result));
    }
    debug::timing_log_scalar("session.batch_wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return out;
}

const HiggsCodecEncodeOutput & HiggsTTSSession::resolve_reference_codes(
    const runtime::AudioBuffer & audio,
    const std::string & reference_text) {
    const uint64_t sample_count = static_cast<uint64_t>(audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(audio);
    ReferenceCacheKey key;
    key.reference_text = reference_text;
    key.sample_rate = audio.sample_rate;
    key.channels = audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    debug::trace_log_scalar("higgs_audio_tts.reference_audio.sample_rate", audio.sample_rate);
    debug::trace_log_scalar("higgs_audio_tts.reference_audio.channels", audio.channels);
    debug::trace_log_f32("higgs_audio_tts.reference_audio.samples",
                         {static_cast<int64_t>(audio.samples.size())},
                         audio.samples);
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.capacity", static_cast<int64_t>(reference_cache_.capacity()));
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.size", static_cast<int64_t>(reference_cache_.size()));
    if (const auto * cached = reference_cache_.find(key)) {
        debug::trace_log_scalar("higgs_audio_tts.reference_cache.hit", 1);
        return cached->codes;
    }
    debug::trace_log_scalar("higgs_audio_tts.reference_cache.hit", 0);

    const auto encode_start = Clock::now();
    ReferenceCacheEntry entry;
    entry.codes = codec_->encode_reference(audio);
    codec_->release_encode_graph();
    debug::trace_log_scalar("higgs_audio_tts.reference_codes.frames", entry.codes.frames);
    debug::trace_log_scalar("higgs_audio_tts.reference_codes.codebooks", entry.codes.codebooks);
    debug::trace_log_i32("higgs_audio_tts.reference_codes.values",
                         {entry.codes.frames, entry.codes.codebooks},
                         entry.codes.codes);
    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(entry);
        debug::timing_log_scalar("higgs_audio_tts.codec.encode_reference_ms", engine::debug::elapsed_ms(encode_start));
        return uncached_reference_->codes;
    }
    reference_cache_.put(key, std::move(entry));
    debug::timing_log_scalar("higgs_audio_tts.codec.encode_reference_ms", engine::debug::elapsed_ms(encode_start));
    return reference_cache_.find(key)->codes;
}

HiggsGenerationRequest HiggsTTSSession::make_generation_request(
    const runtime::TaskRequest & request,
    const HiggsCodecEncodeOutput * resolved_reference_codes) {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Higgs TTS requires text input");
    }
    const std::string reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");

    HiggsGenerationRequest out;
    out.text = request.text_input->text;
    out.reference_text = reference_text;
    out.options = generation_options_from_request(request, assets_->config);
    if (resolved_reference_codes != nullptr) {
        out.reference_codes = resolved_reference_codes->codes;
        out.reference_frames = resolved_reference_codes->frames;
        out.reference_codebooks = resolved_reference_codes->codebooks;
    } else if (const auto * reference_audio = find_reference_audio(request)) {
        const auto & reference_codes = resolve_reference_codes(*reference_audio, reference_text);
        out.reference_codes = reference_codes.codes;
        out.reference_frames = reference_codes.frames;
        out.reference_codebooks = reference_codes.codebooks;
    }
    return out;
}

}  // namespace engine::models::higgs_audio_tts
