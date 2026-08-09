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

// Measured on the v3 4B Q8 checkpoint (long-form English, ASR-verified word
// accuracy across seeds): chunks up to 512 characters generate the full text
// (WER <= 2%); at 768 the model reliably drops ~10% of the words, and at 1024
// every seed either skips whole sentences or rambles (WER 14-27%). The failure
// is silent - audio sounds fine, content is just missing - so the default
// stays below the cliff.
constexpr int64_t kDefaultTextChunkSize = 512;
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

bool resolve_tail_cleanup(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(
            options.options,
            {"higgs_audio_tts.tail_cleanup", "tail_cleanup"})) {
        return runtime::parse_bool_option(*value, "higgs_audio_tts.tail_cleanup");
    }
    return true;
}

int64_t resolve_tail_cleanup_fade_ms(const runtime::SessionOptions & options) {
    const int64_t fade_ms = runtime::parse_i64_option(
        options.options,
        {"higgs_audio_tts.tail_cleanup_fade_ms"})
        .value_or(80);
    if (fade_ms < 0 || fade_ms > 500) {
        throw std::runtime_error("higgs_audio_tts.tail_cleanup_fade_ms must be in [0, 500]");
    }
    return fade_ms;
}

// The model sometimes emits a breath-like noise burst after the last word and
// before end-of-audio: speech decays, then an unvoiced high-frequency burst
// RISES again for a few codec frames, then silence. A legitimate word ending -
// including a final fricative - decays monotonically, so an energy rise after
// the last voiced window is the discriminator. The burst and everything after
// it are cut, a short fade and a natural pause are kept.
void trim_trailing_noise_burst(runtime::AudioBuffer & audio, int64_t fade_ms) {
    if (audio.sample_rate <= 0 || audio.channels != 1 || audio.samples.size() < 4096) {
        return;
    }
    const int sr = audio.sample_rate;
    const int win = sr / 50;   // 20 ms
    const int hop = sr / 100;  // 10 ms
    const int64_t total = static_cast<int64_t>(audio.samples.size());
    // Only the final stretch of the chunk is examined.
    const int64_t analysis_start = std::max<int64_t>(0, total - static_cast<int64_t>(sr) * 3 / 2);
    struct Window {
        int64_t start = 0;
        float rms = 0.0F;
        float zcr = 0.0F;
        bool voiced = false;
    };
    std::vector<Window> windows;
    const int lag_min = sr / 400;
    const int lag_max = sr / 60;
    for (int64_t start = analysis_start; start + win <= total; start += hop) {
        Window w;
        w.start = start;
        double energy = 0.0;
        int crossings = 0;
        for (int i = 0; i < win; ++i) {
            const float s = audio.samples[static_cast<size_t>(start + i)];
            energy += static_cast<double>(s) * s;
            if (i > 0 && (s >= 0.0F) != (audio.samples[static_cast<size_t>(start + i - 1)] >= 0.0F)) {
                ++crossings;
            }
        }
        w.rms = static_cast<float>(std::sqrt(energy / win));
        w.zcr = static_cast<float>(crossings) / static_cast<float>(win);
        if (w.rms > 0.008F) {
            // Autocorrelation peak in the speech F0 range marks voicing.
            double mean = 0.0;
            for (int i = 0; i < win; ++i) {
                mean += audio.samples[static_cast<size_t>(start + i)];
            }
            mean /= win;
            double norm = 0.0;
            for (int i = 0; i < win; ++i) {
                const double v = audio.samples[static_cast<size_t>(start + i)] - mean;
                norm += v * v;
            }
            if (norm > 0.0) {
                double best = 0.0;
                for (int lag = lag_min; lag < lag_max && lag < win; ++lag) {
                    double acc = 0.0;
                    for (int i = 0; i + lag < win; ++i) {
                        const double a = audio.samples[static_cast<size_t>(start + i)] - mean;
                        const double b = audio.samples[static_cast<size_t>(start + i + lag)] - mean;
                        acc += a * b;
                    }
                    best = std::max(best, acc / norm);
                }
                w.voiced = best > 0.4;
            }
        }
        windows.push_back(w);
    }
    // Last voiced window inside the analysis span.
    int64_t last_voiced = -1;
    for (int64_t index = static_cast<int64_t>(windows.size()) - 1; index >= 0; --index) {
        if (windows[static_cast<size_t>(index)].voiced) {
            last_voiced = index;
            break;
        }
    }
    if (last_voiced < 0) {
        return;
    }
    // The unvoiced tail after the last voiced window may only decay. Two burst
    // signatures are cut: a sharp energy rise over the tracked envelope, and a
    // high-zero-crossing noise segment whose energy peaks after its own onset
    // or above the final speech level - a legitimate word-final fricative
    // starts at its loudest and decays, it never peaks later. The noise-segment
    // rule additionally requires the voice to have faded already: a real
    // final fricative attaches to full-level speech, the artifact only appears
    // once the voice has decayed toward silence.
    float speech_rms = 0.0F;
    {
        std::vector<float> voiced_rms;
        for (const auto & w : windows) {
            if (w.voiced) {
                voiced_rms.push_back(w.rms);
            }
        }
        if (voiced_rms.size() < 3) {
            return;
        }
        std::nth_element(voiced_rms.begin(), voiced_rms.begin() + voiced_rms.size() / 2, voiced_rms.end());
        speech_rms = voiced_rms[voiced_rms.size() / 2];
    }
    const float last_voiced_rms = windows[static_cast<size_t>(last_voiced)].rms;
    const bool voice_faded = last_voiced_rms < speech_rms * 0.4F;
    float envelope = last_voiced_rms;
    int64_t cut_window = -1;
    int64_t noise_onset = -1;
    float noise_onset_rms = 0.0F;
    float noise_peak_rms = 0.0F;
    for (int64_t index = last_voiced + 1; index < static_cast<int64_t>(windows.size()); ++index) {
        const auto & w = windows[static_cast<size_t>(index)];
        if (w.rms > envelope * 1.5F && w.rms > 0.006F) {
            cut_window = noise_onset >= 0 ? noise_onset : index;
            break;
        }
        envelope = std::max(w.rms, envelope * 0.85F);
        if (noise_onset < 0) {
            if (w.zcr > 0.3F && w.rms > 0.004F) {
                noise_onset = index;
                noise_onset_rms = w.rms;
                noise_peak_rms = w.rms;
            }
        } else if (w.zcr > 0.3F) {
            noise_peak_rms = std::max(noise_peak_rms, w.rms);
        }
    }
    if (cut_window < 0 && noise_onset >= 0 && voice_faded &&
        (noise_peak_rms > noise_onset_rms * 1.25F || noise_peak_rms > last_voiced_rms * 1.2F) &&
        noise_peak_rms > 0.006F) {
        cut_window = noise_onset;
    }
    if (cut_window < 0) {
        return;
    }
    // Fade out over the first fade_ms of the cut region instead of cutting
    // hard at its start: when a word-final fricative and the artifact are
    // fused, a hard cut would delete the fricative; the fade keeps it audible
    // as a short decaying /s/ while a pure noise burst shrinks from hundreds
    // of milliseconds to a quiet puff. fade_ms 0 restores the hard cut.
    const int64_t fade_begin = windows[static_cast<size_t>(cut_window)].start;
    const int64_t fade_len = std::min<int64_t>(std::max<int64_t>(sr * fade_ms / 1000, sr / 100), total - fade_begin);
    const int64_t cut_sample = fade_begin + fade_len;
    debug::trace_log_scalar("higgs_audio_tts.tail_cleanup.trimmed_ms",
                            (total - cut_sample) * 1000 / sr);
    for (int64_t i = fade_begin; i < cut_sample; ++i) {
        const float gain = static_cast<float>(cut_sample - i) / static_cast<float>(fade_len);
        audio.samples[static_cast<size_t>(i)] *= gain;
    }
    audio.samples.resize(static_cast<size_t>(cut_sample));
    audio.samples.insert(audio.samples.end(), static_cast<size_t>(sr / 8), 0.0F);
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
      max_batch_size_(resolve_max_batch_size(this->options())),
      tail_cleanup_(resolve_tail_cleanup(this->options())),
      tail_cleanup_fade_ms_(resolve_tail_cleanup_fade_ms(this->options())) {
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
            key != "higgs_audio_tts.tail_cleanup" &&
            key != "higgs_audio_tts.tail_cleanup_fade_ms" &&
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
        runtime::AudioBuffer chunk_audio{
            result.audio.sample_rate,
            result.audio.channels,
            std::move(result.audio.values),
        };
        if (tail_cleanup_) {
            trim_trailing_noise_burst(chunk_audio, tail_cleanup_fade_ms_);
        }
        runtime::append_audio_buffer(merged_audio, std::move(chunk_audio));
    }

    runtime::TaskResult out;
    out.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return out;
}

int64_t HiggsTTSSession::max_batch_size() const {
    return max_batch_size_;
}

std::vector<runtime::BatchedTaskResult> HiggsTTSSession::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    require_prepared("Higgs TTS run_batch");
    if (requests.empty()) {
        throw std::runtime_error("Higgs TTS run_batch requires at least one request");
    }
    if (max_batch_size_ <= 1 || requests.size() == 1) {
        // Batching disabled or nothing to amortize. Run each request on its own
        // and translate a throw into the same per-request error the batched path
        // reports, so callers only have to handle one shape of failure.
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
    // A request's audio is the concatenation of its chunks, so one failed chunk
    // fails that request -- but only that one. Its neighbours in the batch, and
    // every other request, still return their audio.
    std::vector<std::string> request_errors(requests.size());

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
            if (!result.error.empty()) {
                auto & slot = request_errors[item.request_index];
                if (slot.empty()) {
                    slot = "chunk " + std::to_string(item.chunk_index + 1) + " of " +
                        std::to_string(chunk_counts[item.request_index]) + ": " + result.error;
                }
                continue;
            }
            runtime::AudioBuffer slot_audio{
                result.audio.sample_rate,
                result.audio.channels,
                std::move(result.audio.values),
            };
            if (tail_cleanup_) {
                trim_trailing_noise_burst(slot_audio, tail_cleanup_fade_ms_);
            }
            chunk_audio[item.request_index][item.chunk_index] = std::move(slot_audio);
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
