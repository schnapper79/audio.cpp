#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::qwen3_tts {

enum class Qwen3TalkerPromptMode {
    VoiceClone,
    VoiceDesign,
    CustomVoice,
};

struct Qwen3TalkerPrefill {
    Qwen3TalkerPromptMode prompt_mode = Qwen3TalkerPromptMode::VoiceClone;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> instruct_ids;
    std::vector<int32_t> reference_ids;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::optional<Qwen3SpeakerEmbedding> speaker_embedding = std::nullopt;
    std::string speaker;
    std::string language = "Auto";
    bool icl_mode = false;
    bool x_vector_only_mode = false;
    // ICL layout only: how many copies of the speaker-embedding row the
    // prompt carries. One row is the trained layout; more rows shift the
    // identity contest toward the embedding when it deliberately differs
    // from the reference codes (hybrid cloning: timbre from the embedding,
    // accent/delivery from the codes).
    int64_t speaker_embedding_repeat = 1;
    // Register priming: these frames are force-fed as the start of the
    // generation instead of being sampled, so every chunk begins in the same
    // acoustic state (register, energy) before free generation continues. The
    // corresponding carrier text must be part of input_ids; the caller trims
    // the primed frames from the decoded audio.
    std::optional<Qwen3SpeechCodes> primer_codes = std::nullopt;
};

struct Qwen3TalkerCodes {
    Qwen3SpeechCodes generated_codes;
    Qwen3SpeechCodes decoder_input_codes;
};

// One sequence of a batched generation run. Options are per item, so requests
// with different sampling settings or token limits can share a batch.
struct Qwen3TalkerBatchItem {
    Qwen3TalkerPrefill prefill;
    Qwen3TTSGenerationOptions options;
    float repetition_penalty = 1.05F;
};

class Qwen3TalkerWeightsRuntime;
class Qwen3TalkerStepRuntime;

class Qwen3TalkerStepRuntime {
public:
    class Impl;
    explicit Qwen3TalkerStepRuntime(std::unique_ptr<Impl> impl);
    ~Qwen3TalkerStepRuntime();

    Qwen3TalkerCodes generate(
        const Qwen3TalkerPrefill & prefill,
        const Qwen3TTSGenerationOptions & options,
        float repetition_penalty = 1.05F);
    // Decodes all items in one batched AR pass; results are in item order.
    // A sequence that reaches max_new_tokens is truncated, not failed, exactly
    // like the single path. Throws on malformed items (prompt over capacity),
    // because no retry would help.
    std::vector<Qwen3TalkerCodes> generate_batch(const std::vector<Qwen3TalkerBatchItem> & items);
    int64_t release_cached_step_graph();
    // Resolved backend type; differs from the requested one when the session
    // was created with BestAvailable.
    core::BackendType backend_type() const;

private:
    std::unique_ptr<Impl> impl_;
};

class Qwen3Talker {
public:
    explicit Qwen3Talker(Qwen3TTSTalkerConfig config);

    const Qwen3TTSTalkerConfig & config() const noexcept;

    std::shared_ptr<const Qwen3TalkerWeightsRuntime> create_weights_runtime(
        std::shared_ptr<const Qwen3TTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t graph_arena_bytes,
        size_t talker_constant_context_bytes,
        size_t code_predictor_constant_context_bytes,
        engine::assets::TensorStorageType weight_storage_type,
        Qwen3TTSPerfMode perf_mode) const;

    std::shared_ptr<Qwen3TalkerStepRuntime> create_step_runtime(
        std::shared_ptr<const Qwen3TalkerWeightsRuntime> weights,
        int64_t prompt_capacity,
        int64_t generation_capacity) const;

private:
    Qwen3TTSTalkerConfig config_;
};

}  // namespace engine::models::qwen3_tts
