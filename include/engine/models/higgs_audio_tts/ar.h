#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/higgs_audio_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::higgs_audio_tts {

struct HiggsQwenDecoderStackWeights {
    std::vector<modules::QwenDecoderLayerWeights> layers;
};

struct HiggsARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue modality_embedding;
    HiggsQwenDecoderStackWeights decoder;
    core::TensorValue norm;
    bool packed_qkv = false;
};

HiggsARWeights load_higgs_ar_weights(
    const HiggsAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class HiggsARRuntime {
public:
    HiggsARRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);

    const HiggsAssets & assets() const noexcept;
    const HiggsARWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int device() const noexcept;
    int threads() const noexcept;

private:
    std::shared_ptr<const HiggsAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int device_ = 0;
    int threads_ = 1;
    std::shared_ptr<const HiggsARWeights> weights_;
};

struct HiggsARDecodeInput {
    std::vector<int32_t> last_codes;
    bool use_last_codes = false;
};

struct HiggsARDecodeOutput {
    std::vector<float> codebook_logits;
};

struct HiggsARDecodeTiming {
    double input_upload_ms = 0.0;
    double mask_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    int64_t steps = 0;

    void add(const HiggsARDecodeTiming & other) noexcept;
};

struct HiggsARPrefillInput {
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> fused_code_ids;
    std::vector<float> text_gate;
    std::vector<float> code_gate;
    int64_t steps = 0;
};

struct HiggsARPrefillOutput {
    HiggsARDecodeOutput output;
    runtime::TransformerKVState kv_state;
    bool wrote_cache = false;
};

class HiggsARKVCache {
public:
    HiggsARKVCache(std::shared_ptr<HiggsARRuntime> runtime, int64_t cache_steps);
    ~HiggsARKVCache();

    HiggsARKVCache(const HiggsARKVCache &) = delete;
    HiggsARKVCache & operator=(const HiggsARKVCache &) = delete;

    bool can_run(const HiggsARRuntime & runtime, int64_t required_steps) const;
    int64_t cache_steps() const;
    int64_t valid_steps() const;
    int64_t current_end() const;
    void reset();
    void retain_prefix(int64_t prefix_steps);
    void import_state(const runtime::TransformerKVState & state);
    runtime::TransformerKVState export_state() const;
    void advance_after_direct_append(int64_t steps);
    const core::TensorValue & key_tensor(size_t layer) const;
    const core::TensorValue & value_tensor(size_t layer) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Batched decode inputs and outputs are slot-major: slot 0's values come first,
// then slot 1's, and so on.
struct HiggsARBatchDecodeInput {
    std::vector<int32_t> last_codes;
};

struct HiggsARBatchDecodeOutput {
    std::vector<float> codebook_logits;
};

// KV cache holding `slots` independent sequences.
//
// Sequences are stored left-padded: slot `b` with a prompt of `P_b` steps
// occupies cache steps [prompt_capacity - P_b, prompt_capacity), so every slot
// decodes at the same cache index and therefore at the same RoPE position.
// That alignment is what makes a single batched decode graph possible at all --
// ggml applies one position vector across the whole batch dimension. Shifting a
// sequence's positions by a constant leaves attention scores unchanged, because
// RoPE scores depend only on position differences.
class HiggsARBatchKVCache {
public:
    HiggsARBatchKVCache(std::shared_ptr<HiggsARRuntime> runtime, int64_t slots, int64_t cache_steps);
    ~HiggsARBatchKVCache();

    HiggsARBatchKVCache(const HiggsARBatchKVCache &) = delete;
    HiggsARBatchKVCache & operator=(const HiggsARBatchKVCache &) = delete;

    int64_t slots() const;
    int64_t cache_steps() const;
    const core::TensorValue & key_tensor(size_t layer) const;
    const core::TensorValue & value_tensor(size_t layer) const;

private:
    friend void copy_higgs_batch_kv_cache(
        HiggsARBatchKVCache & dst,
        const HiggsARBatchKVCache & src,
        int64_t steps);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Copies the first `steps` cache steps of every slot from `src` into `dst` on
// the device. Used to grow a batched cache without a host round trip.
void copy_higgs_batch_kv_cache(
    HiggsARBatchKVCache & dst,
    const HiggsARBatchKVCache & src,
    int64_t steps);

// Prefills one slot of a batched cache. Each slot is prefilled on its own
// because prompt lengths differ; only the decode loop runs batched, which is
// where the time goes.
class HiggsARBatchPrefillGraph {
public:
    HiggsARBatchPrefillGraph(
        std::shared_ptr<HiggsARRuntime> runtime,
        HiggsARBatchKVCache & cache,
        int64_t slot,
        int64_t prompt_steps,
        int64_t cache_offset,
        size_t graph_arena_bytes);
    ~HiggsARBatchPrefillGraph();

    HiggsARBatchPrefillGraph(const HiggsARBatchPrefillGraph &) = delete;
    HiggsARBatchPrefillGraph & operator=(const HiggsARBatchPrefillGraph &) = delete;

    HiggsARDecodeOutput run(const HiggsARPrefillInput & input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsARBatchDecodeGraph {
public:
    HiggsARBatchDecodeGraph(
        std::shared_ptr<HiggsARRuntime> runtime,
        HiggsARBatchKVCache & cache,
        size_t graph_arena_bytes);
    ~HiggsARBatchDecodeGraph();

    HiggsARBatchDecodeGraph(const HiggsARBatchDecodeGraph &) = delete;
    HiggsARBatchDecodeGraph & operator=(const HiggsARBatchDecodeGraph &) = delete;

    // `visible_start[b]` is slot b's left padding, `prompt_capacity` the shared
    // cache index the first generated token is written to. `generated_so_far`
    // resumes an in-flight run after the cache grew and the graph was rebuilt.
    void begin_decode_run(
        const std::vector<int64_t> & visible_start,
        int64_t prompt_capacity,
        int64_t generated_so_far = 0);
    void run_step_into(const HiggsARBatchDecodeInput & input, HiggsARBatchDecodeOutput & output);
    int64_t generated_steps() const;
    HiggsARDecodeTiming timing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsARPrefillGraph {
public:
    HiggsARPrefillGraph(
        std::shared_ptr<HiggsARRuntime> runtime,
        int64_t prompt_steps,
        int64_t start_step,
        HiggsARKVCache * cache,
        size_t graph_arena_bytes);
    ~HiggsARPrefillGraph();

    HiggsARPrefillGraph(const HiggsARPrefillGraph &) = delete;
    HiggsARPrefillGraph & operator=(const HiggsARPrefillGraph &) = delete;

    bool matches(const HiggsARRuntime & runtime, int64_t prompt_steps, int64_t start_step) const;
    HiggsARPrefillOutput run(const HiggsARPrefillInput & input, int64_t start_step = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsARDecodeGraph {
public:
    HiggsARDecodeGraph(
        std::shared_ptr<HiggsARRuntime> runtime,
        int64_t cache_steps,
        HiggsARKVCache & cache,
        size_t graph_arena_bytes);
    ~HiggsARDecodeGraph();

    HiggsARDecodeGraph(const HiggsARDecodeGraph &) = delete;
    HiggsARDecodeGraph & operator=(const HiggsARDecodeGraph &) = delete;

    bool can_run(const HiggsARRuntime & runtime, int64_t required_steps) const;
    int64_t cache_steps() const;
    void import_prefill_state(const runtime::TransformerKVState & state);
    void begin_decode_run();
    HiggsARDecodeTiming timing() const;
    void run_step_into(const HiggsARDecodeInput & input, HiggsARDecodeOutput & output, bool log_timing = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_audio_tts
