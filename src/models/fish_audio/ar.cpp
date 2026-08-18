#include "engine/models/fish_audio/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fish_audio {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

constexpr int64_t kRasWindow = 10;
constexpr float kRasHighTemperature = 1.0F;
constexpr float kRasHighTopP = 0.9F;

struct FishARProfile {
    double graph_build_prefill_ms = 0.0;
    double graph_build_step_ms = 0.0;
    double graph_build_fast_ms = 0.0;
    double slow_embedding_ms = 0.0;
    double fast_embedding_ms = 0.0;
    double prefill_input_upload_ms = 0.0;
    double prefill_graph_ms = 0.0;
    double prefill_output_read_ms = 0.0;
    double step_input_upload_ms = 0.0;
    double step_mask_upload_ms = 0.0;
    double step_graph_ms = 0.0;
    double step_output_read_ms = 0.0;
    double fast_input_upload_ms = 0.0;
    double fast_mask_upload_ms = 0.0;
    double fast_graph_ms = 0.0;
    double fast_output_read_ms = 0.0;
    double sample_bias_ms = 0.0;
    double sample_main_ms = 0.0;
    double sample_high_ms = 0.0;
    double sample_fast_ms = 0.0;
    int64_t prefill_runs = 0;
    int64_t step_runs = 0;
    int64_t fast_runs = 0;
    int64_t generated_frames = 0;
};

struct SampleCandidate {
    int32_t index = 0;
    float probability = 0.0F;
};

struct SampleDistribution {
    size_t source_size = 0;
    std::vector<SampleCandidate> candidates;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct FishLayerWeights {
    assets::TensorDataF32 input_norm;
    core::TensorValue qkv_proj;
    core::TensorValue o_proj;
    std::optional<assets::TensorDataF32> q_norm;
    std::optional<assets::TensorDataF32> k_norm;
    assets::TensorDataF32 post_norm;
    core::TensorValue gate_up_proj;
    core::TensorValue down_proj;
};

struct FishARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    assets::TensorData text_embedding_host;
    assets::TensorData codebook_embedding_host;
    assets::TensorData fast_embedding_host;
    core::TensorValue text_embedding;
    std::vector<FishLayerWeights> slow_layers;
    assets::TensorDataF32 slow_norm;
    std::vector<FishLayerWeights> fast_layers;
    assets::TensorDataF32 fast_norm;
    core::TensorValue fast_output;
};

struct SlowForwardOutput {
    std::vector<float> logits;
    std::vector<float> hidden;
};

struct SlowPrefillOutput {
    SlowForwardOutput forward;
};

struct FishPrefillCacheTarget {
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

modules::QwenDecoderActivationCastPolicy fish_activation_cast_policy(core::BackendType backend_type) {
    modules::QwenDecoderActivationCastPolicy policy;
    if (backend_type == core::BackendType::Vulkan) {
        return policy;
    }
    policy.enabled = true;
    policy.type = GGML_TYPE_BF16;
    policy.after_input_norm = true;
    policy.after_qkv_projection = true;
    policy.after_qk_norm = true;
    policy.after_rope = true;
    policy.after_static_cache_update = true;
    policy.after_attention = true;
    policy.after_attention_output = true;
    policy.after_residual = true;
    policy.after_ffn_norm = true;
    policy.after_mlp_projection = true;
    policy.after_mlp_silu = true;
    policy.after_mlp_mul = true;
    policy.after_output = true;
    return policy;
}

modules::QwenCausalDecoderConfig make_slow_decoder_config(
    const FishAudioTextConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.dim;
    out.stack.num_attention_heads = config.n_head;
    out.stack.num_key_value_heads = config.n_local_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.n_layer;
    out.stack.rms_norm_eps = config.norm_eps;
    out.stack.rope_theta = config.rope_base;
    out.stack.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.use_qk_norm = config.attention_qk_norm;
    out.stack.activation_cast = fish_activation_cast_policy(backend_type);
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.lm_head_precision = GGML_PREC_F32;
    return out;
}

modules::QwenCausalDecoderConfig make_fast_decoder_config(
    const FishAudioFastConfig & config,
    core::BackendType backend_type) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.dim;
    out.stack.num_attention_heads = config.n_head;
    out.stack.num_key_value_heads = config.n_local_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.n_layer;
    out.stack.rms_norm_eps = config.norm_eps;
    out.stack.rope_theta = config.rope_base;
    out.stack.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.qkv_layout = modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.use_qk_norm = config.attention_qk_norm;
    out.stack.activation_cast = fish_activation_cast_policy(backend_type);
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.stack.runtime.mlp.mode = modules::QwenDecoderMLPMode::PackedGateUp;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::QwenCausalDecoderLogitsMode::LastStep;
    out.lm_head_precision = GGML_PREC_F32;
    return out;
}

modules::QwenDecoderLayerWeights bind_layer(
    core::ConstantTensorCache & constants,
    const FishLayerWeights & weights,
    bool use_qk_norm) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_data(constants, weights.input_norm);
    out.self_attention.qkv_weight = weights.qkv_proj;
    out.self_attention.out_weight = weights.o_proj;
    if (use_qk_norm) {
        if (!weights.q_norm.has_value() || !weights.k_norm.has_value()) {
            throw std::runtime_error("Fish Audio q/k norm weights are missing");
        }
        out.q_norm = binding::norm_data(constants, *weights.q_norm);
        out.k_norm = binding::norm_data(constants, *weights.k_norm);
    }
    out.post_norm = binding::norm_data(constants, weights.post_norm);
    out.mlp.gate_up_proj = binding::linear_data(constants, weights.gate_up_proj);
    out.mlp.down_proj = binding::linear_data(constants, weights.down_proj);
    return out;
}

modules::QwenCausalDecoderWeights bind_slow_weights(
    core::ConstantTensorCache & constants,
    const FishARWeights & weights,
    const FishAudioTextConfig & config) {
    modules::QwenCausalDecoderWeights out;
    out.stack.layers.reserve(weights.slow_layers.size());
    for (const auto & layer : weights.slow_layers) {
        out.stack.layers.push_back(bind_layer(constants, layer, config.attention_qk_norm));
    }
    out.final_norm = binding::norm_data(constants, weights.slow_norm);
    out.lm_head = binding::linear_data(constants, weights.text_embedding);
    return out;
}

modules::QwenDecoderLayerWeights bind_fast_layer(
    core::ConstantTensorCache & constants,
    const FishLayerWeights & weights,
    const FishAudioFastConfig & config) {
    return bind_layer(constants, weights, config.attention_qk_norm);
}

void copy_tensor_row_to_f32(const assets::TensorData & table, int64_t row, int64_t width, float * out) {
    if (row < 0 || width <= 0 || table.shape.rank != 2 || table.shape.dims[1] != width ||
        row >= table.shape.dims[0]) {
        throw std::runtime_error("Fish Audio embedding row lookup shape mismatch");
    }
    const size_t row_bytes = ggml_row_size(table.type, width);
    const size_t offset = static_cast<size_t>(row) * row_bytes;
    if (offset + row_bytes > table.bytes.size()) {
        throw std::runtime_error("Fish Audio embedding row lookup exceeded tensor storage");
    }
    const auto * bytes = reinterpret_cast<const uint8_t *>(table.bytes.data()) + offset;
    if (table.type == GGML_TYPE_F32) {
        std::memcpy(out, bytes, static_cast<size_t>(width) * sizeof(float));
    } else if (table.type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes), out, width);
    } else if (table.type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(bytes), out, width);
    } else {
        throw std::runtime_error("Fish Audio host embedding lookup requires f32/f16/bf16 native embeddings");
    }
}

std::vector<float> lookup_row(const assets::TensorData & table, int64_t row, int64_t width) {
    std::vector<float> out(static_cast<size_t>(width), 0.0F);
    copy_tensor_row_to_f32(table, row, width, out.data());
    return out;
}

void add_row(const assets::TensorData & table, int64_t row, int64_t width, std::vector<float> & out) {
    std::vector<float> tmp(static_cast<size_t>(width), 0.0F);
    copy_tensor_row_to_f32(table, row, width, tmp.data());
    for (int64_t i = 0; i < width; ++i) {
        out[static_cast<size_t>(i)] += tmp[static_cast<size_t>(i)];
    }
}

bool is_semantic_token(const FishAudioConfig & config, int32_t token) {
    return token >= config.semantic_start_token_id && token <= config.semantic_end_token_id;
}

std::vector<float> build_slow_embeddings(
    const FishAudioConfig & config,
    const FishARWeights & weights,
    const int32_t * matrix,
    int64_t steps) {
    const int64_t rows = config.fast.num_codebooks + 1;
    const int64_t hidden = config.text.dim;
    std::vector<float> out(static_cast<size_t>(steps * hidden), 0.0F);
    const float semantic_scale = 1.0F / std::sqrt(static_cast<float>(rows));
    for (int64_t step = 0; step < steps; ++step) {
        const int32_t token = matrix[step];
        auto row = lookup_row(weights.text_embedding_host, token, hidden);
        if (is_semantic_token(config, token)) {
            for (int64_t codebook = 0; codebook < config.fast.num_codebooks; ++codebook) {
                const int32_t code = matrix[(codebook + 1) * steps + step];
                add_row(
                    weights.codebook_embedding_host,
                    codebook * config.fast.vocab_size + code,
                    hidden,
                    row);
            }
            for (float & value : row) {
                value *= semantic_scale;
            }
        }
        std::copy(row.begin(), row.end(), out.begin() + static_cast<std::ptrdiff_t>(step * hidden));
    }
    return out;
}

std::vector<float> build_slow_embedding_for_frame(
    const FishAudioConfig & config,
    const FishARWeights & weights,
    const std::vector<int32_t> & frame) {
    if (static_cast<int64_t>(frame.size()) != config.fast.num_codebooks + 1) {
        throw std::runtime_error("Fish Audio frame size mismatch");
    }
    return build_slow_embeddings(config, weights, frame.data(), 1);
}

std::vector<float> build_fast_embedding(
    const FishAudioConfig & config,
    const FishARWeights & weights,
    int32_t code) {
    return lookup_row(weights.fast_embedding_host, code, config.fast.dim);
}

FishLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t intermediate,
    bool qk_norm,
    assets::TensorStorageType storage_type) {
    FishLayerWeights w;
    w.input_norm = source.require_f32_tensor(prefix + ".attention_norm.weight", {hidden});
    {
        const auto q = source.require_tensor(prefix + ".attention.q_proj.weight", storage_type, {heads * head_dim, hidden});
        const auto k = source.require_tensor(prefix + ".attention.k_proj.weight", storage_type, {kv_heads * head_dim, hidden});
        const auto v = source.require_tensor(prefix + ".attention.v_proj.weight", storage_type, {kv_heads * head_dim, hidden});
        if (q.type != k.type || q.type != v.type) {
            throw std::runtime_error("Fish Audio packed QKV weights require matching storage types");
        }
        std::vector<std::byte> packed;
        packed.reserve(q.bytes.size() + k.bytes.size() + v.bytes.size());
        packed.insert(packed.end(), q.bytes.begin(), q.bytes.end());
        packed.insert(packed.end(), k.bytes.begin(), k.bytes.end());
        packed.insert(packed.end(), v.bytes.begin(), v.bytes.end());
        w.qkv_proj = store.make_tensor(
            core::TensorShape::from_dims({(heads + 2 * kv_heads) * head_dim, hidden}),
            q.type,
            packed.data(),
            packed.size());
    }
    w.o_proj = store.load_tensor(source, prefix + ".attention.wo.weight", storage_type, {hidden, heads * head_dim});
    if (qk_norm) {
        w.q_norm = source.require_f32_tensor(prefix + ".attention.q_norm.weight", {head_dim});
        w.k_norm = source.require_f32_tensor(prefix + ".attention.k_norm.weight", {head_dim});
    }
    w.post_norm = source.require_f32_tensor(prefix + ".ffn_norm.weight", {hidden});
    w.down_proj = store.load_tensor(source, prefix + ".feed_forward.w2.weight", storage_type, {hidden, intermediate});
    {
        const auto gate = source.require_tensor(prefix + ".feed_forward.w1.weight", storage_type, {intermediate, hidden});
        const auto up = source.require_tensor(prefix + ".feed_forward.w3.weight", storage_type, {intermediate, hidden});
        if (gate.type != up.type) {
            throw std::runtime_error("Fish Audio packed gate/up weights require matching storage types");
        }
        std::vector<std::byte> packed;
        packed.reserve(gate.bytes.size() + up.bytes.size());
        packed.insert(packed.end(), gate.bytes.begin(), gate.bytes.end());
        packed.insert(packed.end(), up.bytes.begin(), up.bytes.end());
        w.gate_up_proj = store.make_tensor(
            core::TensorShape::from_dims({intermediate * 2, hidden}),
            gate.type,
            packed.data(),
            packed.size());
    }
    return w;
}

FishARWeights load_ar_weights(
    const FishAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & source = *assets.model_weights;
    const auto & config = assets.config;
    FishARWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "fish_audio.ar.weights",
        weight_context_bytes);
    weights.text_embedding_host = source.require_tensor(
        "embeddings.weight",
        assets::TensorStorageType::Native,
        {config.text.vocab_size, config.text.dim});
    weights.codebook_embedding_host = source.require_tensor(
        "codebook_embeddings.weight",
        assets::TensorStorageType::Native,
        {config.fast.vocab_size * config.fast.num_codebooks, config.text.dim});
    weights.fast_embedding_host = source.require_tensor(
        "fast_embeddings.weight",
        assets::TensorStorageType::Native,
        {config.fast.vocab_size, config.fast.dim});
    weights.text_embedding = weights.store->load_tensor(
        source,
        "embeddings.weight",
        storage_type,
        {config.text.vocab_size, config.text.dim});
    weights.slow_layers.reserve(static_cast<size_t>(config.text.n_layer));
    for (int64_t i = 0; i < config.text.n_layer; ++i) {
        weights.slow_layers.push_back(load_layer(
            *weights.store,
            source,
            "layers." + std::to_string(i),
            config.text.dim,
            config.text.n_head,
            config.text.n_local_heads,
            config.text.head_dim,
            config.text.intermediate_size,
            config.text.attention_qk_norm,
            storage_type));
    }
    weights.slow_norm = source.require_f32_tensor("norm.weight", {config.text.dim});
    weights.fast_layers.reserve(static_cast<size_t>(config.fast.n_layer));
    for (int64_t i = 0; i < config.fast.n_layer; ++i) {
        weights.fast_layers.push_back(load_layer(
            *weights.store,
            source,
            "fast_layers." + std::to_string(i),
            config.fast.dim,
            config.fast.n_head,
            config.fast.n_local_heads,
            config.fast.head_dim,
            config.fast.intermediate_size,
            config.fast.attention_qk_norm,
            storage_type));
    }
    weights.fast_norm = source.require_f32_tensor("fast_norm.weight", {config.fast.dim});
    weights.fast_output = weights.store->load_tensor(
        source,
        "fast_output.weight",
        storage_type,
        {config.fast.vocab_size, config.fast.dim});
    weights.store->upload();
    return weights;
}

struct SampleState {
    uint64_t seed = 0;
    uint64_t call_index = 0;
    std::mt19937 rng;
    std::vector<int32_t> previous_main;
};

SampleDistribution logits_to_distribution(
    const std::vector<float> & logits,
    float temperature,
    float top_p,
    int top_k) {
    if (logits.empty()) {
        throw std::runtime_error("Fish Audio sampling requires non-empty logits");
    }
    std::vector<int32_t> order;
    order.reserve(logits.size());
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        const float logit = logits[i];
        if (!std::isfinite(logit)) {
            continue;
        }
        order.push_back(static_cast<int32_t>(i));
        max_logit = std::max(max_logit, logit);
    }
    double denom = 0.0;
    for (const int32_t index : order) {
        denom += std::exp(logits[static_cast<size_t>(index)] - max_logit);
    }
    if (denom <= 0.0) {
        throw std::runtime_error("Fish Audio sampling logits produced zero probability mass");
    }
    const size_t candidate_count = std::min(order.size(), static_cast<size_t>(std::max(top_k, 1)));
    const auto by_logit_desc = [&](int32_t lhs, int32_t rhs) {
        return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)];
    };
    if (candidate_count < order.size()) {
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(candidate_count), order.end(), by_logit_desc);
        order.resize(candidate_count);
    } else {
        std::sort(order.begin(), order.end(), by_logit_desc);
    }
    double cumulative = 0.0;
    std::vector<SampleCandidate> kept;
    kept.reserve(candidate_count);
    for (size_t i = 0; i < order.size(); ++i) {
        const int32_t index = order[i];
        const float logit = logits[static_cast<size_t>(index)];
        const float prob = static_cast<float>(std::exp(logit - max_logit) / denom);
        cumulative += prob;
        const bool remove = cumulative > static_cast<double>(top_p) && i != 0;
        if (!remove) {
            kept.push_back({index, 0.0F});
        }
    }
    float filtered_max = -std::numeric_limits<float>::infinity();
    const float temperature_scale = std::max(temperature, 1.0e-5F);
    for (const auto & candidate : kept) {
        filtered_max = std::max(filtered_max, logits[static_cast<size_t>(candidate.index)] / temperature_scale);
    }
    double filtered_denom = 0.0;
    for (auto & candidate : kept) {
        candidate.probability =
            std::exp(logits[static_cast<size_t>(candidate.index)] / temperature_scale - filtered_max);
        filtered_denom += candidate.probability;
    }
    if (filtered_denom <= 0.0) {
        throw std::runtime_error("Fish Audio sampling filter produced zero probability mass");
    }
    for (auto & candidate : kept) {
        candidate.probability = static_cast<float>(static_cast<double>(candidate.probability) / filtered_denom);
    }
    return {logits.size(), std::move(kept)};
}

int32_t sample_from_logits(
    const std::vector<float> & logits,
    float temperature,
    float top_p,
    int top_k,
    SampleState & state,
    const sampling::TorchCudaSamplingPolicy & policy) {
    const auto distribution = logits_to_distribution(logits, temperature, top_p, top_k);
    const uint64_t call_index = state.call_index++;
    if (!policy.cuda_fast_path) {
        std::vector<double> weights;
        weights.reserve(distribution.candidates.size());
        for (const auto & candidate : distribution.candidates) {
            weights.push_back(static_cast<double>(std::max(candidate.probability, 0.0F)));
        }
        std::discrete_distribution<size_t> sampler(weights.begin(), weights.end());
        return distribution.candidates[sampler(state.rng)].index;
    }
    int32_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto & candidate : distribution.candidates) {
        if (!(candidate.probability > 0.0F)) {
            continue;
        }
        const float exponential = sampling::torch_cuda_tensor_iterator_exponential_element(
            state.seed,
            static_cast<uint64_t>(distribution.source_size),
            static_cast<uint64_t>(candidate.index),
            call_index,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        const float uniform = std::exp(-exponential);
        const float uniform_bf16 = ggml_bf16_to_fp32(ggml_fp32_to_bf16(uniform));
        const float exponential_bf16 = ggml_bf16_to_fp32(ggml_fp32_to_bf16(-std::log(uniform_bf16)));
        const double score = static_cast<double>(candidate.probability) / static_cast<double>(exponential_bf16);
        if (score > best_score) {
            best_score = score;
            best = candidate.index;
        }
    }
    return best;
}

std::vector<float> apply_semantic_bias(
    const FishAudioConfig & config,
    int32_t im_end_id,
    const std::vector<float> & logits) {
    std::vector<float> out(logits.size(), -std::numeric_limits<float>::infinity());
    const int64_t begin = std::max<int64_t>(0, config.semantic_start_token_id);
    const int64_t end = std::min<int64_t>(static_cast<int64_t>(logits.size()) - 1, config.semantic_end_token_id);
    for (int64_t i = begin; i <= end; ++i) {
        out[static_cast<size_t>(i)] = logits[static_cast<size_t>(i)];
    }
    if (im_end_id >= 0 && static_cast<size_t>(im_end_id) < logits.size()) {
        out[static_cast<size_t>(im_end_id)] = logits[static_cast<size_t>(im_end_id)];
    }
    return out;
}

core::TensorValue make_fish_causal_mask(
    core::ModuleBuildContext &,
    core::ConstantTensorCache & constants,
    int64_t steps) {
    auto values = modules::qwen_causal_prefill_mask_values(1, steps);
    return constants.make_tensor(
        core::TensorShape::from_dims({1, 1, steps, steps}),
        GGML_TYPE_F16,
        values.data(),
        values.size() * sizeof(ggml_fp16_t));
}

struct FishCausalDecoderOutputs {
    core::TensorValue hidden;
    core::TensorValue logits;
    modules::QwenDecoderStackState state;
};

FishCausalDecoderOutputs build_fish_causal_decoder(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::QwenCausalDecoderWeights & weights,
    const modules::QwenCausalDecoderConfig & config,
    bool norm_fastlayer_input) {
    auto mask = make_fish_causal_mask(ctx, constants, input.shape.dims[1]);
    auto x = input;
    modules::QwenDecoderStackState state;
    state.layers.reserve(weights.stack.layers.size());
    const auto layer_config = modules::qwen_decoder_layer_config_from_stack(config.stack);
    const modules::QwenDecoderLayerModule layer_module(layer_config);
    for (const auto & layer : weights.stack.layers) {
        auto out = layer_module.build(ctx, x, positions, layer, std::nullopt, std::nullopt, mask);
        x = out.output;
        auto state_key = core::wrap_tensor(ggml_dup(ctx.ggml, out.key.tensor), out.key.shape, out.key.type);
        auto state_value = core::wrap_tensor(ggml_dup(ctx.ggml, out.value.tensor), out.value.shape, out.value.type);
        state.layers.push_back({state_key, state_value});
    }
    auto hidden_sequence = modules::RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false})
                               .build(ctx, x, weights.final_norm);
    const int64_t steps = hidden_sequence.shape.dims[1];
    auto fast_hidden_source = norm_fastlayer_input ? hidden_sequence : x;
    auto hidden = modules::SliceModule({1, steps - 1, 1}).build(ctx, fast_hidden_source);
    auto logits = modules::LinearModule({config.stack.hidden_size, config.logits_size, false, config.lm_head_precision})
                      .build(ctx, modules::SliceModule({1, steps - 1, 1}).build(ctx, hidden_sequence), weights.lm_head);
    auto hidden_out = core::wrap_tensor(ggml_dup(ctx.ggml, hidden.tensor), hidden.shape, hidden.type);
    auto logits_out = core::wrap_tensor(ggml_dup(ctx.ggml, logits.tensor), logits.shape, logits.type);
    return {hidden_out, logits_out, std::move(state)};
}

struct FishStaticDecoderOutputs {
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerKVCache cache;
};

FishStaticDecoderOutputs build_fish_static_decoder(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::QwenCausalDecoderWeights & weights,
    const modules::QwenCausalDecoderConfig & config,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot,
    std::vector<core::TensorValue> cache_keys,
    std::vector<core::TensorValue> cache_values,
    bool norm_fastlayer_input) {
    if (cache_keys.size() != weights.stack.layers.size() || cache_values.size() != weights.stack.layers.size()) {
        throw std::runtime_error("Fish Audio static decoder cache layer count mismatch");
    }
    const int64_t step_elems = config.stack.num_key_value_heads * config.stack.head_dim;
    auto x = input;
    const auto layer_config = modules::qwen_decoder_layer_config_from_stack(config.stack);
    const modules::QwenDecoderLayerModule layer_module(layer_config);
    for (size_t layer_index = 0; layer_index < weights.stack.layers.size(); ++layer_index) {
        auto out = layer_module.build_with_static_cache_tail(
            ctx,
            graph,
            x,
            positions,
            weights.stack.layers[layer_index],
            cache_keys[layer_index],
            cache_values[layer_index],
            cache_slot,
            attention_mask);
        x = out.output;
    }
    auto hidden = modules::RMSNormModule({config.stack.hidden_size, config.stack.rms_norm_eps, true, false})
                      .build(ctx, x, weights.final_norm);
    const auto logits = modules::LinearModule({
                            config.stack.hidden_size,
                            config.logits_size,
                            config.use_lm_head_bias,
                            config.lm_head_precision,
                        })
                            .build(ctx, hidden, weights.lm_head);
    auto fast_hidden = norm_fastlayer_input ? hidden : x;
    runtime::TransformerKVCacheOptions cache_options;
    cache_options.allow_bf16_storage = !cache_keys.empty() && cache_keys.front().type == GGML_TYPE_BF16;
    return {
        fast_hidden,
        logits,
        runtime::TransformerKVCache(
            cache_steps,
            step_elems,
            std::move(cache_keys),
            std::move(cache_values),
            cache_options),
    };
}

// --------------------------------------------------------------------------
// Stapelweises Decodieren
//
// Die Sequenzen liegen in der Token-Achse (ggml ne[1]) nebeneinander, nicht in
// der Kanalachse. Nur so bedient ein Gewichtslesen alle Sequenzen: ggml waehlt
// fuer q8_0-Gewichte den mul_mat_vec_q-Kern, und dessen ncols_dst ist ne[1].
// In der Kanalachse waere ncols_dst weiterhin 1 und jede Sequenz laese die
// Gewichte erneut -- also genau der flache Verlauf, den wir loswerden wollen.
//
// Fuer Aufmerksamkeit und KV-Speicher braucht jede Sequenz dagegen ihren
// eigenen Bereich. Das geht ohne Umkopieren: {1, slots, heads, dim} und
// {slots, 1, heads, dim} liegen Byte fuer Byte gleich im Speicher, solange je
// Sequenz genau ein Schritt bearbeitet wird. Zwischen beiden Sichten steht
// darum nur ein reshape, kein cont.
//
// Daraus folgt die Bitgleichheit: RoPE bekommt je Sequenz ihre echte Position
// (kein Linksauffuellen wie bei higgs/qwen3), die Aufmerksamkeit sieht je
// Sequenz genau die Form des Einzellaufs, und die Matmuls behalten fuer
// ncols_dst 1..4 dieselbe Warp-Aufteilung und damit dieselbe Summenreihenfolge.
// Oberhalb von 4 wechselt ggml die Warp-Zahl (calc_nwarps in mmvq.cu) und die
// Ergebnisse weichen im letzten Bit ab; deshalb ist der Stapel dort gedeckelt.
constexpr int64_t kFishMaxExactBatch = 4;

core::TensorValue fish_activation_cast(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::QwenDecoderActivationCastPolicy & policy) {
    if (policy.type == GGML_TYPE_F32) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    }
    auto rounded = core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, policy.type), input.shape, policy.type);
    return core::wrap_tensor(ggml_cast(ctx.ggml, rounded.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
}

core::TensorValue fish_reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

struct FishBatchCacheBinding {
    core::TensorValue key;    // {slots, cache_steps, kv_heads, head_dim}
    core::TensorValue value;
};

// Der einzige Unterschied zwischen den beiden Stapelachsen. Alles andere im
// Block -- Normen, RoPE, Cache, Aufmerksamkeit -- rechnet zeilenweise und ist
// von der Form unabhaengig; {1, slots, n} und {slots, 1, n} liegen ohnehin
// Byte fuer Byte gleich im Speicher. Erst der Matmul entscheidet anhand der
// Form, ob ggml ncols_dst = slots sieht (ein Gewichtslesen fuer alle) oder
// ncols_dst = 1 (ein Gewichtslesen je Sequenz).
core::TensorValue fish_batch_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,  // {1, slots, in_features}
    int64_t slots,
    FishAudioBatchAxis axis,
    const modules::LinearConfig & config,
    const modules::LinearWeights & weights) {
    if (axis == FishAudioBatchAxis::Token) {
        return modules::LinearModule(config).build(ctx, input, weights);
    }
    // LinearModule plaettet jede Eingabe vor dem Matmul auf zwei Dimensionen
    // ({prefix_elements, last_dim}). Ueber es gerufen landeten die Sequenzen
    // also doch wieder in ne[1], und die Kanal-Achse waere nur ein anderer Weg
    // zum selben Kern. Fuer die Kanal-Achse wird ggml_mul_mat deshalb direkt
    // mit einem dreidimensionalen Eingang gerufen: ne[1] bleibt 1, die
    // Sequenzen sitzen in ne[2], und ncols_dst ist damit wirklich 1.
    if (config.use_bias) {
        throw std::runtime_error("Fish Audio channel-axis batching does not support linear bias");
    }
    auto channel_input = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({slots, 1, config.in_features}));
    ggml_tensor * projected = ggml_mul_mat(ctx.ggml, weights.weight.tensor, channel_input.tensor);
    if (config.precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(projected, config.precision);
    }
    auto out = core::wrap_tensor(
        projected, core::TensorShape::from_dims({slots, 1, config.out_features}), GGML_TYPE_F32);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({1, slots, config.out_features}));
}

// Ein Decoderblock fuer `slots` Sequenzen mit je einem Schritt.
//   input      {1, slots, hidden}
//   positions  {slots}                    -- echte Position je Sequenz
//   cache_slot {slots}, i64               -- Zeile im flachen Speicher
//   mask       {slots, 1, 1, cache_steps}
core::TensorValue build_fish_batch_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::QwenDecoderLayerWeights & weights,
    const modules::QwenDecoderLayerConfig & config,
    int64_t slots,
    FishAudioBatchAxis axis,
    const FishBatchCacheBinding & cache,
    const core::TensorValue & cache_slot,
    const core::TensorValue & mask) {
    const int64_t dim = config.head_dim;
    const int64_t q_out = config.num_attention_heads * dim;
    const int64_t kv_out = config.num_key_value_heads * dim;
    const auto & cast = config.activation_cast;
    const auto maybe_cast = [&](const core::TensorValue & value, bool enabled) {
        return (cast.enabled && enabled) ? fish_activation_cast(ctx, value, cast) : value;
    };

    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    x_norm = maybe_cast(x_norm, cast.after_input_norm);

    if (!weights.self_attention.qkv_weight.has_value()) {
        throw std::runtime_error("Fish Audio batch layer requires packed QKV weights");
    }
    auto qkv = fish_batch_linear(
        ctx,
        x_norm,
        slots,
        axis,
        {config.hidden_size, q_out + 2 * kv_out, false, config.projection_precision},
        {*weights.self_attention.qkv_weight, std::nullopt});
    auto q = maybe_cast(modules::SliceModule({2, 0, q_out}).build(ctx, qkv), cast.after_qkv_projection);
    auto k = maybe_cast(modules::SliceModule({2, q_out, kv_out}).build(ctx, qkv), cast.after_qkv_projection);
    auto v = maybe_cast(modules::SliceModule({2, q_out + kv_out, kv_out}).build(ctx, qkv), cast.after_qkv_projection);

    q = fish_reshape_heads(ctx, q, config.num_attention_heads, dim);
    k = fish_reshape_heads(ctx, k, config.num_key_value_heads, dim);
    if (config.use_qk_norm) {
        q = modules::RMSNormModule({dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.q_norm);
        k = modules::RMSNormModule({dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.k_norm);
        q = maybe_cast(q, cast.after_qk_norm);
        k = maybe_cast(k, cast.after_qk_norm);
    }
    v = fish_reshape_heads(ctx, v, config.num_key_value_heads, dim);

    const modules::RoPEModule rope({dim, config.rope_type, config.rope_theta});
    q = maybe_cast(rope.build(ctx, q, positions, nullptr), cast.after_rope);
    k = maybe_cast(rope.build(ctx, k, positions, nullptr), cast.after_rope);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    // Von "ein Stapel, slots Schritte" nach "slots Stapel, ein Schritt".
    // Reine Umdeutung der Form, die Bytes bleiben liegen.
    const auto slot_rows = core::TensorShape::from_dims({slots, 1, config.num_key_value_heads, dim});
    auto k_rows = core::reshape_tensor(ctx, k, slot_rows);
    auto v_rows = core::reshape_tensor(ctx, v, slot_rows);
    const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::BackendViewOptimized});
    auto key_cache = maybe_cast(set_rows.build(ctx, cache.key, k_rows, cache_slot), cast.after_static_cache_update);
    auto value_cache = maybe_cast(set_rows.build(ctx, cache.value, v_rows, cache_slot), cast.after_static_cache_update);

    auto q_slots = core::reshape_tensor(
        ctx, q, core::TensorShape::from_dims({slots, 1, config.num_attention_heads, dim}));
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q_slots.shape.rank}).build(ctx, q_slots);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, key_cache.shape.rank}).build(ctx, key_cache);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, value_cache.shape.rank}).build(ctx, value_cache);
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);

    // Aufmerksamkeit je Sequenz einzeln. Ein einziger Aufruf ueber die
    // Stapelachse waere kuerzer, aber ggml waehlt seinen Flash-Attention-Kern
    // und dessen Aufteilung der Schluesselachse nach der Groesse des Tensors:
    // ein Stapel bekommt eine andere Summenreihenfolge als ein Einzellauf, und
    // der Unterschied im letzten Bit kippt binnen weniger Rahmen eine
    // Ziehung. Je Sequenz aufgerufen sieht der Kern genau die Form, die er
    // auch allein saehe.
    ggml_tensor * merged = nullptr;
    for (int64_t slot = 0; slot < slots; ++slot) {
        ggml_tensor * q_slot = ggml_view_4d(
            ctx.ggml,
            q_contiguous.tensor,
            dim,
            1,
            config.num_attention_heads,
            1,
            q_contiguous.tensor->nb[1],
            q_contiguous.tensor->nb[2],
            q_contiguous.tensor->nb[3],
            static_cast<size_t>(slot) * q_contiguous.tensor->nb[3]);
        ggml_tensor * k_slot = ggml_view_4d(
            ctx.ggml,
            k_heads.tensor,
            dim,
            k_heads.tensor->ne[1],
            config.num_key_value_heads,
            1,
            k_heads.tensor->nb[1],
            k_heads.tensor->nb[2],
            k_heads.tensor->nb[3],
            static_cast<size_t>(slot) * k_heads.tensor->nb[3]);
        ggml_tensor * v_slot = ggml_view_4d(
            ctx.ggml,
            v_heads.tensor,
            dim,
            v_heads.tensor->ne[1],
            config.num_key_value_heads,
            1,
            v_heads.tensor->nb[1],
            v_heads.tensor->nb[2],
            v_heads.tensor->nb[3],
            static_cast<size_t>(slot) * v_heads.tensor->nb[3]);
        ggml_tensor * mask_slot = ggml_view_4d(
            ctx.ggml,
            mask.tensor,
            mask.tensor->ne[0],
            1,
            1,
            1,
            mask.tensor->nb[1],
            mask.tensor->nb[2],
            mask.tensor->nb[3],
            static_cast<size_t>(slot) * mask.tensor->nb[3]);
        ggml_tensor * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_slot,
            k_slot,
            v_slot,
            mask_slot,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F,
            0.0F);
        ggml_flash_attn_ext_set_prec(flash, config.attention_precision);
        ggml_tensor * flat = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, flash), q_out, 1);
        merged = merged == nullptr ? flat : ggml_concat(ctx.ggml, merged, flat, 1);
    }
    auto context = core::wrap_tensor(
        merged, core::TensorShape::from_dims({slots, q_out}), GGML_TYPE_F32);
    context = maybe_cast(context, cast.after_attention);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, slots, q_out}));

    auto attn_out = fish_batch_linear(
        ctx,
        context,
        slots,
        axis,
        {q_out, config.hidden_size, false, config.projection_precision},
        {weights.self_attention.out_weight, std::nullopt});
    attn_out = maybe_cast(attn_out, cast.after_attention_output);
    auto x = maybe_cast(modules::AddModule{}.build(ctx, input, attn_out), cast.after_residual);

    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    ff_in = maybe_cast(ff_in, cast.after_ffn_norm);
    if (!weights.mlp.gate_up_proj.has_value()) {
        throw std::runtime_error("Fish Audio batch layer requires packed gate/up weights");
    }
    auto gate_up = fish_batch_linear(
        ctx,
        ff_in,
        slots,
        axis,
        {config.hidden_size, config.intermediate_size * 2, false, config.projection_precision},
        *weights.mlp.gate_up_proj);
    auto gate = maybe_cast(
        modules::SliceModule({2, 0, config.intermediate_size}).build(ctx, gate_up), cast.after_mlp_projection);
    auto up = maybe_cast(
        modules::SliceModule({2, config.intermediate_size, config.intermediate_size}).build(ctx, gate_up),
        cast.after_mlp_projection);
    gate = maybe_cast(modules::SiluModule{}.build(ctx, gate), cast.after_mlp_silu);
    auto gated = maybe_cast(modules::MulModule{}.build(ctx, gate, up), cast.after_mlp_mul);
    auto down = fish_batch_linear(
        ctx,
        gated,
        slots,
        axis,
        {config.intermediate_size, config.hidden_size, false, config.projection_precision},
        weights.mlp.down_proj);
    down = maybe_cast(down, cast.after_mlp_projection);
    return maybe_cast(modules::AddModule{}.build(ctx, x, down), cast.after_output);
}

}  // namespace

class FishARWeightsRuntime {
public:
    FishARWeightsRuntime(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend_config,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : assets_(std::move(assets)),
          threads_(threads),
          graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Fish Audio AR weights runtime requires assets");
        }
        backend_config.threads = threads_;
        backend_ = core::init_backend(backend_config);
        backend_type_ = core::backend_type(backend_);
        weights_ = std::make_shared<FishARWeights>(
            load_ar_weights(*assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type));
        slow_step_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "fish_audio.ar.step.constants",
            256ull * 1024ull * 1024ull);
        fast_constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            threads_,
            "fish_audio.ar.fast.constants",
            256ull * 1024ull * 1024ull);
    }

    ~FishARWeightsRuntime() {
        fast_constants_.reset();
        slow_step_constants_.reset();
        weights_.reset();
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    FishARWeightsRuntime(const FishARWeightsRuntime &) = delete;
    FishARWeightsRuntime & operator=(const FishARWeightsRuntime &) = delete;

    const FishAudioAssets & assets() const noexcept {
        return *assets_;
    }

    const FishARWeights & weights() const noexcept {
        return *weights_;
    }

    int threads() const noexcept {
        return threads_;
    }

    size_t graph_arena_bytes() const noexcept {
        return graph_arena_bytes_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    core::ConstantTensorCache & slow_step_constants() const noexcept {
        return *slow_step_constants_;
    }

    core::ConstantTensorCache & fast_constants() const noexcept {
        return *fast_constants_;
    }

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    std::shared_ptr<const FishARWeights> weights_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    std::unique_ptr<core::ConstantTensorCache> slow_step_constants_;
    std::unique_ptr<core::ConstantTensorCache> fast_constants_;
};

class FishAudioARRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend_config,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : runtime_(std::make_shared<FishARWeightsRuntime>(
              std::move(assets),
              backend_config,
              threads,
              graph_arena_bytes,
              weight_context_bytes,
              weight_storage_type)),
          sampling_policy_(sampling::resolve_torch_cuda_sampling_policy(
              runtime_->backend_type(),
              backend_config.device,
              "fish_audio.ar.cuda_sampling_policy",
              "Fish Audio",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {}

    ~Impl() {
        batch_fast_graph_.reset();
        batch_step_graph_.reset();
        step_graph_.reset();
        prefill_graph_.reset();
        fast_graph_.reset();
        runtime_.reset();
    }

    FishAudioCodes generate(const FishAudioPrompt & prompt, const FishAudioGenerationOptions & options) {
        FishARProfile profile;
        const auto & assets = runtime_->assets();
        const auto & weights = runtime_->weights();
        if (prompt.codebook_rows != assets.config.fast.num_codebooks + 1 ||
            static_cast<int64_t>(prompt.matrix.size()) != prompt.codebook_rows * prompt.steps) {
            throw std::runtime_error("Fish Audio AR prompt shape mismatch");
        }
        const int64_t max_new_tokens = std::min(options.max_new_tokens, assets.config.text.max_seq_len - prompt.steps);
        if (max_new_tokens <= 0) {
            throw std::runtime_error("Fish Audio prompt leaves no room for generated tokens");
        }
        // Prefill writes into the reusable step KV cache; rebuild the copy graph for each request.
        prefill_graph_.reset();
        ensure_step_graph(prompt.steps + max_new_tokens, profile);
        ensure_prefill_graph(prompt.steps, profile);
        ensure_fast_graph(profile);
        SampleState sample;
        sample.seed = options.seed;
        sample.rng.seed(options.seed);
        sample.previous_main.assign(static_cast<size_t>(kRasWindow), 0);
        auto timing_start = Clock::now();
        auto embeddings = build_slow_embeddings(assets.config, weights, prompt.matrix.data(), prompt.steps);
        profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        auto prefill = prefill_graph_->run(embeddings, profile);
        std::vector<int32_t> generated_frame_major;
        generated_frame_major.reserve(static_cast<size_t>(max_new_tokens * assets.config.fast.num_codebooks));
        auto frame = sample_frame(prefill.forward.logits, prefill.forward.hidden, options, sample, false, profile);
        if (frame.front() == im_end_id()) {
            log_profile(profile);
            return FishAudioCodes{{}, assets.config.fast.num_codebooks, 0};
        }
        append_frame(generated_frame_major, frame);
        ++profile.generated_frames;
        step_graph_->finish_prefill(prompt.steps);
        bool ended_by_im_end = false;
        for (int64_t step = 1; step < max_new_tokens; ++step) {
            timing_start = Clock::now();
            const auto input = build_slow_embedding_for_frame(assets.config, weights, frame);
            profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            auto step_out = step_graph_->run(input, profile);
            frame = sample_frame(step_out.logits, step_out.hidden, options, sample, true, profile);
            if (frame.front() == im_end_id()) {
                ended_by_im_end = true;
                break;
            }
            append_frame(generated_frame_major, frame);
            ++profile.generated_frames;
        }
        if (!ended_by_im_end && !generated_frame_major.empty()) {
            generated_frame_major.resize(generated_frame_major.size() - static_cast<size_t>(assets.config.fast.num_codebooks));
            --profile.generated_frames;
        }
        FishAudioCodes out;
        out.codebooks = assets.config.fast.num_codebooks;
        out.frames = static_cast<int64_t>(generated_frame_major.size()) / out.codebooks;
        out.codes.assign(static_cast<size_t>(out.codebooks * out.frames), 0);
        for (int64_t frame_index = 0; frame_index < out.frames; ++frame_index) {
            for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
                out.codes[static_cast<size_t>(codebook * out.frames + frame_index)] =
                    generated_frame_major[static_cast<size_t>(frame_index * out.codebooks + codebook)];
            }
        }
        log_profile(profile);
        return out;
    }

    std::vector<FishAudioCodes> generate_batch(
        const std::vector<FishAudioPrompt> & prompts,
        const std::vector<FishAudioGenerationOptions> & options,
        FishAudioBatchAxis axis) {
        if (prompts.empty() || prompts.size() != options.size()) {
            throw std::runtime_error("Fish Audio batch generation requires one option set per prompt");
        }
        if (prompts.size() == 1) {
            return {generate(prompts.front(), options.front())};
        }
        FishARProfile profile;
        const auto & assets = runtime_->assets();
        const auto & weights = runtime_->weights();
        const int64_t slots = static_cast<int64_t>(prompts.size());
        const int64_t codebooks = assets.config.fast.num_codebooks;
        const int64_t vocab = assets.config.text.vocab_size;
        const int64_t hidden_dim = assets.config.text.dim;
        const int64_t fast_dim = assets.config.fast.dim;
        const int64_t fast_vocab = assets.config.fast.vocab_size;

        std::vector<int64_t> max_new(static_cast<size_t>(slots), 0);
        int64_t cache_steps = 0;
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto & prompt = prompts[static_cast<size_t>(slot)];
            if (prompt.codebook_rows != codebooks + 1 ||
                static_cast<int64_t>(prompt.matrix.size()) != prompt.codebook_rows * prompt.steps) {
                throw std::runtime_error("Fish Audio AR prompt shape mismatch");
            }
            const int64_t room = std::min(
                options[static_cast<size_t>(slot)].max_new_tokens,
                assets.config.text.max_seq_len - prompt.steps);
            if (room <= 0) {
                throw std::runtime_error("Fish Audio prompt leaves no room for generated tokens");
            }
            max_new[static_cast<size_t>(slot)] = room;
            cache_steps = std::max(cache_steps, prompt.steps + room);
        }

        // Der Einzelpfad und der Stapelpfad brauchen beide einen KV-Speicher;
        // zusammen passen sie nicht zuverlaessig auf die Karte.
        step_graph_.reset();
        prefill_graph_.reset();
        fast_graph_.reset();
        ensure_batch_graphs(slots, cache_steps, axis, profile);

        std::vector<std::vector<float>> prefill_logits(static_cast<size_t>(slots));
        std::vector<std::vector<float>> prefill_hidden(static_cast<size_t>(slots));
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto & prompt = prompts[static_cast<size_t>(slot)];
            auto timing_start = Clock::now();
            const auto embeddings =
                build_slow_embeddings(assets.config, weights, prompt.matrix.data(), prompt.steps);
            profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            PrefillGraph prefill(
                runtime_,
                prompt.steps,
                batch_step_graph_->prefill_target_cache(),
                slot * batch_step_graph_->cache_steps());
            profile.graph_build_prefill_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            auto out = prefill.run(embeddings, profile);
            prefill_logits[static_cast<size_t>(slot)] = std::move(out.forward.logits);
            prefill_hidden[static_cast<size_t>(slot)] = std::move(out.forward.hidden);
            batch_step_graph_->reset_slot(slot, prompt.steps);
        }

        std::vector<SampleState> samples(static_cast<size_t>(slots));
        std::vector<std::vector<int32_t>> generated(static_cast<size_t>(slots));
        std::vector<std::vector<int32_t>> frames(
            static_cast<size_t>(slots), std::vector<int32_t>(static_cast<size_t>(codebooks + 1), 0));
        std::vector<char> active(static_cast<size_t>(slots), 1);
        std::vector<char> ended(static_cast<size_t>(slots), 0);
        std::vector<int32_t> positions(static_cast<size_t>(slots), 0);
        std::vector<int64_t> rows(static_cast<size_t>(slots), 0);
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto index = static_cast<size_t>(slot);
            samples[index].seed = options[index].seed;
            samples[index].rng.seed(options[index].seed);
            samples[index].previous_main.assign(static_cast<size_t>(kRasWindow), 0);
            generated[index].reserve(static_cast<size_t>(max_new[index] * codebooks));
            positions[index] = static_cast<int32_t>(prompts[index].steps);
            rows[index] = prompts[index].steps;
        }

        std::vector<float> fast_input(static_cast<size_t>(slots * fast_dim), 0.0F);
        std::vector<float> fast_logits;
        std::vector<float> slot_logits;
        std::vector<char> semantic(static_cast<size_t>(slots), 0);
        std::vector<int32_t> codes(static_cast<size_t>(slots), 0);

        // Ein Rahmen fuer alle Bahnen. Der Zufallszustand liegt je Bahn in
        // samples[]; eine Bahn zieht genau dann eine Zahl, wenn sie es auch
        // allein taete, sonst liefe ihr Zaehler anders und der Seed haette
        // eine andere Bedeutung im Stapel als ausserhalb.
        const auto sample_batch_frame = [&](
            const std::function<const float *(int64_t)> & slow_logits_at,
            const std::function<const float *(int64_t)> & slow_hidden_at,
            bool apply_ras) {
            std::fill(semantic.begin(), semantic.end(), 0);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (active[index] == 0) {
                    continue;
                }
                const float * base = slow_logits_at(slot);
                slot_logits.assign(base, base + vocab);
                auto timing_start = Clock::now();
                const auto biased = apply_semantic_bias(assets.config, im_end_id(), slot_logits);
                profile.sample_bias_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
                timing_start = Clock::now();
                int32_t main_token = sample_from_logits(
                    biased,
                    options[index].temperature,
                    options[index].top_p,
                    options[index].top_k,
                    samples[index],
                    sampling_policy_);
                profile.sample_main_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
                timing_start = Clock::now();
                const int32_t high_token = sample_from_logits(
                    biased, kRasHighTemperature, kRasHighTopP, options[index].top_k, samples[index], sampling_policy_);
                profile.sample_high_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
                if (apply_ras && is_semantic_token(assets.config, main_token) &&
                    std::find(samples[index].previous_main.begin(), samples[index].previous_main.end(), main_token) !=
                        samples[index].previous_main.end()) {
                    main_token = high_token;
                }
                std::rotate(
                    samples[index].previous_main.begin(),
                    samples[index].previous_main.begin() + 1,
                    samples[index].previous_main.end());
                samples[index].previous_main.back() = main_token;
                std::fill(frames[index].begin(), frames[index].end(), 0);
                frames[index][0] = main_token;
                semantic[index] = is_semantic_token(assets.config, main_token) ? 1 : 0;
            }
            if (std::find(semantic.begin(), semantic.end(), 1) == semantic.end()) {
                return;
            }
            std::fill(fast_input.begin(), fast_input.end(), 0.0F);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (semantic[index] == 0) {
                    continue;
                }
                const float * source = slow_hidden_at(slot);
                std::copy(
                    source, source + fast_dim, fast_input.begin() + static_cast<std::ptrdiff_t>(slot * fast_dim));
            }
            batch_fast_graph_->run(fast_input, 0, fast_logits, profile);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (semantic[index] == 0) {
                    continue;
                }
                codes[index] = std::clamp<int32_t>(
                    frames[index][0] - static_cast<int32_t>(assets.config.semantic_start_token_id),
                    0,
                    static_cast<int32_t>(fast_vocab - 1));
                frames[index][1] = codes[index];
            }
            for (int64_t codebook = 1; codebook < codebooks; ++codebook) {
                std::fill(fast_input.begin(), fast_input.end(), 0.0F);
                for (int64_t slot = 0; slot < slots; ++slot) {
                    const auto index = static_cast<size_t>(slot);
                    if (semantic[index] == 0) {
                        continue;
                    }
                    auto timing_start = Clock::now();
                    const auto embedding = build_fast_embedding(assets.config, weights, codes[index]);
                    profile.fast_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
                    std::copy(
                        embedding.begin(),
                        embedding.end(),
                        fast_input.begin() + static_cast<std::ptrdiff_t>(slot * fast_dim));
                }
                batch_fast_graph_->run(fast_input, codebook, fast_logits, profile);
                for (int64_t slot = 0; slot < slots; ++slot) {
                    const auto index = static_cast<size_t>(slot);
                    if (semantic[index] == 0) {
                        continue;
                    }
                    const auto begin = fast_logits.begin() + static_cast<std::ptrdiff_t>(slot * fast_vocab);
                    const std::vector<float> logits(begin, begin + fast_vocab);
                    auto timing_start = Clock::now();
                    codes[index] = sample_from_logits(
                        logits,
                        options[index].temperature,
                        options[index].top_p,
                        options[index].top_k,
                        samples[index],
                        sampling_policy_);
                    profile.sample_fast_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
                    frames[index][static_cast<size_t>(codebook + 1)] = codes[index];
                }
            }
        };

        sample_batch_frame(
            [&](int64_t slot) { return prefill_logits[static_cast<size_t>(slot)].data(); },
            [&](int64_t slot) { return prefill_hidden[static_cast<size_t>(slot)].data(); },
            false);
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto index = static_cast<size_t>(slot);
            if (frames[index][0] == im_end_id()) {
                active[index] = 0;
                ended[index] = 1;
                continue;
            }
            append_frame(generated[index], frames[index]);
            ++profile.generated_frames;
        }

        int64_t max_steps = 0;
        for (const int64_t value : max_new) {
            max_steps = std::max(max_steps, value);
        }
        std::vector<float> embeddings(static_cast<size_t>(slots * hidden_dim), 0.0F);
        std::vector<float> step_logits;
        std::vector<float> step_hidden;
        for (int64_t step = 1; step < max_steps; ++step) {
            bool any_active = false;
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (active[index] != 0 && step >= max_new[index]) {
                    active[index] = 0;
                }
                any_active = any_active || active[index] != 0;
            }
            if (!any_active) {
                break;
            }
            auto timing_start = Clock::now();
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                const auto input = build_slow_embedding_for_frame(assets.config, weights, frames[index]);
                std::copy(
                    input.begin(),
                    input.end(),
                    embeddings.begin() + static_cast<std::ptrdiff_t>(slot * hidden_dim));
            }
            profile.slow_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            batch_step_graph_->run(embeddings, positions, rows, step_logits, step_hidden, profile);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (active[index] == 0) {
                    continue;
                }
                ++positions[index];
                ++rows[index];
            }
            sample_batch_frame(
                [&](int64_t slot) { return step_logits.data() + static_cast<size_t>(slot * vocab); },
                [&](int64_t slot) { return step_hidden.data() + static_cast<size_t>(slot * hidden_dim); },
                true);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const auto index = static_cast<size_t>(slot);
                if (active[index] == 0) {
                    continue;
                }
                if (frames[index][0] == im_end_id()) {
                    active[index] = 0;
                    ended[index] = 1;
                    continue;
                }
                append_frame(generated[index], frames[index]);
                ++profile.generated_frames;
            }
        }

        std::vector<FishAudioCodes> results(static_cast<size_t>(slots));
        for (int64_t slot = 0; slot < slots; ++slot) {
            const auto index = static_cast<size_t>(slot);
            auto & frame_major = generated[index];
            if (ended[index] == 0 && !frame_major.empty()) {
                frame_major.resize(frame_major.size() - static_cast<size_t>(codebooks));
            }
            auto & out = results[index];
            out.codebooks = codebooks;
            out.frames = static_cast<int64_t>(frame_major.size()) / codebooks;
            out.codes.assign(static_cast<size_t>(out.codebooks * out.frames), 0);
            for (int64_t frame_index = 0; frame_index < out.frames; ++frame_index) {
                for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
                    out.codes[static_cast<size_t>(codebook * out.frames + frame_index)] =
                        frame_major[static_cast<size_t>(frame_index * out.codebooks + codebook)];
                }
            }
        }
        // Der Stapelspeicher ist gross (slots mal die volle Fensterlaenge). Er
        // wird hier freigegeben, damit der Codec danach noch Platz findet; auf
        // dieser Karte teilen sich mehrere Dienste den Speicher.
        batch_step_graph_.reset();
        batch_fast_graph_.reset();
        log_profile(profile);
        return results;
    }

    void release_runtime_graphs() {
        step_graph_.reset();
        prefill_graph_.reset();
        fast_graph_.reset();
        batch_step_graph_.reset();
        batch_fast_graph_.reset();
    }

private:
    class PrefillGraph {
    public:
        PrefillGraph(
            std::shared_ptr<const FishARWeightsRuntime> runtime,
            int64_t steps,
            FishPrefillCacheTarget target_cache,
            int64_t cache_offset = 0)
            : runtime_(std::move(runtime)),
              steps_(steps),
              cache_offset_(cache_offset),
              target_cache_(std::move(target_cache)) {
            const auto & assets = runtime_->assets();
            const auto & config = assets.config.text;
            if (target_cache_.keys.size() != runtime_->weights().slow_layers.size() ||
                target_cache_.values.size() != runtime_->weights().slow_layers.size()) {
                throw std::runtime_error("Fish Audio prefill target cache layer count mismatch");
            }
            ggml_init_params params{runtime_->graph_arena_bytes(), nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio AR prefill context");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.ar.prefill", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps_, config.dim}));
            input_ = input.tensor;
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps_);
            auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({steps_}), GGML_TYPE_I32);
            constants_ = std::make_unique<core::ConstantTensorCache>(
                runtime_->backend(),
                runtime_->threads(),
                "fish_audio.ar.prefill.constants",
                256ull * 1024ull * 1024ull);
            constants_->begin_graph();
            auto decoder = build_fish_causal_decoder(
                ctx,
                *constants_,
                input,
                positions_value,
                bind_slow_weights(*constants_, runtime_->weights(), config),
                make_slow_decoder_config(config, runtime_->backend_type()),
                assets.config.norm_fastlayer_input);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            for (size_t layer_index = 0; layer_index < decoder.state.layers.size(); ++layer_index) {
                const auto & layer = decoder.state.layers[layer_index];
                if (!layer.key.has_value() || !layer.value.has_value()) {
                    throw std::runtime_error("Fish Audio prefill decoder did not produce K/V state");
                }
                auto key_dest = runtime::view_transformer_kv_cache_steps(
                    ctx,
                    target_cache_.keys[layer_index],
                    cache_offset_,
                    steps_,
                    config.n_local_heads,
                    config.head_dim,
                    "Fish Audio prefill key cache",
                    target_cache_.keys[layer_index].type);
                auto value_dest = runtime::view_transformer_kv_cache_steps(
                    ctx,
                    target_cache_.values[layer_index],
                    cache_offset_,
                    steps_,
                    config.n_local_heads,
                    config.head_dim,
                    "Fish Audio prefill value cache",
                    target_cache_.values[layer_index].type);
                ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), layer.key->tensor, key_dest.tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), layer.value->tensor, value_dest.tensor));
            }
            hidden_ = decoder.hidden.tensor;
            logits_ = decoder.logits.tensor;
            ggml_set_output(hidden_);
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            ggml_build_forward_expand(graph_, hidden_);
            constants_->finish_graph();
            constants_->ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Fish Audio AR prefill graph");
            }
            auto positions = modules::qwen_position_ids(steps_);
            ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        }

        ~PrefillGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
        }

        SlowPrefillOutput run(const std::vector<float> & embeddings, FishARProfile & profile) {
            const auto & config = runtime_->assets().config.text;
            if (static_cast<int64_t>(embeddings.size()) != steps_ * config.dim) {
                throw std::runtime_error("Fish Audio prefill embedding size mismatch");
            }
            ++profile.prefill_runs;
            auto timing_start = Clock::now();
            ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
            profile.prefill_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "fish_audio.ar.prefill");
            ggml_backend_synchronize(runtime_->backend());
            profile.prefill_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Fish Audio AR prefill graph compute failed");
            }
            SlowPrefillOutput out;
            out.forward.logits.resize(static_cast<size_t>(config.vocab_size));
            out.forward.hidden.resize(static_cast<size_t>(config.dim));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, out.forward.logits.data(), 0, out.forward.logits.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_, out.forward.hidden.data(), 0, out.forward.hidden.size() * sizeof(float));
            profile.prefill_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return out;
        }

        int64_t steps() const noexcept { return steps_; }

    private:
        std::shared_ptr<const FishARWeightsRuntime> runtime_;
        int64_t steps_ = 0;
        int64_t cache_offset_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * hidden_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        std::unique_ptr<core::ConstantTensorCache> constants_;
        FishPrefillCacheTarget target_cache_;
    };

    class StepGraph {
    public:
        StepGraph(std::shared_ptr<const FishARWeightsRuntime> runtime, int64_t cache_steps)
            : runtime_(std::move(runtime)),
              cache_steps_(cache_steps) {
            ggml_init_params state_params{8ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            if (state_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio AR step state context");
            }
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio AR step context");
            }
            const auto & assets = runtime_->assets();
            const auto & config = assets.config.text;
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, 1, 1);
            position_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
            std::vector<core::TensorValue> cache_keys;
            std::vector<core::TensorValue> cache_values;
            cache_keys.reserve(runtime_->weights().slow_layers.size());
            cache_values.reserve(runtime_->weights().slow_layers.size());
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            for (size_t layer = 0; layer < runtime_->weights().slow_layers.size(); ++layer) {
                cache_keys.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        cache_steps_,
                        1),
                    core::TensorShape::from_dims({1, cache_steps_, config.n_local_heads, config.head_dim}),
                    cache_type));
                cache_values.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        cache_steps_,
                        1),
                    core::TensorShape::from_dims({1, cache_steps_, config.n_local_heads, config.head_dim}),
                    cache_type));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Fish Audio AR step state tensors");
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "fish_audio.ar.step", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto position_value = core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto mask_value = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, cache_steps_}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 65536, false);
            auto & constants = runtime_->slow_step_constants();
            constants.begin_graph();
            auto decoder = build_fish_static_decoder(
                ctx,
                graph_,
                input,
                position_value,
                bind_slow_weights(constants, runtime_->weights(), config),
                make_slow_decoder_config(config, runtime_->backend_type()),
                cache_steps_,
                mask_value,
                cache_slot_value,
                std::move(cache_keys),
                std::move(cache_values),
                assets.config.norm_fastlayer_input);
            cache_ = std::move(decoder.cache);
            hidden_ = decoder.hidden.tensor;
            logits_ = decoder.logits.tensor;
            ggml_set_output(hidden_);
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            ggml_build_forward_expand(graph_, hidden_);
            constants.finish_graph();
            constants.ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Fish Audio AR step tensors");
            }
            mask_scratch_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        }

        ~StepGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        int64_t cache_steps() const noexcept { return cache_steps_; }

        FishPrefillCacheTarget prefill_target_cache() const {
            FishPrefillCacheTarget out;
            out.keys.reserve(runtime_->weights().slow_layers.size());
            out.values.reserve(runtime_->weights().slow_layers.size());
            for (size_t layer = 0; layer < runtime_->weights().slow_layers.size(); ++layer) {
                out.keys.push_back(cache_.key_tensor(layer));
                out.values.push_back(cache_.value_tensor(layer));
            }
            return out;
        }

        void finish_prefill(int64_t steps) {
            cache_.retain_prefix(0);
            cache_.advance_after_direct_append(steps);
            const auto masked = ggml_fp32_to_fp16(-INFINITY);
            const auto visible = ggml_fp32_to_fp16(0.0F);
            std::fill(mask_scratch_.begin(), mask_scratch_.end(), masked);
            for (int64_t i = 0; i < cache_.valid_steps(); ++i) {
                mask_scratch_[static_cast<size_t>(i)] = visible;
            }
            ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
        }

        SlowForwardOutput run(const std::vector<float> & embedding, FishARProfile & profile) {
            const auto & config = runtime_->assets().config.text;
            if (static_cast<int64_t>(embedding.size()) != config.dim) {
                throw std::runtime_error("Fish Audio step embedding size mismatch");
            }
            if (cache_.valid_steps() >= cache_steps_) {
                throw std::runtime_error("Fish Audio step cache exceeds capacity");
            }
            ++profile.step_runs;
            auto timing_start = Clock::now();
            const int32_t pos = static_cast<int32_t>(cache_.current_end());
            ggml_backend_tensor_set(position_, &pos, 0, sizeof(pos));
            const int32_t cache_slot = static_cast<int32_t>(cache_.valid_steps());
            ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));
            const auto visible = ggml_fp32_to_fp16(0.0F);
            mask_scratch_[static_cast<size_t>(cache_.valid_steps())] = visible;
            ggml_backend_tensor_set(
                mask_,
                &visible,
                static_cast<size_t>(cache_.valid_steps()) * sizeof(ggml_fp16_t),
                sizeof(ggml_fp16_t));
            profile.step_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
            profile.step_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "fish_audio.ar.step");
            ggml_backend_synchronize(runtime_->backend());
            profile.step_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Fish Audio AR step graph compute failed");
            }
            cache_.advance_after_direct_append(1);
            SlowForwardOutput out;
            out.logits.resize(static_cast<size_t>(config.vocab_size));
            out.hidden.resize(static_cast<size_t>(config.dim));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            profile.step_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return out;
        }

    private:
        std::shared_ptr<const FishARWeightsRuntime> runtime_;
        int64_t cache_steps_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * position_ = nullptr;
        ggml_tensor * cache_slot_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * hidden_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        runtime::TransformerKVCache cache_;
        std::vector<ggml_fp16_t> mask_scratch_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    class FastGraph {
    public:
        explicit FastGraph(std::shared_ptr<const FishARWeightsRuntime> runtime)
            : runtime_(std::move(runtime)) {
            ggml_init_params state_params{8ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            if (state_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio fast AR state context");
            }
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio fast AR context");
            }
            const auto & config = runtime_->assets().config.fast;
            const auto & weights = runtime_->weights();
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, 1, 1);
            position_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, config.num_codebooks, 1, 1, 1);
            std::vector<core::TensorValue> cache_keys;
            std::vector<core::TensorValue> cache_values;
            cache_keys.reserve(weights.fast_layers.size());
            cache_values.reserve(weights.fast_layers.size());
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            for (size_t layer = 0; layer < weights.fast_layers.size(); ++layer) {
                cache_keys.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        config.num_codebooks,
                        1),
                    core::TensorShape::from_dims({1, config.num_codebooks, config.n_local_heads, config.head_dim}),
                    cache_type));
                cache_values.push_back(core::wrap_tensor(
                    ggml_new_tensor_4d(
                        state_ctx_.get(),
                        cache_type,
                        config.head_dim,
                        config.n_local_heads,
                        config.num_codebooks,
                        1),
                    core::TensorShape::from_dims({1, config.num_codebooks, config.n_local_heads, config.head_dim}),
                    cache_type));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Fish Audio fast AR state tensors");
            }
            for (const auto & cache : cache_keys) {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(cache.tensor)), 0);
                ggml_backend_tensor_set(cache.tensor, zeros.data(), 0, zeros.size());
            }
            for (const auto & cache : cache_values) {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(cache.tensor)), 0);
                ggml_backend_tensor_set(cache.tensor, zeros.data(), 0, zeros.size());
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "fish_audio.ar.fast", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto position_value = core::wrap_tensor(position_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            auto mask_value = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, config.num_codebooks}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 32768, false);
            auto & constants = runtime_->fast_constants();
            constants.begin_graph();
            modules::QwenCausalDecoderWeights decoder_weights;
            decoder_weights.stack.layers.reserve(weights.fast_layers.size());
            for (const auto & layer : weights.fast_layers) {
                decoder_weights.stack.layers.push_back(bind_fast_layer(constants, layer, config));
            }
            decoder_weights.final_norm = binding::norm_data(constants, weights.fast_norm);
            decoder_weights.lm_head = binding::linear_data(constants, weights.fast_output);
            auto decoder = build_fish_static_decoder(
                ctx,
                graph_,
                input,
                position_value,
                decoder_weights,
                make_fast_decoder_config(config, runtime_->backend_type()),
                config.num_codebooks,
                mask_value,
                position_value,
                std::move(cache_keys),
                std::move(cache_values),
                true);
            logits_ = decoder.logits.tensor;
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            constants.finish_graph();
            constants.ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Fish Audio fast AR graph");
            }
            mask_scratch_.assign(static_cast<size_t>(config.num_codebooks), ggml_fp32_to_fp16(-INFINITY));
        }

        ~FastGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        std::vector<float> run(const std::vector<float> & input, int64_t position, FishARProfile & profile) {
            const auto & config = runtime_->assets().config.fast;
            if (static_cast<int64_t>(input.size()) != config.dim) {
                throw std::runtime_error("Fish Audio fast AR input size mismatch");
            }
            ++profile.fast_runs;
            auto timing_start = Clock::now();
            const int32_t pos = static_cast<int32_t>(position);
            ggml_backend_tensor_set(position_, &pos, 0, sizeof(pos));
            const auto visible = ggml_fp32_to_fp16(0.0F);
            if (position == 0) {
                std::fill(mask_scratch_.begin(), mask_scratch_.end(), ggml_fp32_to_fp16(-INFINITY));
                mask_scratch_[0] = visible;
                ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
            } else {
                mask_scratch_[static_cast<size_t>(position)] = visible;
                ggml_backend_tensor_set(
                    mask_,
                    &visible,
                    static_cast<size_t>(position) * sizeof(ggml_fp16_t),
                    sizeof(ggml_fp16_t));
            }
            profile.fast_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            profile.fast_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status = core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "fish_audio.ar.fast");
            ggml_backend_synchronize(runtime_->backend());
            profile.fast_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Fish Audio fast AR graph compute failed");
            }
            std::vector<float> logits(static_cast<size_t>(config.vocab_size), 0.0F);
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
            profile.fast_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            return logits;
        }

    private:
        std::shared_ptr<const FishARWeightsRuntime> runtime_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * position_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        std::vector<ggml_fp16_t> mask_scratch_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    // Ein Schrittgraph fuer `slots` Sequenzen. Jede Sequenz haelt eine eigene
    // Bahn im KV-Speicher (ne[3]), eine eigene Maske und eine eigene Position.
    class BatchStepGraph {
    public:
        BatchStepGraph(
            std::shared_ptr<const FishARWeightsRuntime> runtime,
            int64_t slots,
            int64_t cache_steps,
            FishAudioBatchAxis axis)
            : runtime_(std::move(runtime)),
              slots_(slots),
              cache_steps_(cache_steps),
              axis_(axis) {
            const auto & assets = runtime_->assets();
            const auto & config = assets.config.text;
            ggml_init_params state_params{64ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            if (state_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio AR batch state context");
            }
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio AR batch context");
            }
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, slots_, 1);
            positions_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, slots_);
            cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I64, slots_);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, slots_);
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            const size_t layers = runtime_->weights().slow_layers.size();
            key_tensors_.reserve(layers);
            value_tensors_.reserve(layers);
            for (size_t layer = 0; layer < layers; ++layer) {
                key_tensors_.push_back(ggml_new_tensor_4d(
                    state_ctx_.get(), cache_type, config.head_dim, config.n_local_heads, cache_steps_, slots_));
                value_tensors_.push_back(ggml_new_tensor_4d(
                    state_ctx_.get(), cache_type, config.head_dim, config.n_local_heads, cache_steps_, slots_));
                // Flache Sicht ueber alle Bahnen; das Vorfuellen schreibt darin
                // mit dem Versatz slot * cache_steps.
                key_flat_.push_back(ggml_reshape_4d(
                    state_ctx_.get(), key_tensors_.back(),
                    config.head_dim, config.n_local_heads, cache_steps_ * slots_, 1));
                value_flat_.push_back(ggml_reshape_4d(
                    state_ctx_.get(), value_tensors_.back(),
                    config.head_dim, config.n_local_heads, cache_steps_ * slots_, 1));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Fish Audio AR batch state tensors");
            }
            {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(key_tensors_.front())), 0);
                for (auto * tensor : key_tensors_) {
                    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
                }
                for (auto * tensor : value_tensors_) {
                    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
                }
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "fish_audio.ar.batch.step", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, slots_, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({slots_}), GGML_TYPE_I32);
            auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({slots_}), GGML_TYPE_I64);
            auto mask_value = core::wrap_tensor(
                mask_, core::TensorShape::from_dims({slots_, 1, 1, cache_steps_}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 262144, false);
            constants_ = std::make_unique<core::ConstantTensorCache>(
                runtime_->backend(), runtime_->threads(), "fish_audio.ar.batch.step.constants", 256ull * 1024ull * 1024ull);
            constants_->begin_graph();
            const auto decoder_config = make_slow_decoder_config(config, runtime_->backend_type());
            const auto layer_config = modules::qwen_decoder_layer_config_from_stack(decoder_config.stack);
            const auto bound = bind_slow_weights(*constants_, runtime_->weights(), config);
            const auto cache_shape =
                core::TensorShape::from_dims({slots_, cache_steps_, config.n_local_heads, config.head_dim});
            auto x = input;
            for (size_t layer = 0; layer < layers; ++layer) {
                const FishBatchCacheBinding cache{
                    core::wrap_tensor(key_tensors_[layer], cache_shape, cache_type),
                    core::wrap_tensor(value_tensors_[layer], cache_shape, cache_type),
                };
                x = build_fish_batch_layer(
                    ctx, x, positions_value, bound.stack.layers[layer], layer_config, slots_, axis_, cache,
                    cache_slot_value, mask_value);
            }
            auto hidden_sequence = modules::RMSNormModule({config.dim, config.norm_eps, true, false})
                                       .build(ctx, x, bound.final_norm);
            auto logits = fish_batch_linear(
                ctx, hidden_sequence, slots_, axis_,
                {config.dim, config.vocab_size, false, GGML_PREC_F32}, bound.lm_head);
            const auto fast_hidden = assets.config.norm_fastlayer_input ? hidden_sequence : x;
            hidden_ = fast_hidden.tensor;
            logits_ = logits.tensor;
            ggml_set_output(hidden_);
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            ggml_build_forward_expand(graph_, hidden_);
            constants_->finish_graph();
            constants_->ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Fish Audio AR batch step tensors");
            }
            mask_scratch_.assign(static_cast<size_t>(slots_ * cache_steps_), ggml_fp32_to_fp16(-INFINITY));
            cache_slot_values_.assign(static_cast<size_t>(slots_), 0);
        }

        ~BatchStepGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        int64_t slots() const noexcept { return slots_; }
        int64_t cache_steps() const noexcept { return cache_steps_; }
        FishAudioBatchAxis axis() const noexcept { return axis_; }

        // Flache Sicht auf alle Bahnen; das Vorfuellen einer Bahn schreibt mit
        // dem Versatz slot * cache_steps hinein.
        FishPrefillCacheTarget prefill_target_cache() const {
            const auto & config = runtime_->assets().config.text;
            const auto shape = core::TensorShape::from_dims(
                {1, slots_ * cache_steps_, config.n_local_heads, config.head_dim});
            FishPrefillCacheTarget out;
            out.keys.reserve(key_flat_.size());
            out.values.reserve(value_flat_.size());
            for (size_t layer = 0; layer < key_flat_.size(); ++layer) {
                out.keys.push_back(core::wrap_tensor(key_flat_[layer], shape, key_flat_[layer]->type));
                out.values.push_back(core::wrap_tensor(value_flat_[layer], shape, value_flat_[layer]->type));
            }
            return out;
        }

        void reset_slot(int64_t slot, int64_t visible_steps) {
            const auto masked = ggml_fp32_to_fp16(-INFINITY);
            const auto visible = ggml_fp32_to_fp16(0.0F);
            auto * row = mask_scratch_.data() + static_cast<size_t>(slot * cache_steps_);
            std::fill(row, row + cache_steps_, masked);
            for (int64_t i = 0; i < visible_steps && i < cache_steps_; ++i) {
                row[i] = visible;
            }
            ggml_backend_tensor_set(
                mask_,
                row,
                static_cast<size_t>(slot * cache_steps_) * sizeof(ggml_fp16_t),
                static_cast<size_t>(cache_steps_) * sizeof(ggml_fp16_t));
        }

        void mark_visible(int64_t slot, int64_t row_index) {
            const auto visible = ggml_fp32_to_fp16(0.0F);
            const size_t offset = static_cast<size_t>(slot * cache_steps_ + row_index);
            if (mask_scratch_[offset] == visible) {
                return;
            }
            mask_scratch_[offset] = visible;
            ggml_backend_tensor_set(mask_, &visible, offset * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
        }

        void run(
            const std::vector<float> & embeddings,
            const std::vector<int32_t> & positions,
            const std::vector<int64_t> & rows,
            std::vector<float> & logits_out,
            std::vector<float> & hidden_out,
            FishARProfile & profile) {
            const auto & config = runtime_->assets().config.text;
            if (static_cast<int64_t>(embeddings.size()) != slots_ * config.dim) {
                throw std::runtime_error("Fish Audio batch step embedding size mismatch");
            }
            ++profile.step_runs;
            auto timing_start = Clock::now();
            ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
            for (int64_t slot = 0; slot < slots_; ++slot) {
                cache_slot_values_[static_cast<size_t>(slot)] = slot * cache_steps_ + rows[static_cast<size_t>(slot)];
                mark_visible(slot, rows[static_cast<size_t>(slot)]);
            }
            ggml_backend_tensor_set(
                cache_slot_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int64_t));
            profile.step_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, embeddings.data(), 0, embeddings.size() * sizeof(float));
            profile.step_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status =
                core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "fish_audio.ar.batch.step");
            ggml_backend_synchronize(runtime_->backend());
            profile.step_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Fish Audio AR batch step graph compute failed");
            }
            logits_out.resize(static_cast<size_t>(slots_ * config.vocab_size));
            hidden_out.resize(static_cast<size_t>(slots_ * config.dim));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, logits_out.data(), 0, logits_out.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_, hidden_out.data(), 0, hidden_out.size() * sizeof(float));
            profile.step_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        }

    private:
        std::shared_ptr<const FishARWeightsRuntime> runtime_;
        int64_t slots_ = 0;
        int64_t cache_steps_ = 0;
        FishAudioBatchAxis axis_ = FishAudioBatchAxis::Token;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * cache_slot_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * hidden_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        std::vector<ggml_tensor *> key_tensors_;
        std::vector<ggml_tensor *> value_tensors_;
        std::vector<ggml_tensor *> key_flat_;
        std::vector<ggml_tensor *> value_flat_;
        std::vector<ggml_fp16_t> mask_scratch_;
        std::vector<int64_t> cache_slot_values_;
        std::unique_ptr<core::ConstantTensorCache> constants_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    // Der schnelle Decoder fuer `slots` Sequenzen. Alle Sequenzen stehen im
    // selben Codebuch-Schritt, die Position ist also gemeinsam; der KV-Speicher
    // und die Maske bleiben trotzdem je Sequenz getrennt.
    class BatchFastGraph {
    public:
        BatchFastGraph(
            std::shared_ptr<const FishARWeightsRuntime> runtime, int64_t slots, FishAudioBatchAxis axis)
            : runtime_(std::move(runtime)),
              slots_(slots),
              axis_(axis) {
            const auto & config = runtime_->assets().config.fast;
            const auto & weights = runtime_->weights();
            cache_steps_ = config.num_codebooks;
            ggml_init_params state_params{64ull * 1024ull * 1024ull, nullptr, true};
            state_ctx_.reset(ggml_init(state_params));
            ggml_init_params graph_params{runtime_->graph_arena_bytes(), nullptr, true};
            graph_ctx_.reset(ggml_init(graph_params));
            if (state_ctx_ == nullptr || graph_ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize Fish Audio fast AR batch context");
            }
            input_ = ggml_new_tensor_3d(state_ctx_.get(), GGML_TYPE_F32, config.dim, slots_, 1);
            positions_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, slots_);
            cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I64, slots_);
            mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, slots_);
            const ggml_type cache_type =
                runtime_->backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F32 : GGML_TYPE_BF16;
            const size_t layers = weights.fast_layers.size();
            for (size_t layer = 0; layer < layers; ++layer) {
                key_tensors_.push_back(ggml_new_tensor_4d(
                    state_ctx_.get(), cache_type, config.head_dim, config.n_local_heads, cache_steps_, slots_));
                value_tensors_.push_back(ggml_new_tensor_4d(
                    state_ctx_.get(), cache_type, config.head_dim, config.n_local_heads, cache_steps_, slots_));
            }
            state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), runtime_->backend());
            if (state_buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate Fish Audio fast AR batch state tensors");
            }
            {
                std::vector<uint8_t> zeros(static_cast<size_t>(ggml_nbytes(key_tensors_.front())), 0);
                for (auto * tensor : key_tensors_) {
                    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
                }
                for (auto * tensor : value_tensors_) {
                    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
                }
            }

            core::ModuleBuildContext ctx{graph_ctx_.get(), "fish_audio.ar.batch.fast", runtime_->backend_type()};
            auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, slots_, config.dim}));
            input = core::wrap_tensor(ggml_cpy(ctx.ggml, input_, input.tensor), input.shape, input.type);
            auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({slots_}), GGML_TYPE_I32);
            auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({slots_}), GGML_TYPE_I64);
            auto mask_value = core::wrap_tensor(
                mask_, core::TensorShape::from_dims({slots_, 1, 1, cache_steps_}), GGML_TYPE_F16);
            graph_ = ggml_new_graph_custom(graph_ctx_.get(), 65536, false);
            constants_ = std::make_unique<core::ConstantTensorCache>(
                runtime_->backend(), runtime_->threads(), "fish_audio.ar.batch.fast.constants", 256ull * 1024ull * 1024ull);
            constants_->begin_graph();
            modules::QwenCausalDecoderWeights decoder_weights;
            decoder_weights.stack.layers.reserve(layers);
            for (const auto & layer : weights.fast_layers) {
                decoder_weights.stack.layers.push_back(bind_fast_layer(*constants_, layer, config));
            }
            decoder_weights.final_norm = binding::norm_data(*constants_, weights.fast_norm);
            decoder_weights.lm_head = binding::linear_data(*constants_, weights.fast_output);
            const auto decoder_config = make_fast_decoder_config(config, runtime_->backend_type());
            const auto layer_config = modules::qwen_decoder_layer_config_from_stack(decoder_config.stack);
            const auto cache_shape =
                core::TensorShape::from_dims({slots_, cache_steps_, config.n_local_heads, config.head_dim});
            auto x = input;
            for (size_t layer = 0; layer < layers; ++layer) {
                const FishBatchCacheBinding cache{
                    core::wrap_tensor(key_tensors_[layer], cache_shape, cache_type),
                    core::wrap_tensor(value_tensors_[layer], cache_shape, cache_type),
                };
                x = build_fish_batch_layer(
                    ctx, x, positions_value, decoder_weights.stack.layers[layer], layer_config, slots_, axis_,
                    cache, cache_slot_value, mask_value);
            }
            auto hidden = modules::RMSNormModule({config.dim, config.norm_eps, true, false})
                              .build(ctx, x, decoder_weights.final_norm);
            auto logits = fish_batch_linear(
                ctx, hidden, slots_, axis_,
                {config.dim, config.vocab_size, false, GGML_PREC_F32}, decoder_weights.lm_head);
            logits_ = logits.tensor;
            ggml_set_output(logits_);
            ggml_build_forward_expand(graph_, logits_);
            constants_->finish_graph();
            constants_->ensure_uploaded();
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("failed to allocate Fish Audio fast AR batch graph");
            }
            mask_scratch_.assign(static_cast<size_t>(slots_ * cache_steps_), ggml_fp32_to_fp16(-INFINITY));
            cache_slot_values_.assign(static_cast<size_t>(slots_), 0);
            positions_scratch_.assign(static_cast<size_t>(slots_), 0);
        }

        ~BatchFastGraph() {
            core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
            if (state_buffer_ != nullptr) {
                ggml_backend_buffer_free(state_buffer_);
            }
        }

        int64_t slots() const noexcept { return slots_; }
        FishAudioBatchAxis axis() const noexcept { return axis_; }

        void run(
            const std::vector<float> & input,
            int64_t position,
            std::vector<float> & logits_out,
            FishARProfile & profile) {
            const auto & config = runtime_->assets().config.fast;
            if (static_cast<int64_t>(input.size()) != slots_ * config.dim) {
                throw std::runtime_error("Fish Audio fast AR batch input size mismatch");
            }
            ++profile.fast_runs;
            auto timing_start = Clock::now();
            const auto masked = ggml_fp32_to_fp16(-INFINITY);
            const auto visible = ggml_fp32_to_fp16(0.0F);
            if (position == 0) {
                std::fill(mask_scratch_.begin(), mask_scratch_.end(), masked);
                for (int64_t slot = 0; slot < slots_; ++slot) {
                    mask_scratch_[static_cast<size_t>(slot * cache_steps_)] = visible;
                }
                ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
            } else {
                for (int64_t slot = 0; slot < slots_; ++slot) {
                    const size_t offset = static_cast<size_t>(slot * cache_steps_ + position);
                    mask_scratch_[offset] = visible;
                    ggml_backend_tensor_set(mask_, &visible, offset * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                }
            }
            for (int64_t slot = 0; slot < slots_; ++slot) {
                positions_scratch_[static_cast<size_t>(slot)] = static_cast<int32_t>(position);
                cache_slot_values_[static_cast<size_t>(slot)] = slot * cache_steps_ + position;
            }
            ggml_backend_tensor_set(
                positions_, positions_scratch_.data(), 0, positions_scratch_.size() * sizeof(int32_t));
            ggml_backend_tensor_set(
                cache_slot_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int64_t));
            profile.fast_mask_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            timing_start = Clock::now();
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            profile.fast_input_upload_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            timing_start = Clock::now();
            const ggml_status status =
                core::compute_backend_graph(runtime_->backend(), graph_, nullptr, "fish_audio.ar.batch.fast");
            ggml_backend_synchronize(runtime_->backend());
            profile.fast_graph_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Fish Audio fast AR batch graph compute failed");
            }
            logits_out.resize(static_cast<size_t>(slots_ * config.vocab_size));
            timing_start = Clock::now();
            ggml_backend_tensor_get(logits_, logits_out.data(), 0, logits_out.size() * sizeof(float));
            profile.fast_output_read_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        }

    private:
        std::shared_ptr<const FishARWeightsRuntime> runtime_;
        int64_t slots_ = 0;
        int64_t cache_steps_ = 0;
        FishAudioBatchAxis axis_ = FishAudioBatchAxis::Token;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * cache_slot_ = nullptr;
        ggml_tensor * mask_ = nullptr;
        ggml_tensor * logits_ = nullptr;
        std::vector<ggml_tensor *> key_tensors_;
        std::vector<ggml_tensor *> value_tensors_;
        std::vector<ggml_fp16_t> mask_scratch_;
        std::vector<int64_t> cache_slot_values_;
        std::vector<int32_t> positions_scratch_;
        std::unique_ptr<core::ConstantTensorCache> constants_;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_backend_buffer_t state_buffer_ = nullptr;
    };

    void ensure_prefill_graph(int64_t steps, FishARProfile & profile) {
        if (!prefill_graph_ || prefill_graph_->steps() != steps) {
            const auto build_start = Clock::now();
            if (!step_graph_) {
                throw std::runtime_error("Fish Audio AR prefill requires a step graph");
            }
            prefill_graph_ = std::make_unique<PrefillGraph>(runtime_, steps, step_graph_->prefill_target_cache());
            profile.graph_build_prefill_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    void ensure_step_graph(int64_t cache_steps, FishARProfile & profile) {
        if (!step_graph_ || step_graph_->cache_steps() < cache_steps) {
            const auto build_start = Clock::now();
            step_graph_ = std::make_unique<StepGraph>(runtime_, cache_steps);
            prefill_graph_.reset();
            profile.graph_build_step_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    void ensure_fast_graph(FishARProfile & profile) {
        if (!fast_graph_) {
            const auto build_start = Clock::now();
            fast_graph_ = std::make_unique<FastGraph>(runtime_);
            profile.graph_build_fast_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    void ensure_batch_graphs(
        int64_t slots, int64_t cache_steps, FishAudioBatchAxis axis, FishARProfile & profile) {
        if (batch_step_graph_ &&
            (batch_step_graph_->slots() != slots || batch_step_graph_->cache_steps() < cache_steps ||
             batch_step_graph_->axis() != axis)) {
            batch_step_graph_.reset();
        }
        if (!batch_step_graph_) {
            const auto build_start = Clock::now();
            batch_step_graph_ = std::make_unique<BatchStepGraph>(runtime_, slots, cache_steps, axis);
            profile.graph_build_step_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
        if (batch_fast_graph_ && (batch_fast_graph_->slots() != slots || batch_fast_graph_->axis() != axis)) {
            batch_fast_graph_.reset();
        }
        if (!batch_fast_graph_) {
            const auto build_start = Clock::now();
            batch_fast_graph_ = std::make_unique<BatchFastGraph>(runtime_, slots, axis);
            profile.graph_build_fast_ms += engine::debug::elapsed_ms(build_start, Clock::now());
        }
    }

    int32_t im_end_id() const {
        return static_cast<int32_t>(runtime_->assets().config.im_end_token_id);
    }

    void append_frame(std::vector<int32_t> & out, const std::vector<int32_t> & frame) const {
        if (static_cast<int64_t>(frame.size()) != runtime_->assets().config.fast.num_codebooks + 1) {
            throw std::runtime_error("Fish Audio generated frame shape mismatch");
        }
        out.insert(out.end(), frame.begin() + 1, frame.end());
    }

    std::vector<int32_t> sample_frame(
        const std::vector<float> & slow_logits,
        const std::vector<float> & slow_hidden,
        const FishAudioGenerationOptions & options,
        SampleState & sample,
        bool apply_ras,
        FishARProfile & profile) {
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        auto timing_start = Clock::now();
        const auto biased = apply_semantic_bias(config, im_end_id(), slow_logits);
        profile.sample_bias_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        timing_start = Clock::now();
        int32_t main_token = sample_from_logits(
            biased,
            options.temperature,
            options.top_p,
            options.top_k,
            sample,
            sampling_policy_);
        profile.sample_main_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        timing_start = Clock::now();
        const int32_t high_token = sample_from_logits(
            biased,
            kRasHighTemperature,
            kRasHighTopP,
            options.top_k,
            sample,
            sampling_policy_);
        profile.sample_high_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (apply_ras && is_semantic_token(config, main_token) &&
            std::find(sample.previous_main.begin(), sample.previous_main.end(), main_token) != sample.previous_main.end()) {
            main_token = high_token;
        }
        std::rotate(sample.previous_main.begin(), sample.previous_main.begin() + 1, sample.previous_main.end());
        sample.previous_main.back() = main_token;

        std::vector<int32_t> frame(static_cast<size_t>(config.fast.num_codebooks + 1), 0);
        frame[0] = main_token;
        if (!is_semantic_token(config, main_token)) {
            return frame;
        }
        const auto fast0_logits = fast_graph_->run(slow_hidden, 0, profile);
        int32_t code = std::clamp<int32_t>(
            main_token - static_cast<int32_t>(config.semantic_start_token_id),
            0,
            static_cast<int32_t>(config.fast.vocab_size - 1));
        frame[1] = code;
        for (int64_t codebook = 1; codebook < config.fast.num_codebooks; ++codebook) {
            timing_start = Clock::now();
            const auto embedding = build_fast_embedding(config, weights, code);
            profile.fast_embedding_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            const auto logits = fast_graph_->run(embedding, codebook, profile);
            timing_start = Clock::now();
            code = sample_from_logits(
                logits,
                options.temperature,
                options.top_p,
                options.top_k,
                sample,
                sampling_policy_);
            profile.sample_fast_ms += engine::debug::elapsed_ms(timing_start, Clock::now());
            frame[static_cast<size_t>(codebook + 1)] = code;
        }
        return frame;
    }

    void log_profile(const FishARProfile & profile) const {
        engine::debug::timing_log_scalar("fish_audio.ar.profile.graph_build_prefill_ms", profile.graph_build_prefill_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.graph_build_step_ms", profile.graph_build_step_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.graph_build_fast_ms", profile.graph_build_fast_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.slow_embedding_ms", profile.slow_embedding_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.fast_embedding_ms", profile.fast_embedding_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.prefill_input_upload_ms", profile.prefill_input_upload_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.prefill_graph_ms", profile.prefill_graph_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.prefill_output_read_ms", profile.prefill_output_read_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.step_input_upload_ms", profile.step_input_upload_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.step_mask_upload_ms", profile.step_mask_upload_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.step_graph_ms", profile.step_graph_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.step_output_read_ms", profile.step_output_read_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.fast_input_upload_ms", profile.fast_input_upload_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.fast_mask_upload_ms", profile.fast_mask_upload_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.fast_graph_ms", profile.fast_graph_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.fast_output_read_ms", profile.fast_output_read_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.sample_bias_ms", profile.sample_bias_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.sample_main_ms", profile.sample_main_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.sample_high_ms", profile.sample_high_ms);
        engine::debug::timing_log_scalar("fish_audio.ar.profile.sample_fast_ms", profile.sample_fast_ms);
        engine::debug::trace_log_scalar("fish_audio.ar.profile.prefill_runs", profile.prefill_runs);
        engine::debug::trace_log_scalar("fish_audio.ar.profile.step_runs", profile.step_runs);
        engine::debug::trace_log_scalar("fish_audio.ar.profile.fast_runs", profile.fast_runs);
        engine::debug::trace_log_scalar("fish_audio.ar.profile.generated_frames", profile.generated_frames);
    }

    std::shared_ptr<const FishARWeightsRuntime> runtime_;
    sampling::TorchCudaSamplingPolicy sampling_policy_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<StepGraph> step_graph_;
    std::unique_ptr<FastGraph> fast_graph_;
    std::unique_ptr<BatchStepGraph> batch_step_graph_;
    std::unique_ptr<BatchFastGraph> batch_fast_graph_;
};

FishAudioARRuntime::FishAudioARRuntime(
    std::shared_ptr<const FishAudioAssets> assets,
    core::BackendConfig backend,
    int threads,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          backend,
          threads,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

FishAudioARRuntime::~FishAudioARRuntime() = default;

FishAudioCodes FishAudioARRuntime::generate(
    const FishAudioPrompt & prompt,
    const FishAudioGenerationOptions & options) {
    return impl_->generate(prompt, options);
}

std::vector<FishAudioCodes> FishAudioARRuntime::generate_batch(
    const std::vector<FishAudioPrompt> & prompts,
    const std::vector<FishAudioGenerationOptions> & options,
    FishAudioBatchAxis axis) {
    return impl_->generate_batch(prompts, options, axis);
}

void FishAudioARRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::fish_audio
