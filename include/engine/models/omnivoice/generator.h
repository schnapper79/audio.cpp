#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/types.h"

#include <memory>

namespace engine::models::omnivoice {

struct OmniVoiceGeneratorRuntimeStats {
    bool graph_rebuilt = false;
    int64_t total_token_capacity = 0;
    int64_t target_frame_capacity = 0;
    double rebuild_ms = 0.0;
    double rebuild_clear_ms = 0.0;
    double rebuild_build_ms = 0.0;
    double rebuild_alloc_ms = 0.0;
    double rebuild_init_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
};

enum class OmniVoiceGeneratorPerfMode {
    Standard,
    FlashAttention,
};

class OmniVoiceGeneratorRuntime {
public:
    OmniVoiceGeneratorRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type,
        bool mem_saver,
        OmniVoiceGeneratorPerfMode perf_mode);
    ~OmniVoiceGeneratorRuntime();

    OmniVoiceGeneratedAudioTokens generate(
        const OmniVoicePrompt & prompt,
        const OmniVoiceGenerationOptions & options);

    // Mehrere Prompts gemeinsam durch die Masken-Diffusion ziehen. Omnivoice
    // ist kein AR-Modell: jede Anfrage packt ihre Sequenz in EINE Batch-Achse
    // (CFG braucht ohnehin zwei Bahnen), und der Stapel verbreitert diese Achse
    // von 2 auf 2*N Bahnen - jedes Gewichtslesen der vollen Vorwaertslaeufe
    // bedient dann alle Anfragen gleichzeitig. Der Zufallsgenerator und der
    // Entmaskungs-Zeitplan bleiben je Anfrage isoliert; options[i].seed
    // steuert Bahn i direkt (statt des shared Zustands im Einzelpfad).
    std::vector<OmniVoiceGeneratedAudioTokens> generate_batch(
        const std::vector<OmniVoicePrompt> & prompts,
        const std::vector<OmniVoiceGenerationOptions> & options);

    void release_runtime_graphs();
    void seed_rng(uint32_t seed);
    const OmniVoiceGeneratorRuntimeStats & last_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
