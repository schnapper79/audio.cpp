#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/types.h"

#include <memory>
#include <vector>

namespace engine::models::fish_audio {

class FishAudioARRuntime {
public:
    FishAudioARRuntime(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~FishAudioARRuntime();

    FishAudioCodes generate(const FishAudioPrompt & prompt, const FishAudioGenerationOptions & options);

    // Zieht mehrere Sequenzen gemeinsam durch den langsamen und den schnellen
    // Decoder. Der Gewinn kommt allein daher, dass ein Gewichtslesen mehrere
    // Sequenzen bedient: die Schrittgraphen sind speicherbandbreitengebunden.
    //
    // Der Zufallszustand bleibt je Sequenz. Jede Sequenz hat ihre eigene
    // Positionsfolge, ihren eigenen KV-Speicher und ihre eigene Maske, deshalb
    // haengt ihr Ergebnis nicht davon ab, wer sonst im Stapel sitzt.
    std::vector<FishAudioCodes> generate_batch(
        const std::vector<FishAudioPrompt> & prompts,
        const std::vector<FishAudioGenerationOptions> & options,
        FishAudioBatchAxis axis);

    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
