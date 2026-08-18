#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::fish_audio {

// Wo im Tensor die Sequenzen eines Stapels nebeneinander liegen.
//
// Token:   in ggml ne[1]. Ein Gewichtslesen bedient alle Sequenzen, weil
//          ncols_dst gleich ne[1] ist. Dafuer wechselt ggml oberhalb von vier
//          Spalten die Warp-Aufteilung, und die Summen laufen anders.
// Channel: in ggml ne[2]. ncols_dst bleibt 1, die Warp-Regel greift also nie
//          und der Stapel darf groesser werden -- aber jede Sequenz liest die
//          Gewichte erneut. So stapeln higgs_audio_tts und qwen3_tts.
enum class FishAudioBatchAxis {
    Token,
    Channel,
};

struct FishAudioGenerationOptions {
    int64_t max_new_tokens = 1024;
    int64_t text_chunk_size = 200;
    float top_p = 0.8F;
    int top_k = 30;
    float temperature = 0.8F;
    uint32_t seed = 1234;
};

struct FishAudioReference {
    std::optional<runtime::AudioBuffer> audio = std::nullopt;
    std::string text;
    std::string cache_id;
};

struct FishAudioRequest {
    std::string text;
    std::optional<FishAudioReference> reference = std::nullopt;
    FishAudioGenerationOptions generation;
};

struct FishAudioCodes {
    std::vector<int32_t> codes;
    int64_t codebooks = 0;
    int64_t frames = 0;
};

struct FishAudioConversationTurn {
    std::string text;
    FishAudioCodes codes;
};

struct FishAudioPrompt {
    std::vector<int32_t> matrix;
    int64_t codebook_rows = 0;
    int64_t steps = 0;
    std::string text;
};

struct FishAudioTextConfig {
    int64_t vocab_size = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = true;
    bool attention_qk_norm = true;
};

struct FishAudioFastConfig {
    int64_t vocab_size = 0;
    int64_t num_codebooks = 0;
    int64_t n_layer = 0;
    int64_t dim = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_local_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    float rope_base = 1000000.0F;
    float norm_eps = 1.0e-6F;
    bool tie_word_embeddings = false;
    bool attention_qk_norm = false;
};

struct FishAudioCodecConfig {
    int sample_rate = 44100;
    int64_t semantic_codebook_size = 4096;
    int64_t residual_codebook_size = 1024;
    int64_t quantizer_codebooks = 9;
    int64_t total_codebooks = 10;
    int64_t codebook_dim = 8;
    int64_t latent_dim = 1024;
    int64_t frame_length = 2048;
};

struct FishAudioConfig {
    std::string model_type;
    std::string torch_dtype;
    int64_t semantic_start_token_id = 0;
    int64_t semantic_end_token_id = 0;
    int64_t im_end_token_id = 0;
    bool norm_fastlayer_input = false;
    FishAudioTextConfig text;
    FishAudioFastConfig fast;
    FishAudioCodecConfig codec;
};

}  // namespace engine::models::fish_audio
