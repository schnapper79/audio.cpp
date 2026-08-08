#pragma once

#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/speaker_encoder.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"
#include "engine/models/qwen3_tts/types.h"

namespace engine::models::qwen3_tts {

class Qwen3TTSCustomVoicePromptBuilder {
public:
    Qwen3TTSCustomVoicePromptBuilder(
        const Qwen3TextTokenizer & tokenizer,
        const Qwen3SpeakerEncoderRuntime * speaker_encoder,
        int64_t text_token_limit,
        int64_t instruction_token_limit);

    // `precomputed_speaker_embedding` skips the speaker-encoder pass for a
    // cloned voice; the batched session uses it to encode a request's
    // reference clip once instead of once per text chunk.
    Qwen3TalkerPrefill build_prefill(
        const Qwen3TTSRequest & request,
        const Qwen3SpeakerEmbedding * precomputed_speaker_embedding = nullptr) const;

private:
    const Qwen3TextTokenizer & tokenizer_;
    const Qwen3SpeakerEncoderRuntime * speaker_encoder_ = nullptr;
    int64_t text_token_limit_ = 0;
    int64_t instruction_token_limit_ = 0;
};

}  // namespace engine::models::qwen3_tts
