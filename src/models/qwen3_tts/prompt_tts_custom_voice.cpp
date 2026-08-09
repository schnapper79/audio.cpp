#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"

#include <stdexcept>
#include <string>

namespace engine::models::qwen3_tts {
namespace {

void require_token_limit(size_t actual, int64_t limit, const char * what) {
    if (limit <= 0) {
        throw std::runtime_error(std::string("Qwen3 custom voice ") + what + " token limit must be positive");
    }
    if (actual > static_cast<size_t>(limit)) {
        throw std::runtime_error(
            std::string("Qwen3 custom voice ") + what + " token count "
            + std::to_string(actual) + " exceeds limit " + std::to_string(limit));
    }
}

}  // namespace

Qwen3TTSCustomVoicePromptBuilder::Qwen3TTSCustomVoicePromptBuilder(
    const Qwen3TextTokenizer & tokenizer,
    const Qwen3SpeakerEncoderRuntime * speaker_encoder,
    int64_t text_token_limit,
    int64_t instruction_token_limit)
    : tokenizer_(tokenizer),
      speaker_encoder_(speaker_encoder),
      text_token_limit_(text_token_limit),
      instruction_token_limit_(instruction_token_limit) {}

Qwen3TalkerPrefill Qwen3TTSCustomVoicePromptBuilder::build_prefill(
    const Qwen3TTSRequest & request,
    const Qwen3SpeakerEmbedding * precomputed_speaker_embedding) const {
    if (!request.custom_voice.has_value()) {
        throw std::runtime_error("Qwen3 custom voice prefill requires custom voice input");
    }
    // A voice comes from reference audio (encoded here), or directly as a
    // precomputed embedding vector - the latter needs neither audio nor the
    // speaker encoder and is what embedding import/mixing feeds in.
    const bool cloned =
        request.custom_voice->reference_audio.has_value() || precomputed_speaker_embedding != nullptr;
    if (request.custom_voice->speaker.empty() && !cloned) {
        throw std::runtime_error(
            "Qwen3 custom voice prefill requires speaker, reference audio, or speaker embedding");
    }
    if (cloned && precomputed_speaker_embedding == nullptr && speaker_encoder_ == nullptr) {
        throw std::runtime_error(
            "Qwen3 custom voice reference audio needs speaker encoder weights - "
            "this checkpoint has none");
    }
    Qwen3TalkerPrefill prefill;
    prefill.prompt_mode = Qwen3TalkerPromptMode::CustomVoice;
    prefill.input_ids = tokenizer_.encode(tokenizer_.build_assistant_prompt(request.text));
    require_token_limit(prefill.input_ids.size(), text_token_limit_, "text");
    if (!request.custom_voice->instruct.empty()) {
        prefill.instruct_ids = tokenizer_.encode(tokenizer_.build_instruct_prompt(request.custom_voice->instruct));
        require_token_limit(prefill.instruct_ids.size(), instruction_token_limit_, "instruction");
    }
    if (cloned) {
        prefill.speaker_embedding = precomputed_speaker_embedding != nullptr
            ? *precomputed_speaker_embedding
            : speaker_encoder_->encode(*request.custom_voice->reference_audio);
    }
    prefill.speaker = request.custom_voice->speaker;
    prefill.language = request.language;
    return prefill;
}

}  // namespace engine::models::qwen3_tts
