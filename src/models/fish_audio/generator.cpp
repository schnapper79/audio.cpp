#include "engine/models/fish_audio/generator.h"

#include "engine/framework/debug/profiler.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

FishAudioGenerator::FishAudioGenerator(
    std::shared_ptr<const FishAudioAssets> assets,
    std::unique_ptr<FishAudioARRuntime> ar,
    std::unique_ptr<FishAudioCodecRuntime> codec)
    : assets_(std::move(assets)),
      tokenizer_(assets_),
      prompt_builder_(assets_, tokenizer_),
      ar_(std::move(ar)),
      codec_(std::move(codec)) {
    if (assets_ == nullptr || ar_ == nullptr || codec_ == nullptr) {
        throw std::runtime_error("Fish Audio generator requires assets, AR runtime, and codec runtime");
    }
}

FishAudioGenerator::~FishAudioGenerator() = default;

FishAudioCodes FishAudioGenerator::encode_reference(const runtime::AudioBuffer & audio) {
    auto codes = codec_->encode_reference(audio);
    codec_->release_encode_graph();
    return codes;
}

FishAudioGenerationResult FishAudioGenerator::generate(
    const FishAudioRequest & request,
    const std::optional<FishAudioCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver) {
    engine::debug::trace_log_scalar("fish_audio.request.has_reference", request.reference.has_value());
    engine::debug::trace_log_scalar("fish_audio.request.text_chars", static_cast<int64_t>(request.text.size()));
    engine::debug::trace_log_scalar("fish_audio.request.has_previous_turn", previous_turn.has_value());
    engine::debug::trace_log_scalar("fish_audio.sampler.seed", request.generation.seed);
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(request, reference_codes, previous_turn);
    engine::debug::timing_log_scalar(
        "fish_audio.prompt_build_ms",
        engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    FishAudioGenerationResult result;
    result.codes = ar_->generate(prompt, request.generation);
    engine::debug::trace_log_scalar("fish_audio.generated.frames", result.codes.frames);
    engine::debug::trace_log_scalar("fish_audio.generated.codebooks", result.codes.codebooks);
    engine::debug::timing_log_scalar(
        "fish_audio.ar_generate_ms",
        engine::debug::elapsed_ms(ar_start, Clock::now()));

    const auto decode_start = Clock::now();
    result.audio = codec_->decode(result.codes);
    engine::debug::timing_log_scalar(
        "fish_audio.codec_decode_ms",
        engine::debug::elapsed_ms(decode_start, Clock::now()));
    codec_->release_runtime_graphs();
    if (mem_saver) {
        ar_->release_runtime_graphs();
    }
    return result;
}

std::vector<FishAudioGenerationResult> FishAudioGenerator::generate_batch(
    const std::vector<FishAudioRequest> & requests,
    const std::vector<std::optional<FishAudioCodes>> & reference_codes,
    FishAudioBatchAxis axis,
    bool mem_saver) {
    if (requests.empty() || requests.size() != reference_codes.size()) {
        throw std::runtime_error("Fish Audio batch generation requires one reference per request");
    }
    engine::debug::trace_log_scalar("fish_audio.batch.items", static_cast<int64_t>(requests.size()));
    const auto prompt_start = Clock::now();
    std::vector<FishAudioPrompt> prompts;
    std::vector<FishAudioGenerationOptions> options;
    prompts.reserve(requests.size());
    options.reserve(requests.size());
    for (size_t index = 0; index < requests.size(); ++index) {
        prompts.push_back(prompt_builder_.build(requests[index], reference_codes[index], std::nullopt));
        options.push_back(requests[index].generation);
    }
    engine::debug::timing_log_scalar(
        "fish_audio.batch.prompt_build_ms", engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    engine::debug::trace_log_scalar(
        "fish_audio.batch.axis", axis == FishAudioBatchAxis::Token ? "token" : "channel");
    auto codes = ar_->generate_batch(prompts, options, axis);
    engine::debug::timing_log_scalar(
        "fish_audio.batch.ar_generate_ms", engine::debug::elapsed_ms(ar_start, Clock::now()));

    const auto decode_start = Clock::now();
    std::vector<FishAudioGenerationResult> results(requests.size());
    for (size_t index = 0; index < requests.size(); ++index) {
        results[index].codes = std::move(codes[index]);
        results[index].audio = codec_->decode(results[index].codes);
        // Genau wie im Einzelpfad je Stueck freigeben. Laesst man den Graphen
        // ueber mehrere Stuecke stehen, rechnet der Codec minimal anders --
        // gemessen -66 dB, unhoerbar, aber die Datei ist dann nicht mehr
        // byteweise dieselbe wie beim Einzellauf. Der Neubau kostet nichts,
        // was der Einzelpfad nicht ohnehin zahlt.
        codec_->release_runtime_graphs();
    }
    engine::debug::timing_log_scalar(
        "fish_audio.batch.codec_decode_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));
    if (mem_saver) {
        ar_->release_runtime_graphs();
    }
    return results;
}

}  // namespace engine::models::fish_audio
