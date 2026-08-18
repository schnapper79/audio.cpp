#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/generator.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::fish_audio {

class FishAudioSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IBatchedOfflineVoiceTaskSession {
public:
    FishAudioSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const FishAudioAssets> assets);
    ~FishAudioSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    int64_t max_batch_size() const override;
    std::vector<runtime::BatchedTaskResult> run_batch(
        const std::vector<runtime::TaskRequest> & requests) override;

private:
    struct ReferenceCacheKey {
        std::string source_id;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct ReferenceCacheKeyEqual {
        bool operator()(const ReferenceCacheKey & lhs, const ReferenceCacheKey & rhs) const;
    };

    struct ReferenceCacheEntry {
        FishAudioCodes codes;
    };

    FishAudioRequest make_request(const runtime::TaskRequest & request) const;
    const FishAudioCodes & resolve_reference_codes(const FishAudioReference & reference);
    void forget_context(const std::string & id);
    void remember_context(const std::string & id, FishAudioConversationTurn turn);

    runtime::TaskSpec task_;
    std::shared_ptr<const FishAudioAssets> assets_;
    std::unique_ptr<FishAudioGenerator> generator_;
    std::optional<FishAudioRequest> defaults_;
    runtime::CacheSlots<ReferenceCacheKey, ReferenceCacheEntry, ReferenceCacheKeyEqual> reference_cache_;
    std::optional<ReferenceCacheEntry> uncached_reference_;

    // Der letzte Zug je Kontext, ueber Anfragen hinweg. Im Prompt steht ohnehin
    // schon ein "previous turn"; er lebte bisher nur innerhalb einer Anfrage und
    // war danach weg. Fuer ein Hoerbuch ist das die falsche Grenze: eine Rolle
    // spricht ueber ein Kapitel verteilt in vielen Anfragen, und ohne Gedaechtnis
    // faengt die Sprechweise jedes Mal neu an. Der Schluessel kommt vom Aufrufer
    // (fish_audio.context_id), damit sich Rollen nicht gegenseitig anstecken.
    std::unordered_map<std::string, FishAudioConversationTurn> carried_turns_;
    std::deque<std::string> carried_order_;
    std::size_t carried_capacity_ = 0;

    // Wieviele Sequenzen der AR-Decoder gemeinsam zieht. Vorgabe 1: ohne die
    // Option bleibt alles wie vorher, Stueck fuer Stueck.
    int64_t max_batch_size_ = 1;
    // Wo die Sequenzen im Tensor nebeneinander liegen; siehe FishAudioBatchAxis.
    FishAudioBatchAxis batch_axis_ = FishAudioBatchAxis::Token;
};

}  // namespace engine::models::fish_audio
