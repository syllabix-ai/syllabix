#include "speaker_identifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace pk {

namespace {
constexpr int kSr = 16000;
constexpr double kMinPieceSec = 0.2;   // shorter clean slivers carry no usable voice
}  // namespace

std::string validate_speaker_opts(const SpeakerIdOpts& o) {
    if (!(o.min_voice_sec > 0.0f)) return "min_voice_sec must be > 0";
    if (!(o.refresh_sec > 0.0f)) return "refresh_sec must be > 0";
    if (!(o.max_voice_sec >= o.min_voice_sec)) return "max_voice_sec must be >= min_voice_sec";
    if (!(o.ring_sec >= o.max_voice_sec)) return "ring_sec must be >= max_voice_sec";
    if (!(o.accept_threshold >= -1.0f && o.accept_threshold <= 1.0f))
        return "accept_threshold must be in [-1, 1]";
    if (!(o.margin >= 0.0f)) return "margin must be >= 0";
    return "";
}

std::vector<Interval> clean_intervals(const Interval& seg, const std::vector<Interval>& others,
                                      double min_len) {
    std::vector<Interval> cover;
    for (const Interval& o : others)
        if (o.end > seg.start && o.start < seg.end) cover.push_back(o);
    std::sort(cover.begin(), cover.end(),
              [](const Interval& a, const Interval& b) { return a.start < b.start; });
    std::vector<Interval> out;
    double cursor = seg.start;
    auto emit = [&](double a, double b) {
        if (b - a >= min_len) out.push_back({a, b});
    };
    for (const Interval& c : cover) {
        if (c.start > cursor) emit(cursor, std::min(c.start, seg.end));
        cursor = std::max(cursor, c.end);
        if (cursor >= seg.end) break;
    }
    if (cursor < seg.end) emit(cursor, seg.end);
    return out;
}

SpeakerIdentifier::SpeakerIdentifier(SpeakerEmbed embed, const SpeakerRegistry* registry,
                                     SpeakerIdOpts opts, int profile_dim)
    : profile_dim_(profile_dim), embed_(std::move(embed)), registry_(registry), opts_(opts) {
    const std::string err = validate_speaker_opts(opts_);
    if (!err.empty()) throw std::invalid_argument("invalid speaker options: " + err);
    if (!embed_ || !registry_) throw std::invalid_argument("speaker identifier needs an embedder and a registry");
}

void SpeakerIdentifier::push_pcm(const float* pcm, int n) {
    if (n <= 0 || !pcm) return;
    ring_.insert(ring_.end(), pcm, pcm + n);
    total_ += n;
    const size_t cap = (size_t)((double)opts_.ring_sec * kSr);
    if (ring_.size() > cap + (size_t)kSr) {   // trim in 1 s steps so the erase is amortized
        const size_t drop = ring_.size() - cap;
        ring_.erase(ring_.begin(), ring_.begin() + (long)drop);
        ring_base_ += (long long)drop;
    }
}

void SpeakerIdentifier::add_audio(int slot, const Interval& iv) {
    long long a = std::llround(iv.start * kSr);
    long long b = std::llround(iv.end * kSr);
    a = std::max(a, ring_base_);
    b = std::min(b, total_);
    if (b <= a) return;
    Slot& s = slots_[slot];
    const float* src = ring_.data() + (a - ring_base_);
    s.voice.insert(s.voice.end(), src, src + (b - a));
    s.gained_sec += (double)(b - a) / kSr;
    if (profile_dim_ > 0) s.profile.intervals.push_back({(double)a / kSr, (double)b / kSr});
    const size_t cap = (size_t)((double)opts_.max_voice_sec * kSr);
    if (s.voice.size() > cap) s.voice.erase(s.voice.begin(), s.voice.end() - (long)cap);
    if (profile_dim_ > 0) {
        auto& p = s.profile;
        long long keep = (long long)s.voice.size();
        for (size_t i = p.intervals.size(); i > 0; --i) {
            auto& iv = p.intervals[i - 1];
            const long long n = std::llround((iv.end - iv.start) * kSr);
            if (keep <= n) {
                iv.start = iv.end - (double)keep / kSr;
                p.intervals.erase(p.intervals.begin(), p.intervals.begin() + (long)(i - 1));
                break;
            }
            keep -= n;
        }
        p.clean_duration = (double)s.voice.size() / kSr;
    }
}

void SpeakerIdentifier::apply(Slot& s, const SpeakerMatch& m) {
    if (m.name.empty()) {            // unknown: keep the current name, break any pending run
        s.pending.clear();
        return;
    }
    if (s.current.name.empty() || m.name == s.current.name) {
        s.current = {m.name, m.score};
        s.pending.clear();
        return;
    }
    if (m.name == s.pending) {       // a different name won twice in a row
        s.current = {m.name, m.score};
        s.pending.clear();
    } else {
        s.pending = m.name;
    }
}

void SpeakerIdentifier::maybe_embed(Slot& s, bool is_last) {
    if ((double)s.voice.size() / kSr < (double)opts_.min_voice_sec) return;
    if (s.gained_sec <= 0.0) return;
    const bool first = !s.embedded;
    const bool due = first || s.gained_sec >= (double)opts_.refresh_sec || is_last;
    if (!due) return;
    std::vector<float> emb;
    if (!embed_(s.voice.data(), (int)s.voice.size(), emb)) {
        if (profile_dim_ == 0) throw std::runtime_error("speaker embedding failed");
        s.profile.unavailable_reason = "embedding_failed";
        return;
    }
    if (profile_dim_ > 0) {
        double norm = 0;
        for (float v : emb) norm += (double)v * v;
        if ((int)emb.size() != profile_dim_ || !std::isfinite(norm) || norm <= 0) {
            s.profile.unavailable_reason = "invalid_embedding";
            return;
        }
        for (float& v : emb) v = (float)(v / std::sqrt(norm));
        s.profile.embedding = emb;
        s.profile.unavailable_reason.clear();
    }
    s.embedded = true;
    s.gained_sec = 0.0;
    apply(s, registry_->identify(emb, opts_.accept_threshold, opts_.margin));
}

void SpeakerIdentifier::consume(const SpeakerSegment& seg, const std::vector<SpeakerSegment>& open,
                                bool growing) {
    Slot& s = slots_[seg.speaker];   // a slot is known as soon as it has any segment
    const double from = std::max((double)seg.start, s.consumed_until);
    const double to = seg.end;
    if (to <= from) return;
    std::vector<Interval> others;
    for (const SpeakerSegment& h : history_)
        if (h.speaker != seg.speaker) others.push_back({h.start, h.end});
    for (const SpeakerSegment& o : open)
        if (o.speaker != seg.speaker) others.push_back({o.start, o.end});
    double cursor = to;
    // The tail of a growing segment is taken with min_len 0 so a short clean
    // piece at the end can be held back and joined to the audio that follows.
    std::vector<Interval> pieces = clean_intervals({from, to}, others, growing ? 0.0 : kMinPieceSec);
    if (growing && !pieces.empty() && pieces.back().end >= to &&
        pieces.back().end - pieces.back().start < kMinPieceSec) {
        cursor = pieces.back().start;
        pieces.pop_back();
    }
    for (const Interval& iv : pieces)
        if (iv.end - iv.start >= kMinPieceSec) add_audio(seg.speaker, iv);
    s.consumed_until = std::max(s.consumed_until, cursor);
}

void SpeakerIdentifier::update(const std::vector<SpeakerSegment>& closed,
                               const std::vector<SpeakerSegment>& open, bool is_last) {
    for (const SpeakerSegment& c : closed) history_.push_back(c);
    for (const SpeakerSegment& c : closed) consume(c, open, false);
    for (const SpeakerSegment& o : open) consume(o, open, true);
    const double horizon = (double)total_ / kSr - (double)opts_.ring_sec;
    history_.erase(std::remove_if(history_.begin(), history_.end(),
                                  [&](const SpeakerSegment& h) { return h.end < horizon; }),
                   history_.end());
    for (auto& kv : slots_) maybe_embed(kv.second, is_last);
}

SlotName SpeakerIdentifier::name(int slot) const {
    auto it = slots_.find(slot);
    return it == slots_.end() ? SlotName{} : it->second.current;
}

std::map<int, SlotName> SpeakerIdentifier::names() const {
    std::map<int, SlotName> out;
    for (const auto& kv : slots_) out[kv.first] = kv.second.current;
    return out;
}

std::map<int, SpeakerProfile> SpeakerIdentifier::profiles() const {
    std::map<int, SpeakerProfile> out;
    for (const auto& kv : slots_) out[kv.first] = kv.second.profile;
    return out;
}

std::map<int, SlotName> identify_offline(const std::vector<float>& pcm16k,
                                         const std::vector<SpeakerSegment>& segs,
                                         const SpeakerEmbed& embed, const SpeakerRegistry& reg,
                                         const SpeakerIdOpts& opts) {
    return identify_offline(pcm16k, segs, embed, reg, opts, nullptr, 0);
}

std::map<int, SlotName> identify_offline(const std::vector<float>& pcm16k,
    const std::vector<SpeakerSegment>& segs, const SpeakerEmbed& embed,
    const SpeakerRegistry& reg, const SpeakerIdOpts& opts,
    std::map<int, SpeakerProfile>* profiles, int expected_dim) {
    if (profiles && (expected_dim <= 0 || (reg.dim() != 0 && reg.dim() != expected_dim)))
        throw std::invalid_argument("profile encoder dimension mismatch");
    SpeakerIdOpts o = opts;
    o.ring_sec = std::max(o.ring_sec, (float)pcm16k.size() / (float)kSr + 1.0f);   // keep the whole recording
    o.max_voice_sec = std::max(o.max_voice_sec, 30.0f);                            // a whole recording can afford more voice
    o.ring_sec = std::max(o.ring_sec, o.max_voice_sec);
    SpeakerIdentifier id(embed, &reg, o, profiles ? expected_dim : 0);
    id.push_pcm(pcm16k.data(), (int)pcm16k.size());
    auto ordered = segs;
    if (profiles) {
        for (auto& seg : ordered) {
            if (!std::isfinite(seg.start) || !std::isfinite(seg.end))
                throw std::invalid_argument("nonfinite speaker interval");
            seg.start = std::max(0.0f, seg.start);
            seg.end = std::min(seg.end, (float)((double)pcm16k.size() / kSr));
        }
        std::stable_sort(ordered.begin(), ordered.end(), [](const SpeakerSegment& a, const SpeakerSegment& b) {
            return a.start < b.start;
        });
    }
    id.update(ordered, {}, true);
    if (profiles) *profiles = id.profiles();
    return id.names();
}

}  // namespace pk
