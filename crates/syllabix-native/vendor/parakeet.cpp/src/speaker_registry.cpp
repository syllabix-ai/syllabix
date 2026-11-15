#include "speaker_registry.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace pk {

namespace {

constexpr char kMagic[4] = {'P', 'K', 'S', 'R'};
constexpr uint32_t kVersion = 1;    // no fingerprint
constexpr uint32_t kVersion2 = 2;   // with a fingerprint
constexpr uint32_t kMaxFpLen = 4096;
constexpr uint32_t kMaxSpeakers = 1u << 20;
constexpr uint32_t kMaxNameLen = 4096;
constexpr int kMaxDim = 1 << 16;

// L2-normalized copy; empty when the norm is 0 or a value is NaN or Inf.
std::vector<float> normalized(const std::vector<float>& v) {
    double n2 = 0.0;
    for (float x : v) {
        if (!std::isfinite(x)) return {};
        n2 += (double)x * x;
    }
    if (!(n2 > 0.0) || !std::isfinite(n2)) return {};
    const float inv = (float)(1.0 / std::sqrt(n2));
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv;
    return out;
}

void put(std::string& s, const void* p, size_t n) { s.append(static_cast<const char*>(p), n); }

struct Reader {
    const std::string& s;
    size_t pos = 0;
    void get(void* dst, size_t n) {
        if (n > s.size() - pos) throw std::runtime_error("speaker registry: truncated");
        std::memcpy(dst, s.data() + pos, n);
        pos += n;
    }
};

std::string shown(const std::string& s) { return s.empty() ? "unknown" : s; }

}  // namespace

FingerprintVerdict check_fingerprint(const EncoderFingerprint& reg, const EncoderFingerprint& enc,
                                     bool strict) {
    FingerprintVerdict v;
    if (enc.family.empty()) return v;
    if (reg.empty()) {
        v.status = strict ? FingerprintStatus::Required : FingerprintStatus::Unfingerprinted;
        v.message = strict
            ? "speaker registry has no encoder fingerprint and strict mode is on; enroll again "
              "with this encoder (" + enc.family + ") or re-stamp the registry"
            : "speaker registry has no encoder fingerprint; its encoder is unverified, so names "
              "may come from the wrong embedding space (current encoder: " + enc.family + ")";
        return v;
    }
    if (!reg.family.empty() && reg.family != enc.family) {
        v.status = FingerprintStatus::FamilyDiffer;
        v.message = "speaker registry was made with encoder family " + shown(reg.family) +
                    ", but the encoder in use is " + enc.family +
                    "; the embeddings are not comparable, enroll again with this encoder";
        return v;
    }
    if (!reg.weights.empty() && !enc.weights.empty() && reg.weights != enc.weights) {
        v.status = FingerprintStatus::WeightsDiffer;
        v.message = "speaker registry was made with other weights of encoder family " + enc.family +
                    " (" + reg.weights + ", now " + enc.weights +
                    "); a different quantization is expected to work, scores may shift slightly";
    }
    return v;
}

FingerprintVerdict check_registry_for_encoder(const SpeakerRegistry& reg, int enc_dim,
                                              const EncoderFingerprint& enc, bool strict) {
    if (reg.dim() != 0 && reg.dim() != enc_dim) {
        FingerprintVerdict v;
        v.status = FingerprintStatus::DimDiffer;
        v.message = "registry holds " + std::to_string(reg.dim()) +
                    "-value embeddings, this model produces " + std::to_string(enc_dim);
        return v;
    }
    if (reg.size() == 0 && reg.fingerprint().empty()) return {};   // nothing enrolled, nothing to mix up
    return check_fingerprint(reg.fingerprint(), enc, strict);
}

std::vector<std::string> SpeakerRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const Entry& e : entries_) out.push_back(e.name);
    return out;
}

void SpeakerRegistry::enroll(const std::string& name, const std::vector<float>& emb) {
    if (!fp_.empty())
        throw std::invalid_argument(
            "speaker registry is fingerprinted (" + shown(fp_.family) +
            "); the embedding came with no encoder fingerprint, so it is not added");
    enroll_raw(name, emb);
}

void SpeakerRegistry::enroll_raw(const std::string& name, const std::vector<float>& emb) {
    if (name.empty()) throw std::invalid_argument("speaker name is empty");
    if (emb.empty()) throw std::invalid_argument("speaker embedding is empty");
    if (dim_ != 0 && (int)emb.size() != dim_)
        throw std::invalid_argument("speaker embedding has " + std::to_string(emb.size()) +
                                    " values, registry expects " + std::to_string(dim_));
    const std::vector<float> n = normalized(emb);
    if (n.empty()) throw std::invalid_argument("speaker embedding is all zero or not finite");
    if (dim_ == 0) dim_ = (int)emb.size();
    for (Entry& e : entries_) {
        if (e.name != name) continue;
        for (size_t i = 0; i < n.size(); ++i) e.sum[i] += n[i];
        ++e.count;
        return;
    }
    entries_.push_back({name, n, 1});
}

FingerprintVerdict SpeakerRegistry::enroll(const std::string& name, const std::vector<float>& emb,
                                           const EncoderFingerprint& fp) {
    FingerprintVerdict v;
    if (fp.empty()) {
        enroll(name, emb);
        return v;
    }
    if (!fp_.empty()) {
        v = check_fingerprint(fp_, fp, false);
        if (v.is_error()) throw std::invalid_argument(v.message);
    } else if (!entries_.empty()) {
        throw std::invalid_argument(
            "speaker registry has speakers but no encoder fingerprint; they are not stamped "
            "silently. Re-stamp the registry with the encoder that made them, or enroll into a new registry");
    }
    enroll_raw(name, emb);   // throws before anything changes
    if (fp_.empty()) fp_ = fp;
    return v;
}

bool SpeakerRegistry::remove(const std::string& name) {
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].name == name) {
            entries_.erase(entries_.begin() + (long)i);
            return true;
        }
    }
    return false;
}

SpeakerMatch SpeakerRegistry::identify(const std::vector<float>& emb, float accept,
                                       float margin) const {
    if (dim_ != 0 && (int)emb.size() != dim_)
        throw std::invalid_argument("speaker embedding has " + std::to_string(emb.size()) +
                                    " values, registry expects " + std::to_string(dim_));
    SpeakerMatch out;
    const std::vector<float> q = normalized(emb);
    if (q.empty() || entries_.empty()) return out;
    float best = -2.0f, second = -2.0f;
    const Entry* best_e = nullptr;
    for (const Entry& e : entries_) {
        const std::vector<float> c = normalized(e.sum);
        if (c.empty()) continue;   // enrollments that cancel out exactly
        float dot = 0.0f;
        for (size_t i = 0; i < q.size(); ++i) dot += q[i] * c[i];
        if (dot > best) { second = best; best = dot; best_e = &e; }
        else if (dot > second) { second = dot; }
    }
    if (!best_e) return out;
    out.score = best;
    if (best < accept) return out;
    if (entries_.size() >= 2 && second > -2.0f && best - second < margin) return out;
    out.name = best_e->name;
    return out;
}

std::string SpeakerRegistry::serialize() const {
    std::string s;
    put(s, kMagic, 4);
    const uint32_t ver = fp_.empty() ? kVersion : kVersion2;
    put(s, &ver, 4);
    const int32_t dim = dim_;
    put(s, &dim, 4);
    const uint32_t n = (uint32_t)entries_.size();
    put(s, &n, 4);
    if (ver == kVersion2) {
        for (const std::string* f : {&fp_.family, &fp_.weights}) {
            const uint32_t len = (uint32_t)f->size();
            put(s, &len, 4);
            put(s, f->data(), f->size());
        }
    }
    for (const Entry& e : entries_) {
        const uint32_t len = (uint32_t)e.name.size();
        put(s, &len, 4);
        put(s, e.name.data(), e.name.size());
        const int32_t count = e.count;
        put(s, &count, 4);
        put(s, e.sum.data(), e.sum.size() * sizeof(float));
    }
    return s;
}

SpeakerRegistry SpeakerRegistry::deserialize(const std::string& blob) {
    Reader r{blob};
    char magic[4];
    r.get(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("speaker registry: bad magic");
    uint32_t ver = 0;
    r.get(&ver, 4);
    if (ver != kVersion && ver != kVersion2) throw std::runtime_error("speaker registry: unsupported version");
    int32_t dim = 0;
    r.get(&dim, 4);
    uint32_t n = 0;
    r.get(&n, 4);
    if (dim < 0 || dim > kMaxDim || n > kMaxSpeakers || (dim < 1 && n > 0))
        throw std::runtime_error("speaker registry: implausible header");
    SpeakerRegistry out(dim);
    if (ver == kVersion2) {
        for (std::string* f : {&out.fp_.family, &out.fp_.weights}) {
            uint32_t len = 0;
            r.get(&len, 4);
            if (len > kMaxFpLen) throw std::runtime_error("speaker registry: bad fingerprint");
            f->resize(len);
            if (len) r.get(&(*f)[0], len);
        }
        if (out.fp_.empty()) throw std::runtime_error("speaker registry: bad fingerprint");
    }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0;
        r.get(&len, 4);
        if (len == 0 || len > kMaxNameLen) throw std::runtime_error("speaker registry: bad name");
        Entry e;
        e.name.resize(len);
        r.get(&e.name[0], len);
        int32_t count = 0;
        r.get(&count, 4);
        if (count < 1) throw std::runtime_error("speaker registry: bad count");
        e.count = count;
        e.sum.resize((size_t)dim);
        r.get(e.sum.data(), (size_t)dim * sizeof(float));
        out.entries_.push_back(std::move(e));
    }
    if (r.pos != blob.size()) throw std::runtime_error("speaker registry: trailing bytes");
    return out;
}

}  // namespace pk
