#pragma once
#include <string>
#include <vector>

namespace pk {

// Result of matching one embedding against the registry. An empty `name` means
// unknown; `score` is then the best cosine seen (0 for an empty registry).
struct SpeakerMatch {
    std::string name;
    float score = 0.0f;
};

// Which speaker encoder made a set of embeddings. Equal embedding sizes do not
// mean the same embedding space, so a registry remembers its encoder.
//   family:  names the embedding space. Built from the encoder GGUF metadata:
//            "voicedetect:<voicedetect.arch>:<general.name>:<embedding_dim>".
//            A mismatch is a hard error.
//   weights: names the exact weights: "sha256:<64 hex>" of the encoder GGUF
//            file bytes. Another quantization of the same family differs here.
//            A mismatch is a warning.
// Either field may be empty (unknown). Both empty means "no fingerprint".
struct EncoderFingerprint {
    std::string family;
    std::string weights;
    bool empty() const { return family.empty() && weights.empty(); }
};

enum class FingerprintStatus {
    Match,             // same family and same (or unrecorded) weights
    Unfingerprinted,   // the registry has no fingerprint: accepted with a warning
    WeightsDiffer,     // same family, other weights: accepted with a warning
    FamilyDiffer,      // other family: refused
    Required,          // no fingerprint and strict mode: refused
    DimDiffer,         // other embedding size: refused (checked before the fingerprint)
};

struct FingerprintVerdict {
    FingerprintStatus status = FingerprintStatus::Match;
    std::string message;   // "" for Match; the one text used by every entry point
    bool is_error() const {
        return status == FingerprintStatus::FamilyDiffer || status == FingerprintStatus::Required ||
               status == FingerprintStatus::DimDiffer;
    }
    bool is_warning() const {
        return status == FingerprintStatus::Unfingerprinted || status == FingerprintStatus::WeightsDiffer;
    }
};

// Compares the fingerprint stored in a registry with the encoder in use.
// `strict` turns "registry has no fingerprint" into an error. An encoder with
// no family is not checked.
FingerprintVerdict check_fingerprint(const EncoderFingerprint& registry,
                                     const EncoderFingerprint& encoder, bool strict);

class SpeakerRegistry;

// The check every entry point runs where an encoder meets a registry, before
// any name is assigned: embedding size first, then check_fingerprint. One text
// for the CLI and the C-API. `enc_dim` is the loaded encoder's embedding size.
FingerprintVerdict check_registry_for_encoder(const SpeakerRegistry& reg, int enc_dim,
                                              const EncoderFingerprint& enc, bool strict);

// Enrolled speakers. Each speaker is the L2-normalized mean of the L2-normalized
// embeddings enrolled under its name (a centroid), so enrolling more clips
// tightens the match. Model-independent: it never sees audio. Not thread-safe.
class SpeakerRegistry {
public:
    // dim 0 means "fixed by the first enrollment".
    explicit SpeakerRegistry(int dim = 0) : dim_(dim) {}

    int dim() const { return dim_; }
    size_t size() const { return entries_.size(); }
    std::vector<std::string> names() const;   // enrollment order

    // Throws std::invalid_argument: empty name, empty or all-zero embedding,
    // a size different from dim() when dim() != 0, or a fingerprinted registry
    // (use the overload with a fingerprint).
    void enroll(const std::string& name, const std::vector<float>& emb);
    // Same, and records which encoder made `emb`. With a non-empty `fp`:
    //  - an empty registry takes `fp` as its fingerprint;
    //  - a registry with another family throws std::invalid_argument (same text
    //    as check_fingerprint);
    //  - a registry with the same family and other weights keeps its own
    //    weights hash and the returned status is WeightsDiffer;
    //  - a non-empty registry with no fingerprint throws: it is never stamped
    //    silently (use set_fingerprint, for example `parakeet-cli registry
    //    --restamp`).
    // With an empty `fp` and a fingerprinted registry it throws too, because the
    // new vector would be unverifiable. Nothing changes when it throws.
    FingerprintVerdict enroll(const std::string& name, const std::vector<float>& emb,
                              const EncoderFingerprint& fp);
    const EncoderFingerprint& fingerprint() const { return fp_; }
    // Replaces the fingerprint with no checks. For an explicit re-stamp only.
    void set_fingerprint(const EncoderFingerprint& fp) { fp_ = fp; }
    bool remove(const std::string& name);

    // Best speaker by cosine. Known only if the best score >= accept and, when
    // two or more speakers are enrolled, it beats the runner-up by >= margin.
    // Throws std::invalid_argument on a size different from dim() (when dim()
    // is set). An all-zero embedding is unknown, not an error.
    SpeakerMatch identify(const std::vector<float>& emb, float accept, float margin) const;

    // Binary blob, little-endian. Version 1 (no fingerprint, written when the
    // registry has none, so old readers keep working):
    //   "PKSR", u32 1, i32 dim, u32 n, then per speaker
    //   u32 name length, name bytes, i32 count, dim x f32 sum.
    // Version 2 (written when there is a fingerprint) puts the fingerprint
    // after the header and before the speakers:
    //   "PKSR", u32 2, i32 dim, u32 n,
    //   u32 family length, family bytes, u32 weights length, weights bytes,
    //   then the same per-speaker records.
    std::string serialize() const;
    // Throws std::runtime_error on bad magic or version, truncation, an absurd
    // count, or trailing bytes.
    static SpeakerRegistry deserialize(const std::string& blob);

private:
    void enroll_raw(const std::string& name, const std::vector<float>& emb);
    struct Entry {
        std::string name;
        std::vector<float> sum;   // sum of L2-normalized enrollments
        int count = 0;
    };
    int dim_;
    EncoderFingerprint fp_;
    std::vector<Entry> entries_;
};

}  // namespace pk
