#pragma once
#include "speaker_identifier.hpp"   // pk::SpeakerEmbed

#include <memory>
#include <string>
#include <vector>

namespace pk {

// A loaded voice-detect.cpp speaker encoder (WeSpeaker, CAM++, ECAPA or ERes2Net).
// The only parakeet code that talks to voice-detect.cpp, and only through
// voicedetect_capi.h. Not thread-safe: one stream at a time per encoder, like
// the other contexts.
class SpeakerEncoder {
public:
    // False when parakeet was built with PARAKEET_WITH_VOICEDETECT=OFF.
    static bool available();
    // nullptr on failure, when unavailable, or when the GGUF has no speaker
    // embedding (for example an age/gender/emotion model). With a non-empty
    // `component`, `gguf_path` is a bundle GGUF (docs/bundle.md) and the encoder is
    // its component of kind "voice". The bundle is mapped read-only and
    // voice-detect.cpp copies the component's tensors out of the map during the call:
    // no temporary file, and the other components are not read (see bundle_map.hpp).
    // `err`, when given, receives the reason for a failure.
    static std::unique_ptr<SpeakerEncoder> load(const std::string& gguf_path, const std::string& component = "",
                                                std::string* err = nullptr);
    ~SpeakerEncoder();
    SpeakerEncoder(const SpeakerEncoder&) = delete;
    SpeakerEncoder& operator=(const SpeakerEncoder&) = delete;

    int dim() const { return dim_; }
    // Which encoder this is: family from the GGUF metadata, weights as the
    // sha256 of the GGUF file bytes, or, for a bundle component, the sha256 of the
    // single-model file recorded in the bundle header (source_sha256). See
    // speaker_encoder_family.
    const EncoderFingerprint& fingerprint() const { return fp_; }
    // L2-normalized embedding of 16 kHz mono PCM. False on failure (see last_error).
    bool embed(const float* pcm, int n, std::vector<float>& emb);
    // A SpeakerEmbed bound to this encoder; valid while the encoder lives.
    SpeakerEmbed embedder();
    const std::string& last_error() const { return last_error_; }

private:
    SpeakerEncoder() = default;
    static std::unique_ptr<SpeakerEncoder> adopt(void* ctx, const std::string& path, const std::string& prefix,
                                                 const std::string& weights, std::string* err);
    void* ctx_ = nullptr;   // voicedetect_ctx*
    int dim_ = 0;
    EncoderFingerprint fp_;
    std::string last_error_;
};

// Encoder family id of a voice-detect GGUF, read from its header only:
//   "voicedetect:<voicedetect.arch>:<general.name>:<voicedetect.embedding_dim>"
// A missing key leaves its field empty ("voicedetect::name:256"). "" when the file
// is not a readable voice-detect GGUF. `dim_fallback` is used when the GGUF has no
// embedding_dim key (0 leaves the field empty). With a non-empty `prefix` (for a bundle
// component: "<component>.") every key is looked up under that prefix.
std::string speaker_encoder_family(const std::string& gguf_path, int dim_fallback = 0,
                                   const std::string& prefix = "");

// True when the GGUF's general.architecture is "voicedetect". Reads only the header.
bool gguf_is_voicedetect(const std::string& gguf_path);

}  // namespace pk
