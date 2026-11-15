#pragma once
// Standalone voice-activity detection: speech segments (and optionally the
// per-frame probabilities) from the model's own VAD head, as one JSON document.
// Shared by the C-API (parakeet_capi_vad_*) and `parakeet-cli vad`.
#include <string>
#include <vector>

#include "transcription.hpp"
#include "vad_segmenter.hpp"

namespace pk {

class Model;
class SileroVad;

struct VadRequest {
    enum class Mode { kSpeech, kSegments };
    SegmenterOpts opts;               // threshold, min_pause, min_speech, max_segment
    Mode mode = Mode::kSpeech;
    bool probabilities = false;       // add "probabilities" to the document
    VadKind kind = VadKind::kHead;    // which model the defaults in `opts` are for
    WordFilter filter;                // word filter keys, only with allow_filter
};

// Parses an options document: a flat JSON object, or NULL / "" for the
// defaults. Keys: "threshold" (0 < x <= 1), "min_pause", "min_speech",
// "max_segment" (seconds, > 0), "trim" (seconds >= 0, "segments" mode and the
// transcribe functions; 0 = no trimming), "run_gate" (0 <= x < 1, default 0 =
// off; a speech run whose median probability is below it is dropped, in both
// modes and in the transcribe functions; not for streams), "mode" ("speech" or "segments"),
// "probabilities" (bool), "speech_pad" (seconds >= 0, "speech" mode). With
// `allow_filter` the word filter keys of parse_filter_options are accepted too
// and stored in `req.filter`. Unknown keys and bad values are errors. The
// values that a key leaves out come from default_segmenter_opts(kind). Returns
// false and sets `err` on failure.
bool parse_vad_options(const char* json, VadRequest& req, std::string& err,
                       VadKind kind = VadKind::kHead, bool allow_filter = false);

// The word filter options alone: a flat JSON object, or NULL / "" for a filter
// that is off. Keys: "min_local_conf" (0 <= x <= 1, 0 = off), "local_radius"
// (seconds > 0, default 5), "drop_punct_only" (bool). Unknown keys and bad
// values are errors. Returns false and sets `err` on failure.
bool parse_filter_options(const char* json, WordFilter& filter, std::string& err);

// Same document for a Silero VAD model. `pcm` is mono at `sample_rate`, which
// must be 16000 or 8000 (the caller resamples anything else); times are on the
// input timeline and frame_sec is 0.032 at both rates.
std::string silero_vad_to_json(const SileroVad& m, const std::vector<float>& pcm, int sample_rate,
                               const VadRequest& req);

// True when the GGUF's general.architecture is "silero_vad". Reads only the header.
bool gguf_is_silero(const std::string& gguf_path);

// True when the GGUF's parakeet.arch is "vad": a VAD-only slice of an Ultra or
// Redux model (scripts/slice_vad_gguf.py). Reads only the header.
bool gguf_is_vad_only(const std::string& gguf_path);

// Runs the VAD head over 16 kHz mono PCM and returns the JSON document
// {"mode","duration","frame_sec","backend","segments":[{"start","end"}],
//  "probabilities":[...]?}. Throws std::runtime_error("model has no VAD head")
// for a model without the head.
std::string vad_to_json(const Model& m, const std::vector<float>& pcm16k, const VadRequest& req);

}  // namespace pk
