#include "vad_json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backend.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"
#include "silero_vad.hpp"

#include "gguf.h"

namespace pk {

namespace {

struct Cursor {
    const char* p;
    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }
};

bool parse_string(Cursor& c, std::string& out) {
    if (*c.p != '"') return false;
    ++c.p;
    out.clear();
    while (*c.p && *c.p != '"') {
        if (*c.p == '\\') return false;  // no escapes are needed for any key or value
        out += *c.p++;
    }
    if (*c.p != '"') return false;
    ++c.p;
    return true;
}

bool parse_number(Cursor& c, double& v) {
    char* end = nullptr;
    v = std::strtod(c.p, &end);
    if (end == c.p || !std::isfinite(v)) return false;
    c.p = end;
    return true;
}

bool positive_seconds(double v) { return std::isfinite(v) && v > 0.0 && v <= 1e6; }

}  // namespace

namespace {

// Shared parser of the flat option objects. `vad_keys` accepts the segmenter
// and mode keys, `filter_keys` the word filter keys ("trim" belongs to the
// segmenter). `what` names the document in error messages.
bool parse_options(const char* json, VadRequest& req, std::string& err, bool vad_keys,
                   bool filter_keys, const char* what) {
    const std::string docs = std::string(what) + "s";  // "VAD options" / "options"
    if (!json) return true;
    Cursor c{json};
    c.ws();
    if (*c.p == '\0') return true;
    if (*c.p != '{') { err = "invalid " + docs + ": expected a JSON object"; return false; }
    ++c.p;
    c.ws();
    if (*c.p == '}') { ++c.p; c.ws(); if (*c.p) { err = "invalid " + docs + ": trailing text"; return false; } return true; }
    for (;;) {
        c.ws();
        std::string key;
        if (!parse_string(c, key)) { err = "invalid " + docs + ": expected a key"; return false; }
        c.ws();
        if (*c.p != ':') { err = "invalid " + docs + ": expected ':' after \"" + key + "\""; return false; }
        ++c.p;
        c.ws();
        const std::string opt = std::string(what) + " " + key;
        const bool is_seg_num = key == "threshold" || key == "min_pause" || key == "min_speech" ||
                                key == "max_segment" || key == "speech_pad" || key == "trim" || key == "run_gate";
        if (vad_keys && key == "mode") {
            std::string v;
            if (!parse_string(c, v)) { err = "invalid " + opt + ": expected a string"; return false; }
            if (v == "speech") req.mode = VadRequest::Mode::kSpeech;
            else if (v == "segments") req.mode = VadRequest::Mode::kSegments;
            else { err = "invalid " + opt + ": use \"speech\" or \"segments\""; return false; }
        } else if ((vad_keys && key == "probabilities") || (filter_keys && key == "drop_punct_only")) {
            bool v = false;
            if (std::strncmp(c.p, "true", 4) == 0) { v = true; c.p += 4; }
            else if (std::strncmp(c.p, "false", 5) == 0) { v = false; c.p += 5; }
            else { err = "invalid " + opt + ": expected true or false"; return false; }
            if (key == "probabilities") req.probabilities = v;
            else req.filter.drop_punct_only = v;
        } else if ((vad_keys && is_seg_num) || (filter_keys && (key == "min_local_conf" || key == "local_radius"))) {
            double v = 0.0;
            if (!parse_number(c, v)) { err = "invalid " + opt + ": expected a number"; return false; }
            if (key == "threshold") {
                if (!(v > 0.0 && v <= 1.0)) { err = "invalid " + opt + ": must be in (0, 1]"; return false; }
                req.opts.threshold = (float)v;
            } else if (key == "run_gate") {
                if (!(v >= 0.0 && v < 1.0)) { err = "invalid " + opt + ": must be in [0, 1) (0 = off)"; return false; }
                req.opts.run_gate = (float)v;
            } else if (key == "min_local_conf") {
                if (!(v >= 0.0 && v <= 1.0)) { err = "invalid " + opt + ": must be in [0, 1] (0 = off)"; return false; }
                req.filter.min_local_conf = (float)v;
            } else if (key == "local_radius") {
                if (!positive_seconds(v)) { err = "invalid " + opt + ": must be a number of seconds > 0"; return false; }
                req.filter.local_radius_sec = (float)v;
            } else if (key == "speech_pad" || key == "trim") {
                if (!(std::isfinite(v) && v >= 0.0 && v <= 1e6)) { err = "invalid " + opt + ": must be a number of seconds >= 0"; return false; }
                (key == "trim" ? req.opts.trim_sec : req.opts.pad_sec) = v;
            } else {
                if (!positive_seconds(v)) { err = "invalid " + opt + ": must be a number of seconds > 0"; return false; }
                if (key == "min_pause") req.opts.min_pause_sec = v;
                else if (key == "min_speech") req.opts.min_speech_sec = v;
                else req.opts.max_seg_sec = v;
            }
        } else {
            err = std::string("unknown ") + what + ": " + key;
            return false;
        }
        c.ws();
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == '}') { ++c.p; break; }
        err = "invalid " + docs + ": expected ',' or '}'";
        return false;
    }
    c.ws();
    if (*c.p) { err = "invalid " + docs + ": trailing text"; return false; }
    return true;
}

}  // namespace

bool parse_vad_options(const char* json, VadRequest& req, std::string& err, VadKind kind,
                       bool allow_filter) {
    req = VadRequest();
    req.kind = kind;
    req.opts = default_segmenter_opts(kind);
    return parse_options(json, req, err, true, allow_filter, "VAD option");
}

bool parse_filter_options(const char* json, WordFilter& filter, std::string& err) {
    VadRequest req;
    if (!parse_options(json, req, err, false, true, "option")) return false;
    filter = req.filter;
    return true;
}

namespace {

std::string format_vad_json(const VadRequest& req, double total, double frame_sec,
                            const std::vector<float>& p, const std::vector<VadSegment>& segs) {
    char buf[64];
    std::string j = "{\"mode\":\"";
    j += req.mode == VadRequest::Mode::kSpeech ? "speech" : "segments";
    std::snprintf(buf, sizeof(buf), "\",\"duration\":%.3f,\"frame_sec\":%.3f,\"backend\":\"",
                  total, frame_sec);
    j += buf;
    j += pk::global_backend().device_name();
    j += "\",\"segments\":[";
    for (size_t i = 0; i < segs.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%s{\"start\":%.3f,\"end\":%.3f}", i ? "," : "",
                      segs[i].start, segs[i].end);
        j += buf;
    }
    j += ']';
    if (req.probabilities) {
        j += ",\"probabilities\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%s%.4f", i ? "," : "", (double)p[i]);
            j += buf;
        }
        j += ']';
    }
    j += '}';
    return j;
}

}  // namespace

std::string vad_to_json(const Model& m, const std::vector<float>& pcm16k, const VadRequest& req) {
    const ParakeetConfig& cfg = m.config();
    if (!cfg.vad.present) throw std::runtime_error("model has no VAD head");
    SegmenterOpts o = req.opts;
    o.frame_sec = cfg.vad.frame_sec;
    if (req.mode == VadRequest::Mode::kSegments && !(o.max_seg_sec > 2.0 * o.frame_sec))
        throw std::runtime_error("invalid VAD option max_segment: too small");
    const double total = (double)pcm16k.size() / 16000.0;
    std::vector<float> p;
    if (!pcm16k.empty()) p = m.vad_probabilities(pcm16k);
    std::vector<VadSegment> segs;
    if (!pcm16k.empty())
        segs = req.mode == VadRequest::Mode::kSpeech ? speech_regions(p, total, o)
                                                      : segment_by_vad(p, total, o);
    return format_vad_json(req, total, (double)o.frame_sec, p, segs);
}

std::string silero_vad_to_json(const SileroVad& m, const std::vector<float>& pcm, int sample_rate,
                               const VadRequest& req) {
    if (!m.supports(sample_rate))
        throw std::runtime_error("Silero VAD needs 16000 or 8000 Hz audio");
    SegmenterOpts o = req.opts;
    o.frame_sec = kSileroFrameSec;
    if (req.mode == VadRequest::Mode::kSegments && !(o.max_seg_sec > 2.0 * o.frame_sec))
        throw std::runtime_error("invalid VAD option max_segment: too small");
    const double total = (double)pcm.size() / (double)sample_rate;
    std::vector<float> p;
    if (!pcm.empty()) {
        p = m.probabilities(pcm.data(), pcm.size(), sample_rate);
        if (p.empty()) throw std::runtime_error("Silero VAD failed");
    }
    std::vector<VadSegment> segs;
    if (!pcm.empty())
        segs = req.mode == VadRequest::Mode::kSpeech ? speech_regions(p, total, o)
                                                      : segment_by_vad(p, total, o);
    return format_vad_json(req, total, o.frame_sec, p, segs);
}

bool gguf_is_silero(const std::string& path) {
    gguf_init_params ip{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "general.architecture");
    const bool ok = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(g, id)) == "silero_vad";
    gguf_free(g);
    return ok;
}

bool gguf_is_vad_only(const std::string& path) {
    gguf_init_params ip{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "parakeet.arch");
    const bool ok = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(g, id)) == "vad";
    gguf_free(g);
    return ok;
}

}  // namespace pk
