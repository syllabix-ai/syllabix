#include "vad_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace pk {

namespace {
// Above this many seconds a frame count could overflow int64.
constexpr double kMaxSec = 1e6;

// Run gate: every run of sp == 1 (a maximal run of frames with p >= threshold)
// whose median probability is below `gate` becomes silence. The median is over
// p[a, b) of the run; for an even count it is the mean of the two middle values,
// computed in float (as numpy does for float32). A run is kept when median >= gate. No-op for gate <= 0.
void apply_run_gate(std::vector<char>& sp, const std::vector<float>& p, int64_t n, float gate) {
    if (!(gate > 0.0f)) return;
    std::vector<float> v;
    int64_t f = 0;
    while (f < n) {
        if (!sp[(size_t)f]) { ++f; continue; }
        int64_t e = f;
        while (e < n && sp[(size_t)e]) ++e;
        v.assign(p.begin() + f, p.begin() + e);  // sp is only set where f < p.size()
        const size_t m = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + m, v.end());
        float med = v[m];
        if (v.size() % 2 == 0) med = (*std::max_element(v.begin(), v.begin() + m) + med) * 0.5f;
        if (med < gate) std::fill(sp.begin() + f, sp.begin() + e, 0);
        f = e;
    }
}
}

std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o) {
    std::vector<VadSegment> out;
    // Degenerate options: no usable frame grid, so do not cut at all.
    if (!(o.frame_sec > 0.0) || !std::isfinite(o.frame_sec) || !std::isfinite(o.max_seg_sec) ||
        !(o.max_seg_sec > 2.0 * o.frame_sec) || !std::isfinite(total_sec) ||
        !std::isfinite(o.threshold) || !std::isfinite(o.min_seg_sec) ||
        !std::isfinite(o.min_pause_sec) || !std::isfinite(o.bridge_sec) ||
        !std::isfinite(o.min_speech_sec) || !std::isfinite(o.trim_sec) || !std::isfinite(o.run_gate) || o.max_seg_sec > kMaxSec || o.min_seg_sec > kMaxSec ||
        o.min_pause_sec > kMaxSec || o.bridge_sec > kMaxSec || o.min_speech_sec > kMaxSec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    if (total_sec <= o.max_seg_sec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    const double fs = o.frame_sec;
    const int64_t n = std::max<int64_t>(0, (int64_t)std::ceil(total_sec / fs - 1e-9));
    const int64_t max_f = std::max<int64_t>(2, (int64_t)std::floor(o.max_seg_sec / fs + 1e-9));
    const int64_t min_f = std::min<int64_t>(max_f - 1, std::max<int64_t>(1, (int64_t)std::ceil(o.min_seg_sec / fs - 1e-9)));
    const int64_t pause_f = std::max<int64_t>(1, (int64_t)std::ceil(o.min_pause_sec / fs - 1e-9));

    // Step 1: speech mask over n frames (frames without a probability are silent).
    std::vector<char> sp((size_t)n, 0);
    for (int64_t f = 0; f < n && f < (int64_t)p.size(); ++f) sp[(size_t)f] = p[(size_t)f] >= o.threshold;
    apply_run_gate(sp, p, n, o.run_gate);
    // Runs of equal value as [begin, end) frame ranges.
    auto for_runs = [&](bool value, auto&& fn) {
        int64_t f = 0;
        while (f < n) {
            if ((sp[(size_t)f] != 0) != value) { ++f; continue; }
            int64_t e = f;
            while (e < n && (sp[(size_t)e] != 0) == value) ++e;
            fn(f, e);
            f = e;
        }
    };
    // Bridge speech gaps shorter than bridge_sec (only gaps between two speech runs).
    for_runs(false, [&](int64_t a, int64_t b) {
        if (a > 0 && b < n && (double)(b - a) * fs + 1e-9 < o.bridge_sec)
            std::fill(sp.begin() + a, sp.begin() + b, 1);
    });
    // Drop speech runs shorter than min_speech_sec.
    for_runs(true, [&](int64_t a, int64_t b) {
        if ((double)(b - a) * fs + 1e-9 < o.min_speech_sec) std::fill(sp.begin() + a, sp.begin() + b, 0);
    });
    // Prefix sums of speech frames to test a range for speech in O(1).
    std::vector<int64_t> cum((size_t)n + 1, 0);
    for (int64_t f = 0; f < n; ++f) cum[(size_t)f + 1] = cum[(size_t)f] + sp[(size_t)f];
    auto has_speech = [&](int64_t a, int64_t b) { return b > a && cum[(size_t)b] - cum[(size_t)a] > 0; };
    // Pauses: silent runs of at least pause_f frames, in order.
    std::vector<std::pair<int64_t, int64_t>> pauses;
    for_runs(false, [&](int64_t a, int64_t b) { if (b - a >= pause_f) pauses.emplace_back(a, b); });

    // Step 3: cut from the front. A segment starting at s may end at a pause.
    // With trim_sec > 0 the segment shrinks to its speech plus trim_sec on each
    // side (never beyond the cut itself), so the decoder gets little else.
    auto emit = [&](int64_t a, int64_t b, double end_sec) {
        const int64_t be = std::min(b, n);
        if (!has_speech(a, be)) return;
        double s0 = (double)a * fs, e0 = end_sec;
        if (o.trim_sec > 0.0) {
            int64_t fa = a, fb = be;
            while (fa < fb && !sp[(size_t)fa]) ++fa;
            while (fb > fa && !sp[(size_t)(fb - 1)]) --fb;
            s0 = std::max(s0, (double)fa * fs - o.trim_sec);
            e0 = std::min(e0, (double)fb * fs + o.trim_sec);
        }
        out.push_back({s0, e0});
    };
    int64_t s = 0;
    while (total_sec - (double)s * fs > o.max_seg_sec + 1e-9) {
        const int64_t lo = s + min_f, hi = s + max_f;
        int64_t c = -1, c_mid = -1;
        for (const auto& pz : pauses) {  // ordered, so later pauses overwrite
            const int64_t mid = (pz.first + pz.second) / 2;
            if (pz.first >= lo && pz.second <= hi) c = mid;
            if (mid >= lo && mid <= hi) c_mid = mid;
        }
        if (c < 0) c = c_mid;
        if (c <= s) c = hi;  // hard cut
        emit(s, c, (double)c * fs);
        s = c;
    }
    emit(s, n, total_sec);
    return out;
}

std::vector<VadSegment> speech_regions(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o) {
    std::vector<VadSegment> out;
    if (!(o.frame_sec > 0.0) || !std::isfinite(o.frame_sec) || !std::isfinite(total_sec) ||
        !std::isfinite(o.threshold) || !std::isfinite(o.min_pause_sec) ||
        !std::isfinite(o.bridge_sec) || !std::isfinite(o.min_speech_sec) ||
        !std::isfinite(o.pad_sec) || o.pad_sec < 0.0 || o.pad_sec > kMaxSec || !std::isfinite(o.run_gate) ||
        o.min_pause_sec > kMaxSec || o.bridge_sec > kMaxSec || o.min_speech_sec > kMaxSec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    const double fs = o.frame_sec;
    const int64_t n = std::max<int64_t>(0, (int64_t)std::ceil(total_sec / fs - 1e-9));
    const int64_t pause_f = std::max<int64_t>(1, (int64_t)std::ceil(o.min_pause_sec / fs - 1e-9));
    std::vector<char> sp((size_t)n, 0);
    for (int64_t f = 0; f < n && f < (int64_t)p.size(); ++f) sp[(size_t)f] = p[(size_t)f] >= o.threshold;
    apply_run_gate(sp, p, n, o.run_gate);
    auto for_runs = [&](bool value, auto&& fn) {
        int64_t f = 0;
        while (f < n) {
            if ((sp[(size_t)f] != 0) != value) { ++f; continue; }
            int64_t e = f;
            while (e < n && (sp[(size_t)e] != 0) == value) ++e;
            fn(f, e);
            f = e;
        }
    };
    for_runs(false, [&](int64_t a, int64_t b) {
        if (a > 0 && b < n && (double)(b - a) * fs + 1e-9 < o.bridge_sec)
            std::fill(sp.begin() + a, sp.begin() + b, 1);
    });
    for_runs(true, [&](int64_t a, int64_t b) {
        if ((double)(b - a) * fs + 1e-9 < o.min_speech_sec) std::fill(sp.begin() + a, sp.begin() + b, 0);
    });
    std::vector<std::pair<int64_t, int64_t>> runs;
    for_runs(true, [&](int64_t a, int64_t b) {
        if (!runs.empty() && a - runs.back().second < pause_f) runs.back().second = b;
        else runs.emplace_back(a, b);
    });
    for (const auto& r : runs)
        out.push_back({(double)r.first * fs, std::min((double)r.second * fs, total_sec)});
    if (o.pad_sec > 0.0) {
        const double pad = o.pad_sec;
        std::vector<VadSegment> padded = out;
        for (size_t i = 0; i < out.size(); ++i) {
            padded[i].start = std::max(0.0, out[i].start - pad);
            padded[i].end = std::min(total_sec, out[i].end + pad);
        }
        for (size_t i = 0; i + 1 < out.size(); ++i) {
            const double gap = out[i + 1].start - out[i].end;
            if (gap < 2.0 * pad) {
                const double mid = out[i].end + gap / 2.0;
                padded[i].end = mid;
                padded[i + 1].start = mid;
            }
        }
        out = padded;
    }
    return out;
}

SegmenterOpts default_segmenter_opts(VadKind kind) {
    SegmenterOpts o;
    if (kind == VadKind::kSilero) {
        o.frame_sec = 0.032;
        o.min_speech_sec = 0.25;
        o.min_pause_sec = 0.1;
        o.bridge_sec = 0.1;
        o.pad_sec = 0.03;
    }
    return o;
}

VadEventTracker::VadEventTracker(const SegmenterOpts& o) : o_(o) {
    degenerate_ = !(o.frame_sec > 0.0) || !std::isfinite(o.frame_sec) || !std::isfinite(o.threshold) ||
                  !std::isfinite(o.min_pause_sec) || !std::isfinite(o.bridge_sec) ||
                  !std::isfinite(o.min_speech_sec) || !std::isfinite(o.pad_sec) || o.pad_sec < 0.0 ||
                  o.pad_sec > kMaxSec || o.min_pause_sec > kMaxSec || o.bridge_sec > kMaxSec ||
                  o.min_speech_sec > kMaxSec;
    if (!degenerate_)
        pause_f_ = std::max<int64_t>(1, (int64_t)std::ceil(o.min_pause_sec / o.frame_sec - 1e-9));
}

void VadEventTracker::reset() {
    f_ = 0;
    run_open_ = false;
    run_start_ = run_end_ = 0;
    run_kept_ = false;
    reg_open_ = false;
    reg_end_ = 0;
    last_end_time_ = 0.0;
}

// The open run has reached min_speech_sec: it belongs to a region.
void VadEventTracker::keep_run(std::vector<VadEvent>* out) {
    run_kept_ = true;
    if (reg_open_ && run_start_ - reg_end_ >= pause_f_) {
        // Too far from the open region to join it. check_region_end normally
        // closed it already; this covers a run that grew while pending.
        emit_end(out, 1e300);
        reg_open_ = false;
    }
    if (!reg_open_) {
        reg_open_ = true;
        const double t = std::max(last_end_time_, (double)run_start_ * o_.frame_sec - o_.pad_sec);
        out->push_back({true, std::max(0.0, t)});
    }
    reg_end_ = run_end_;
}

void VadEventTracker::close_run(std::vector<VadEvent>* out) {
    (void)out;
    run_open_ = false;
    run_kept_ = false;
}

void VadEventTracker::emit_end(std::vector<VadEvent>* out, double total_sec) {
    const double t = std::min(total_sec, (double)reg_end_ * o_.frame_sec + o_.pad_sec);
    last_end_time_ = t;
    out->push_back({false, t});
}

// An open region ends once no run that starts later could join it.
void VadEventTracker::check_region_end(std::vector<VadEvent>* out) {
    if (!reg_open_) return;
    const int64_t next_start = run_open_ && !run_kept_ ? run_start_ : f_;
    if (run_open_ && run_kept_) return;  // the open run extends the region
    if (next_start - reg_end_ >= pause_f_) {
        emit_end(out, 1e300);
        reg_open_ = false;
    }
}

void VadEventTracker::push(float p, std::vector<VadEvent>* out) {
    const int64_t f = f_++;
    if (degenerate_) {
        if (f == 0) out->push_back({true, 0.0});
        return;
    }
    const double fs = o_.frame_sec;
    if (p >= o_.threshold) {
        if (run_open_ && f > run_end_ && (double)(f - run_end_) * fs + 1e-9 >= o_.bridge_sec) close_run(out);
        if (!run_open_) {
            run_open_ = true;
            run_start_ = f;
            run_kept_ = false;
        }
        run_end_ = f + 1;
        if (!run_kept_ && (double)(run_end_ - run_start_) * fs + 1e-9 >= o_.min_speech_sec) keep_run(out);
        else if (run_kept_) reg_end_ = run_end_;
    } else if (run_open_ && (double)(f + 1 - run_end_) * fs + 1e-9 >= o_.bridge_sec) {
        close_run(out);
    }
    check_region_end(out);
}

void VadEventTracker::finish(double total_sec, std::vector<VadEvent>* out) {
    if (degenerate_) {
        if (f_ == 0) out->push_back({true, 0.0});
        out->push_back({false, total_sec});
        return;
    }
    // A run that is open at the end is closed by the end of the stream; it was
    // kept already if it was long enough.
    run_open_ = false;
    run_kept_ = false;
    if (reg_open_) {
        emit_end(out, total_sec);
        reg_open_ = false;
    }
}

}  // namespace pk
