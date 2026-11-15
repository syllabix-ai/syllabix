#pragma once
#include <cstdint>
#include <vector>

namespace pk {

struct VadSegment {
    double start = 0.0;  // seconds
    double end = 0.0;
};

struct SegmenterOpts {
    float threshold = 0.5f;       // frame is speech when p >= threshold
    double frame_sec = 0.08;
    double max_seg_sec = 30.0;
    double min_pause_sec = 0.2;   // shortest silence that may be cut in
    double min_seg_sec = 1.0;     // earliest cut position inside a segment
    double bridge_sec = 0.1;      // speech gaps shorter than this are bridged
    double min_speech_sec = 0.1;  // speech runs shorter than this are dropped
    double pad_sec = 0.0;         // speech_regions only: padding added on both sides of a region
    // segment_by_vad only: a segment cut from long audio shrinks to its first
    // and last speech frame plus this many seconds on each side. 0 keeps the
    // whole cut (the behaviour before trimming existed). Audio of at most
    // max_seg_sec is not affected.
    double trim_sec = 0.3;
    // Run gate, off at 0. A speech run (a maximal stretch of frames with
    // p >= threshold, before bridging) is dropped when the median of the
    // probabilities of its frames is below this value, a probability in [0, 1).
    // A median equal to the gate keeps the run. For an even number of frames
    // the median is the mean of the two middle values. The gate acts first:
    // dropped frames are silence for bridging, min_speech, pauses and trim. It
    // applies to segment_by_vad and speech_regions, not to VadEventTracker. It
    // is meant for the Ultra/Redux heads, whose noise runs have a lower median
    // than speech runs (docs/vad.md). Any value <= 0 is off; NaN or infinity is
    // a degenerate option.
    float run_gate = 0.0f;
};

// Which model made the probabilities. The two kinds differ in frame period
// (0.08 s for the Ultra/Redux head, 0.032 s for Silero) and in the recommended
// post-processing, so each has its own option defaults. The segmenter code is
// the same for both; only the numbers differ.
enum class VadKind { kHead, kSilero };

// Defaults for a kind, with frame_sec set.
//  kHead:   the SegmenterOpts defaults, frame_sec 0.08, no padding.
//  kSilero: frame_sec 0.032, threshold 0.5, min_speech 0.25, min_pause 0.1,
//           bridge 0.1, pad 0.03. These are the values of Silero's own
//           get_speech_timestamps (threshold 0.5, min_speech_duration_ms 250,
//           min_silence_duration_ms 100, speech_pad_ms 30). Silero closes a
//           segment after min_silence of silence and then drops it when shorter
//           than min_speech; bridging gaps shorter than 0.1 s before the drop
//           gives the same order of operations. Silero's reference also lowers
//           the threshold by 0.15 once speech has started (hysteresis); this
//           segmenter has no hysteresis.
SegmenterOpts default_segmenter_opts(VadKind kind);

// Cuts [0, total_sec] into segments of at most max_seg_sec, at pauses found in
// the per-frame speech probabilities p.
//
// 1. A frame is speech when p >= threshold. With run_gate > 0, a run of speech
//    frames whose median probability is below run_gate becomes silence. Speech gaps shorter than bridge_sec
//    are filled, then speech runs shorter than min_speech_sec are removed. The
//    remaining silent runs of at least min_pause_sec are the pauses.
// 2. Audio of at most max_seg_sec is returned whole, with or without speech.
// 3. Longer audio is cut from the front. For a segment starting at s the cut is
//    the midpoint of the last pause that lies fully inside [s + min_seg_sec,
//    s + max_seg_sec]; else the midpoint of the last pause whose midpoint lies
//    in that range; else a hard cut at s + max_seg_sec.
// 4. Segments that contain no speech are dropped, including the trailing
//    remainder. The result is empty when no segment has speech. Kept segments
//    are ordered and disjoint but need not touch.
// 5. With trim_sec > 0 each kept segment is cut down to [first speech frame -
//    trim_sec, end of last speech frame + trim_sec], limited to the segment
//    itself. Speech is the smoothed mask of step 1. This keeps the quiet or
//    noisy parts that a hard or pause cut leaves at the edges away from the
//    decoder. A segment is never trimmed to nothing: it has speech, so it keeps
//    at least that run. Trimmed edges are not on the frame grid.
//
// Degenerate options (frame_sec not finite or <= 0, max_seg_sec not finite or
// <= 2 * frame_sec, threshold, run_gate, trim_sec or any of the four durations not finite,
// any duration above 1e6 seconds) return the single segment {0, total_sec}.
// Every cut between two segments is a whole number of frames.
std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

// The speech regions themselves, for any audio length (no cap, no cuts).
// Uses threshold, frame_sec, bridge_sec, min_speech_sec and min_pause_sec: the
// speech mask is gated (run_gate) and smoothed as in segment_by_vad (bridge gaps shorter than
// bridge_sec, drop runs shorter than min_speech_sec), then speech runs
// separated by a gap shorter than min_pause_sec are merged. Regions are
// ordered, disjoint and inside [0, total_sec]; the result is empty when there
// is no speech. Each region is then widened by pad_sec on both sides, clamped
// to [0, total_sec]; when the gap between two regions is shorter than
// 2 * pad_sec, both move to the middle of the gap, so regions stay disjoint.
// Degenerate options (also a pad that is negative, not finite or above 1e6
// seconds) return the single region {0, total_sec}.
std::vector<VadSegment> speech_regions(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

// The same speech regions as speech_regions, produced as the frames arrive. Push
// one probability per frame; a "start" or "end" event is appended when it is
// known. A start is known once a speech run reaches min_speech_sec, an end once
// the silence after the last kept run reaches min_pause_sec, so both arrive late
// but carry the time of the speech boundary (padded by pad_sec). finish() closes
// the stream. For pad_sec <= min_pause_sec / 2 the events give exactly the
// regions of speech_regions on the same probabilities (a test checks this). For
// a larger pad the start of a region is kept at or after the end of the
// previous one instead of at the middle of the gap.
struct VadEvent {
    bool start = false;  // false: end
    double time = 0.0;   // seconds from the first frame
};

class VadEventTracker {
public:
    explicit VadEventTracker(const SegmenterOpts& o);
    void reset();
    // Frames pushed so far.
    int64_t frames() const { return f_; }
    void push(float p, std::vector<VadEvent>* out);
    // total_sec: audio length, the end of the last frame may be past it.
    void finish(double total_sec, std::vector<VadEvent>* out);

private:
    void keep_run(std::vector<VadEvent>* out);
    void close_run(std::vector<VadEvent>* out);
    void check_region_end(std::vector<VadEvent>* out);
    void emit_end(std::vector<VadEvent>* out, double total_sec);
    SegmenterOpts o_;
    bool degenerate_ = false;
    int64_t pause_f_ = 1;
    int64_t f_ = 0;               // next frame index
    bool run_open_ = false;       // a speech run, with bridged gaps, is in progress
    int64_t run_start_ = 0, run_end_ = 0;  // frames, end exclusive (last speech frame + 1)
    bool run_kept_ = false;       // the open run already reached min_speech_sec
    bool reg_open_ = false;       // a region is open (start event emitted)
    int64_t reg_end_ = 0;         // end of the last kept run
    double last_end_time_ = 0.0;  // time of the last end event
};

}  // namespace pk
