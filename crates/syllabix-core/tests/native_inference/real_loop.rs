//! Six real-provider turns exercising queue bounds, shutdown, and the RSS ceiling.
//!
//! Does not score LLM reply text. Non-empty assistant text is a completion check
//! only when the suite finishes within the sustainment wall budget; longer than
//! that budget is a suspicious host/generation stall.
//!
//! Capture is paced: each clip waits for VAD to reopen after TTS so later turns
//! are not dropped while the assistant holds the mic.

use std::collections::VecDeque;
use std::fs;
use std::io::Cursor;
use std::thread;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, DrainingPlayback, PcmFormat},
    process_rss_bytes, run_loop_captured, AudioCapture, AudioFrame, Cancel, Error, HttpFetcher,
    LoopConfig, ModelCache, PipelineStages, Result, RuntimeControls, SileroVad, StderrProgress,
    END_SILENCE_FRAMES, FRAME_SAMPLES, LOOP_RSS_GROWTH_CEILING_BYTES,
};

use crate::{hex, native, skip_unless_launch_stack};

const SPEECH_SHA256: &str = "e72f1ccf42dc827252141e927a0969793169fbe8039392e207285aa306f09daa";
const REAL_TURNS: usize = 6;
/// Whole-suite wall budget. Paced launch-stack turns (STT+LFM+Pocket) need
/// minutes on modest CPUs / llvm-cov; over this is a suspicious stall.
const SUSTAINMENT_BUDGET: Duration = Duration::from_secs(300);
const CAPTURE_POLL: Duration = Duration::from_millis(5);
/// If SpeechEnd never closes capture (pathological), stop waiting and feed the next clip.
const CAPTURE_CLOSE_GRACE: Duration = Duration::from_secs(30);

fn speech_frames() -> Vec<AudioFrame> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vad/speech.wav");
    let bytes = fs::read(&path).expect("speech.wav");
    assert_eq!(hex(Sha256::digest(&bytes)), SPEECH_SHA256);
    let wav = read_wav(Cursor::new(bytes)).expect("wav");
    record_fixture_to_frames(&wav).expect("frames")
}

fn silence_frame(seq: u64) -> AudioFrame {
    AudioFrame::new(
        seq,
        syllabix_core::DEFAULT_SAMPLE_RATE_HZ,
        syllabix_core::DEFAULT_CHANNELS,
        vec![0; FRAME_SAMPLES],
    )
    .expect("silence")
}

/// One speech clip plus endpoint silence, per turn.
fn six_turn_clips() -> Vec<Vec<AudioFrame>> {
    let speech = speech_frames();
    let mut seq = 0_u64;
    let pad = END_SILENCE_FRAMES + 2;
    (0..REAL_TURNS)
        .map(|_| {
            let mut clip = Vec::new();
            for mut frame in speech.clone() {
                frame.seq = seq;
                seq += 1;
                clip.push(frame);
            }
            for _ in 0..pad {
                clip.push(silence_frame(seq));
                seq += 1;
            }
            clip
        })
        .collect()
}

/// Feeds one clip at a time. After each clip, waits for VAD to pause (assistant
/// hold) then reopen before the next clip — otherwise firehose audio is dropped
/// while TTS plays.
struct PacedTurnCapture {
    controls: RuntimeControls,
    remaining: VecDeque<Vec<AudioFrame>>,
    current: std::vec::IntoIter<AudioFrame>,
    await_reopen: bool,
}

impl PacedTurnCapture {
    fn new(controls: RuntimeControls, clips: Vec<Vec<AudioFrame>>) -> Self {
        let mut remaining: VecDeque<_> = clips.into();
        let current = remaining.pop_front().unwrap_or_default().into_iter();
        Self {
            controls,
            remaining,
            current,
            await_reopen: false,
        }
    }

    fn wait_for_reopen(&mut self, cancel: &Cancel) -> Result<()> {
        let grace = Instant::now() + CAPTURE_CLOSE_GRACE;
        // Phase 1: wait until the assistant holds capture (or grace expires on a
        // pathologically fast path that never paused).
        while self.controls.capture_open() {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if Instant::now() >= grace {
                break;
            }
            thread::sleep(CAPTURE_POLL);
        }
        // Phase 2: wait until listening resumes after TTS.
        while !self.controls.capture_open() {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            thread::sleep(CAPTURE_POLL);
        }
        Ok(())
    }
}

impl AudioCapture for PacedTurnCapture {
    fn name(&self) -> &'static str {
        "paced-turn"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }

            if self.await_reopen {
                self.wait_for_reopen(cancel)?;
                self.await_reopen = false;
                let Some(clip) = self.remaining.pop_front() else {
                    return Ok(None);
                };
                self.current = clip.into_iter();
            }

            if let Some(frame) = self.current.next() {
                return Ok(Some(frame));
            }

            if self.remaining.is_empty() {
                return Ok(None);
            }
            // Finished a clip with more queued: wait for the listen window.
            self.await_reopen = true;
        }
    }
}

fn turn_summaries(
    report: &syllabix_core::LoopReport,
) -> Vec<(syllabix_core::TurnId, usize, usize, usize)> {
    report
        .turns
        .iter()
        .map(|t| {
            (
                t.id,
                t.user_text.trim().len(),
                t.assistant_text.trim().len(),
                t.audio_chunks,
            )
        })
        .collect()
}

#[test]
fn six_real_turns_complete_within_queue_and_memory_bounds() {
    skip_unless_launch_stack!();
    let mut n = native();
    // Fresh Silero session: GRU hangover on the shared instance would stall later turns.
    let vad = SileroVad::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Silero for six-turn");
    // Warm providers before the RSS baseline so growth measures the loop, not loads.
    let stt = n.stt().clone();
    let llm = n.llm().clone();
    let tts = n.take_tts();
    let warmup_rss = process_rss_bytes();

    let sink = DrainingPlayback::new(PcmFormat {
        sample_rate_hz: 48_000,
        channels: 2,
    })
    .expect("playback converter");
    let drain = sink.stats();

    let config = LoopConfig::six_turns();
    let capture = PacedTurnCapture::new(config.controls.clone(), six_turn_clips());

    let started = Instant::now();
    let report = run_loop_captured(
        // StopAfterTurns(6): native sustainment gate, not UntilInputEnds.
        config,
        PipelineStages {
            vad,
            stt,
            llm,
            tts,
            sink,
        },
        capture,
        Cancel::new(),
    )
    .expect("real loop");
    let elapsed = started.elapsed();

    assert!(
        elapsed <= SUSTAINMENT_BUDGET,
        "six-turn sustainment exceeded {:?} budget (took {:?}); suspicious host/generation stall; turns={:?}",
        SUSTAINMENT_BUDGET,
        elapsed,
        turn_summaries(&report)
    );

    assert_eq!(report.tasks_exited, 6);
    assert_eq!(report.tasks_still_running, 0);
    assert!(
        report.queues.within_capacity(),
        "queue occupancy exceeded a bound: {:?}",
        report.queues
    );
    for occupancy in report.queues.all() {
        assert_eq!(
            occupancy.current, 0,
            "{} still held {} items after shutdown",
            occupancy.name, occupancy.current
        );
    }
    assert_eq!(
        report.turns.len(),
        REAL_TURNS,
        "expected {REAL_TURNS} real turns in {:?}, got {:?}",
        elapsed,
        turn_summaries(&report)
    );
    for (i, turn) in report.turns.iter().enumerate() {
        assert_eq!(turn.id, syllabix_core::TurnId(i as u64));
        assert!(
            !turn.user_text.trim().is_empty(),
            "turn {i} missing STT text"
        );
        // Under the wall budget: non-empty assistant text is a completion check,
        // not an LLM-quality gate. (Over-budget already failed above.)
        assert!(
            !turn.assistant_text.trim().is_empty(),
            "turn {i} missing LLM text after {:?} (under {:?} budget)",
            elapsed,
            SUSTAINMENT_BUDGET
        );
        assert!(turn.audio_chunks > 0, "turn {i} missing TTS audio");
        if i > 0 {
            assert!(turn.id > report.turns[i - 1].id);
        }
    }
    assert!(drain.chunks() > 0);
    assert!(drain.samples_played() > 0);
    assert!(
        drain.live_high_water() <= syllabix_core::audio::AUDIO_LIVE_BYTES_CEILING,
        "playback leftover {} exceeded {}",
        drain.live_high_water(),
        syllabix_core::audio::AUDIO_LIVE_BYTES_CEILING
    );

    if let (Some(before), Some(after)) = (warmup_rss, process_rss_bytes()) {
        let growth = after.saturating_sub(before);
        assert!(
            growth <= LOOP_RSS_GROWTH_CEILING_BYTES,
            "RSS grew {growth} bytes after warm-up (ceiling {LOOP_RSS_GROWTH_CEILING_BYTES}); before={before} after={after}"
        );
    }
}
