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
use syllabix_core::config::{STT_API_KEY_ENV, TTS_API_KEY_ENV};
use syllabix_core::defaults::{SttProvider, TtsProvider};
use syllabix_core::llm::LFM25_230M_ASSET;
use syllabix_core::memory::{process_rss_bytes, LOOP_RSS_GROWTH_CEILING_BYTES};
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::real::{build_stt, build_tts, LiveStt, LiveTts};
use syllabix_core::types::FRAME_SAMPLES;
use syllabix_core::vad::{SileroVad, END_SILENCE_FRAMES};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, DrainingPlayback, PcmFormat},
    run_loop_captured, AgentConfig, AudioCapture, AudioFrame, Cancel, Error, LoopConfig,
    ModelCache, PipelineStages, Result, RuntimeControls,
};

use crate::{hex, launch_stack_selected, native, native_model_selected, Native};

/// When set, the six-turn loop uses `build_stt` / `build_tts` remote clients
/// instead of in-process Whisper / Pocket TTS. Keys stay on
/// `SYLLABIX_STT_API_KEY` / `SYLLABIX_TTS_API_KEY` (no shared fallback).
const STT_BASE_URL_ENV: &str = "SYLLABIX_STT_BASE_URL";
const TTS_BASE_URL_ENV: &str = "SYLLABIX_TTS_BASE_URL";

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
        syllabix_core::types::DEFAULT_SAMPLE_RATE_HZ,
        syllabix_core::types::DEFAULT_CHANNELS,
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
    // Default launch stack (LFM3-1.7B when SYLLABIX_NATIVE_MODELS is unset):
    // run the loop on lfm3-1.7b via n.llm(). Weekly CI sets whisper-small
    // + lfm2.5-230m + pocket-tts instead; run the same completion check on
    // 230m via the dedicated LFM slot. Other id lists skip.
    let weekly_230m_stack = !launch_stack_selected()
        && native_model_selected("whisper-small")
        && native_model_selected("pocket-tts")
        && native_model_selected(LFM25_230M_ASSET);
    if !launch_stack_selected() && !weekly_230m_stack {
        eprintln!(
            "skipping six-turn native test (needs whisper-small, pocket-tts, and lfm3-1.7b or lfm2.5-230m in SYLLABIX_NATIVE_MODELS)"
        );
        return;
    }
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
    // URL env vars swap STT/TTS to the remote clients; VAD and LLM stay in-process.
    let stt = live_stt(&mut n);
    let llm = if weekly_230m_stack {
        n.lfm_asset_mut(LFM25_230M_ASSET).clone()
    } else {
        n.llm().clone()
    };
    let tts = live_tts(&mut n);
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
        "expected {REAL_TURNS} real turns in {:?}, got {:?} skipped={} cancelled={}",
        elapsed,
        turn_summaries(&report),
        report.skipped_turns,
        report.cancelled
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

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn live_stt(n: &mut Native) -> LiveStt {
    match env_nonempty(STT_BASE_URL_ENV) {
        Some(url) => {
            let key = syllabix_core::remote_stt::resolve_api_key(|name| std::env::var(name).ok())
                .unwrap_or_else(|err| panic!("{STT_API_KEY_ENV}: {err}"));
            let mut config = AgentConfig::v0();
            config.stt = SttProvider::Online;
            config.stt_base_url = Some(url);
            build_stt(
                &ModelCache::v0(),
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
                &config,
                Some(&key),
            )
            .unwrap_or_else(|err| panic!("remote STT client: {err}"))
        }
        None => LiveStt::Whisper(n.stt().clone()),
    }
}

fn live_tts(n: &mut Native) -> LiveTts {
    match env_nonempty(TTS_BASE_URL_ENV) {
        Some(url) => {
            let key = syllabix_core::remote_tts::resolve_api_key(|name| std::env::var(name).ok())
                .unwrap_or_else(|err| panic!("{TTS_API_KEY_ENV}: {err}"));
            let mut config = AgentConfig::v0();
            config.tts = TtsProvider::Online;
            config.tts_base_url = Some(url);
            build_tts(
                &ModelCache::v0(),
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
                &config,
                Some(&key),
            )
            .unwrap_or_else(|err| panic!("remote TTS client: {err}"))
        }
        None => n.take_tts(),
    }
}
