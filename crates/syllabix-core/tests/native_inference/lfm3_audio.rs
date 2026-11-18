//! Opt-in LFM3-Audio S2S benchmarks (`SYLLABIX_NATIVE_MODELS=lfm3-audio-2b`).

use std::fs;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use syllabix_core::config::{AgentConfig, PipelineMode, S2sConfig};
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::pipeline::{LoopEvent, LoopMode};
use syllabix_core::s2s::{Lfm3AudioEngine, S2sEngine, LFM3_AUDIO_2B, LFM3_AUDIO_2B_MMPROJ};
use syllabix_core::testing::CollectingSink;
use syllabix_core::types::{SynthesizedAudio, TurnId, FRAME_SAMPLES};
use syllabix_core::vad::END_SILENCE_FRAMES;
use syllabix_core::AudioSink;
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    Cancel, ModelCache, Result, Session, Utterance,
};
use syllabix_native::Lfm3AudioContext;

use crate::{hex, native, skip_unless_model};

const FIXTURE: &str = "librispeech-1089-134686-0000.wav";
const FIXTURE_SHA256: &str = "c6517d6052651b2ff22d19526f49fc7da5fc56de30790b9ed54238abd9c48c36";
const N_THREADS: i32 = 4;
const CAP_OUT_SAMPLES: i32 = 4 * 24_000;
const CAP_OUT_S: f64 = 4.0;
const NATIVE_OUT_RATE: f64 = 24_000.0;

#[derive(Clone, Default)]
struct TimedSink {
    inner: Arc<Mutex<CollectingSink>>,
    first_pcm_at: Arc<Mutex<Option<Instant>>>,
}

impl TimedSink {
    fn pcm_samples(&self) -> usize {
        self.inner
            .lock()
            .expect("sink")
            .chunks
            .iter()
            .map(|c| c.samples.len())
            .sum()
    }
}

impl AudioSink for TimedSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if !audio.samples.is_empty() {
            let mut slot = self.first_pcm_at.lock().expect("first_pcm");
            if slot.is_none() {
                *slot = Some(Instant::now());
            }
        }
        self.inner.lock().expect("sink").play(audio, cancel)
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        self.inner.lock().expect("sink").finish_turn(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.inner.lock().expect("sink").interrupt();
    }
}

fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Ok(kb) = rest.split_whitespace().next().unwrap_or("0").parse::<u64>() {
                return kb as f64 / 1024.0;
            }
        }
    }
    0.0
}

fn fixture_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(FIXTURE);
    fs::read(&path).expect("fixture bytes")
}

fn short_fixture_utterance(max_secs: f64) -> Utterance {
    let mut utterance = fixture_utterance();
    let max_frames = ((max_secs * 16_000.0) / FRAME_SAMPLES as f64)
        .ceil()
        .max(1.0) as usize;
    utterance
        .frames
        .truncate(max_frames.min(utterance.frames.len()));
    utterance
}

fn fixture_utterance() -> Utterance {
    let bytes = fixture_bytes();
    assert_eq!(hex(Sha256::digest(&bytes)), FIXTURE_SHA256, "fixture hash");
    let wav = read_wav(Cursor::new(bytes)).expect("wav");
    let mut frames = record_fixture_to_frames(&wav).expect("frames");
    for frame in &mut frames {
        if !frame.has_energy() {
            frame.samples[0] = 1;
        }
    }
    Utterance {
        turn: TurnId(0),
        frames,
    }
}

fn fixture_frames_with_trailing_silence() -> Vec<syllabix_core::AudioFrame> {
    let wav = read_wav(Cursor::new(fixture_bytes())).expect("wav");
    let mut frames = record_fixture_to_frames(&wav).expect("frames");
    let mut seq = frames.last().map(|f| f.seq + 1).unwrap_or(0);
    let pad = END_SILENCE_FRAMES + 4;
    for _ in 0..pad {
        frames.push(
            syllabix_core::AudioFrame::new(
                seq,
                syllabix_core::types::DEFAULT_SAMPLE_RATE_HZ,
                syllabix_core::types::DEFAULT_CHANNELS,
                vec![0; FRAME_SAMPLES],
            )
            .expect("silence frame"),
        );
        seq += 1;
    }
    frames
}

fn load_engine() -> Lfm3AudioEngine {
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    Lfm3AudioEngine::from_cache(&cache, &HttpFetcher, &mut progress, &cancel, N_THREADS)
        .expect("load lfm3-audio-2b")
        .with_max_out_samples(CAP_OUT_SAMPLES)
}

fn shared_engine() -> std::sync::MutexGuard<'static, Lfm3AudioEngine> {
    static CELL: OnceLock<Mutex<Lfm3AudioEngine>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(load_engine()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn host_label() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "linux".into())
}

#[test]
fn lfm3_audio_engine_generate() {
    skip_unless_model!(LFM3_AUDIO_2B);
    let _native = native();

    let host = host_label();
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    eprintln!(
        "lfm3-audio-2b: host={host} cpus={cpus} threads={N_THREADS} cap_out_s={CAP_OUT_S:.2} fixture={FIXTURE}"
    );
    eprintln!("lfm3-audio-2b: phase0_median ttfa_ms=665 gen_rtf=0.95 peak_rss_mib=4034");

    let utterance = fixture_utterance();
    let in_s = utterance.pcm().len() as f64 / 16_000.0;

    drop(shared_engine());

    let mut first_audio_at: Option<Instant> = None;
    let t0 = Instant::now();
    let mut engine = shared_engine();
    engine
        .generate(&utterance, &Cancel::new(), &mut |audio| {
            if first_audio_at.is_none() && !audio.samples.is_empty() {
                first_audio_at = Some(Instant::now());
            }
            Ok(())
        })
        .expect("engine generate");
    let gen_s = t0.elapsed().as_secs_f64();
    let ttfa_ms = first_audio_at
        .map(|t| (t - t0).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let ttfa_s = first_audio_at
        .map(|t| (t - t0).as_secs_f64())
        .unwrap_or(0.0);
    let after_ttfa_s = gen_s - ttfa_s;
    let native_out_s = engine.last_native_out_samples() as f64 / NATIVE_OUT_RATE;
    let headline_rtf = if native_out_s > 0.0 {
        gen_s / native_out_s
    } else {
        0.0
    };
    let gen_rtf = if native_out_s > 0.0 {
        after_ttfa_s / native_out_s
    } else {
        0.0
    };
    let peak_rss_mib = peak_rss_mib();
    eprintln!(
        "lfm3-audio-2b: engine turn=1 in_s={in_s:.3} ttfa_ms={ttfa_ms:.0} gen_after_ttfa_s={after_ttfa_s:.3} out_s={native_out_s:.3} gen_rtf={gen_rtf:.2} headline_rtf={headline_rtf:.2} peak_rss_mib={peak_rss_mib:.0}"
    );
}

#[test]
fn lfm3_audio_session_wiring_without_with_s2s() {
    skip_unless_model!(LFM3_AUDIO_2B);
    let _native = native();

    let mut config = AgentConfig::v0();
    config.pipeline_mode = PipelineMode::S2s;
    config.s2s = Some(S2sConfig {
        model: Some(LFM3_AUDIO_2B.to_string()),
    });
    let sink = TimedSink::default();
    let thinking_at = Arc::new(Mutex::new(None::<Instant>));
    let thinking_listener = thinking_at.clone();
    let (tx, rx) = mpsc::channel();
    let listener = std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            if matches!(event, LoopEvent::Thinking { .. }) {
                let mut slot = thinking_listener.lock().expect("thinking");
                if slot.is_none() {
                    *slot = Some(Instant::now());
                }
            }
        }
    });

    let session_t0 = Instant::now();
    let report = Session::from_config(&config)
        .with_sink(sink.clone())
        .with_events(Some(tx))
        .with_frames(fixture_frames_with_trailing_silence())
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("s2s session without with_s2s");
    listener.join().expect("event listener");

    assert_eq!(report.tasks_exited, 4);
    assert_eq!(report.turns.len(), 1);
    assert!(sink.pcm_samples() > 0, "sink must receive synthesized PCM");
    let session_ttfa_ms = match (
        *thinking_at.lock().expect("thinking"),
        *sink.first_pcm_at.lock().expect("first_pcm"),
    ) {
        (Some(speech_end), Some(first_pcm)) => (first_pcm - speech_end).as_secs_f64() * 1000.0,
        _ => 0.0,
    };
    eprintln!(
        "lfm3-audio-2b: session turn=1 tasks_exited={} session_ttfa_ms={session_ttfa_ms:.0} session_wall_ms={:.0} sink_samples={} (Silero via resolve_s2s_vad)",
        report.tasks_exited,
        session_t0.elapsed().as_secs_f64() * 1000.0,
        sink.pcm_samples()
    );
}

#[test]
fn lfm3_audio_engine_cancel() {
    skip_unless_model!(LFM3_AUDIO_2B);
    let _native = native();
    assert_engine_honours_cancel(&mut load_engine());
}

/// Engine abort is the native barge-in gate; loop barge-in stays on `fake_s2s_loop` (FakeS2s + AEC).
fn assert_engine_honours_cancel(engine: &mut Lfm3AudioEngine) {
    let already = Cancel::new();
    already.shutdown();
    let pre = engine
        .generate(&short_fixture_utterance(0.1), &already, &mut |_| Ok(()))
        .expect_err("pre-shutdown must cancel");
    assert!(
        matches!(pre, syllabix_core::Error::Cancelled),
        "pre-shutdown got {pre:?}"
    );

    let utterance = short_fixture_utterance(1.0);
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    let saw_pcm = Arc::new(AtomicBool::new(false));
    let saw_pcm_for_wait = saw_pcm.clone();
    let wait_start = Instant::now();
    std::thread::spawn(move || {
        while !saw_pcm_for_wait.load(Ordering::Relaxed)
            && wait_start.elapsed() < Duration::from_millis(200)
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        cancel_thread.cancel_generation();
    });

    let start = Instant::now();
    let cancel_on_pcm = cancel.clone();
    let err = engine
        .generate(&utterance, &cancel, &mut |audio| {
            if !audio.samples.is_empty() {
                saw_pcm.store(true, Ordering::Relaxed);
                cancel_on_pcm.cancel_generation();
            }
            Ok(())
        })
        .expect_err("cancelled generate must error");
    assert!(
        matches!(err, syllabix_core::Error::Cancelled),
        "got {err:?}"
    );
    let cancel_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "lfm3-audio-2b: engine cancel_ms={cancel_ms:.0} saw_first_pcm={}",
        saw_pcm.load(Ordering::Relaxed)
    );
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "engine cancel took {:?}, budget is 5 s",
        start.elapsed()
    );
}

#[test]
fn lfm3_s2s_n_gpu_layers_matches_host() {
    skip_unless_model!(LFM3_AUDIO_2B);
    let _native = native();
    let cache = ModelCache::v0();
    let manifest = syllabix_native::models::Manifest::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let fetcher = HttpFetcher;
    let model = cache
        .resolve(
            manifest.asset(LFM3_AUDIO_2B).expect("manifest backbone"),
            &fetcher,
            &mut progress,
            &cancel,
        )
        .expect("model path");
    let mmproj = cache
        .resolve(
            manifest
                .asset(LFM3_AUDIO_2B_MMPROJ)
                .expect("manifest mmproj"),
            &fetcher,
            &mut progress,
            &cancel,
        )
        .expect("mmproj path");
    let ctx = Lfm3AudioContext::load(&model, &mmproj, N_THREADS, u32::MAX).expect("load s2s");
    assert_eq!(
        ctx.n_gpu_layers(),
        syllabix_native::llama_n_gpu_layers(),
        "S2S must use the same GPU placement as cascade LLM and Whisper"
    );
}
