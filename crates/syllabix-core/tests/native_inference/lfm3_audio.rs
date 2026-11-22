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
use syllabix_core::pocket_tts::PocketTts;
use syllabix_core::s2s::{
    Lfm3AudioEngine, S2sEngine, LFM3_AUDIO_2B, LFM3_AUDIO_2B_MMPROJ,
    LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT, S2S_INTERRUPT_MARKER,
};
use syllabix_core::testing::CollectingSink;
use syllabix_core::tts::Tts;
use syllabix_core::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId, FRAME_SAMPLES};
use syllabix_core::vad::END_SILENCE_FRAMES;
use syllabix_core::AudioSink;
use syllabix_core::Stt;
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, write_wav, WavPcm},
    Cancel, ModelCache, Result, Session, Utterance,
};
use syllabix_native::Lfm3AudioContext;
use syllabix_stt::{SttModel, WhisperStt};

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

fn reset_shared_engine() {
    shared_engine().reset_conversation();
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
    let utterance = fixture_utterance();
    let in_s = utterance.pcm().len() as f64 / 16_000.0;

    reset_shared_engine();

    let mut first_audio_at: Option<Instant> = None;
    let t0 = Instant::now();
    let mut engine = shared_engine();
    engine
        .generate(
            &utterance,
            &Cancel::new(),
            &mut |audio| {
                if first_audio_at.is_none() && !audio.samples.is_empty() {
                    first_audio_at = Some(Instant::now());
                }
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
        .expect("engine generate");
    let reply_text = engine
        .last_text()
        .unwrap_or_else(|| panic!("mixed s2s must expose reply text"));
    let word_count = reply_text.split_whitespace().count();
    assert!(
        word_count >= 3,
        "mixed reply text too short ({word_count} words): {reply_text:?}"
    );
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
    assert!(
        gen_s < 42.0,
        "generate wall time {gen_s:.1}s exceeded 42 s cap"
    );
    eprintln!(
        "lfm3-audio-2b: engine turn=1 in_s={in_s:.3} ttfa_ms={ttfa_ms:.0} gen_after_ttfa_s={after_ttfa_s:.3} out_s={native_out_s:.3} gen_rtf={gen_rtf:.2} headline_rtf={headline_rtf:.2} reply_words={word_count} gen_wall_s={gen_s:.2} peak_rss_mib={peak_rss_mib:.0} reply_text={reply_text:?}"
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
        .generate(
            &short_fixture_utterance(0.1),
            &already,
            &mut |_| Ok(()),
            &mut |_, _| Ok(()),
        )
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
        .generate(
            &utterance,
            &cancel,
            &mut |audio| {
                if !audio.samples.is_empty() {
                    saw_pcm.store(true, Ordering::Relaxed);
                    cancel_on_pcm.cancel_generation();
                }
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
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
    let ctx = Lfm3AudioContext::load(
        &model,
        &mmproj,
        N_THREADS,
        u32::MAX,
        0,
        LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
    )
    .expect("load s2s");
    assert_eq!(
        ctx.n_gpu_layers(),
        syllabix_native::llama_n_gpu_layers(),
        "S2S must use the same GPU placement as cascade LLM and Whisper"
    );
}

/// Manual check: write 24 kHz reply WAV and transcribe with whisper-small (not in default CI).
#[test]
#[ignore = "manual native: SYLLABIX_NATIVE_MODELS=lfm3-audio-2b,whisper-small cargo test -p syllabix-core --features native-inference --test native_inference lfm3_audio_reply_whisper -- --ignored --nocapture"]
fn lfm3_audio_reply_whisper() {
    skip_unless_model!(LFM3_AUDIO_2B);
    skip_unless_model!("whisper-small");
    let _native = native();
    reset_shared_engine();

    let utterance = fixture_utterance();
    let mut pcm = Vec::new();
    shared_engine()
        .generate(
            &utterance,
            &Cancel::new(),
            &mut |audio| {
                pcm.extend_from_slice(&audio.samples);
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
        .expect("generate");
    assert!(!pcm.is_empty(), "reply must include PCM");

    let wav = WavPcm {
        format: syllabix_native::audio::PcmFormat {
            sample_rate_hz: NATIVE_OUT_RATE as u32,
            channels: 1,
        },
        samples: pcm,
    };
    let out_path = std::path::Path::new("/opt/cursor/artifacts/lfm3_s2s_reply.wav");
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    write_wav(fs::File::create(out_path).expect("create wav"), &wav).expect("write wav");
    eprintln!("wrote {}", out_path.display());

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let whisper_path = cache
        .resolve(
            cache
                .manifest()
                .asset("whisper-small")
                .expect("whisper asset"),
            &HttpFetcher,
            &mut progress,
            &cancel,
        )
        .expect("whisper weights");
    let mut stt = WhisperStt::from_model_path(whisper_path, SttModel::Small).expect("whisper");
    let frames = record_fixture_to_frames(&wav).expect("reply frames");
    let transcript = stt
        .transcribe(
            &Utterance {
                turn: TurnId(99),
                frames,
            },
            &Cancel::new(),
        )
        .expect("whisper transcript");
    eprintln!("whisper-small transcript={:?}", transcript.text.trim());
    assert!(
        transcript.text.split_whitespace().count() >= 2,
        "whisper must hear speech: {:?}",
        transcript.text
    );
}

fn wav_fixture_utterance(path: &std::path::Path, turn: TurnId) -> Utterance {
    let wav = read_wav(fs::File::open(path).expect("open wav")).expect("wav");
    let frames = record_fixture_to_frames(&wav).expect("frames");
    Utterance { turn, frames }
}

fn resolve_lfm3_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let cache = ModelCache::v0();
    let manifest = syllabix_native::models::Manifest::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let fetcher = HttpFetcher;
    let model = cache
        .resolve(
            manifest.asset(LFM3_AUDIO_2B).expect("backbone"),
            &fetcher,
            &mut progress,
            &cancel,
        )
        .expect("model");
    let mmproj = cache
        .resolve(
            manifest.asset(LFM3_AUDIO_2B_MMPROJ).expect("mmproj"),
            &fetcher,
            &mut progress,
            &cancel,
        )
        .expect("mmproj");
    (model, mmproj)
}

fn judge_barge_followup_text_delta(text: &str) -> &'static str {
    let lower = text.trim().to_lowercase();
    if lower.is_empty() {
        return "empty";
    }
    if lower.contains(&S2S_INTERRUPT_MARKER.to_lowercase()) {
        "echoed marker"
    } else if lower.contains("sorry") || lower.contains("apolog") {
        "apologised"
    } else if lower.split_whitespace().count() >= 2 {
        "coherent (no marker verbatim)"
    } else {
        "short or unclear"
    }
}

fn report_post_barge_reply(last_text: &str, whisper: &str) -> String {
    let primary = judge_barge_followup_text_delta(last_text);
    let cross = judge_barge_followup_text_delta(whisper);
    let agree = if last_text.trim().is_empty() || whisper.trim().is_empty() {
        "n/a"
    } else if primary == cross {
        "agrees"
    } else {
        "differs"
    };
    format!("primary(last_text)={primary} whisper_cross_check={cross} ({agree})")
}

/// Manual: barge-in on turn 1, then turn 2 reply (report last_text + Whisper in PR body).
#[test]
#[ignore = "manual native: SYLLABIX_NATIVE_MODELS=lfm3-audio-2b,whisper-small cargo test -p syllabix-core --features native-inference --test native_inference lfm3_audio_barge_in_next_reply -- --ignored --nocapture"]
fn lfm3_audio_barge_in_next_reply() {
    skip_unless_model!(LFM3_AUDIO_2B);
    skip_unless_model!("whisper-small");
    let _native = native();
    reset_shared_engine();
    let mut engine = shared_engine();
    let turn1 = short_fixture_utterance(0.8);
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    let saw_pcm = Arc::new(AtomicBool::new(false));
    let saw = saw_pcm.clone();
    std::thread::spawn(move || {
        while !saw.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
        }
        cancel_thread.cancel_generation();
    });
    let err = engine
        .generate(
            &turn1,
            &cancel,
            &mut |audio| {
                if !audio.samples.is_empty() {
                    saw_pcm.store(true, Ordering::Relaxed);
                }
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
        .expect_err("turn 1 must cancel");
    assert!(matches!(err, syllabix_core::Error::Cancelled));
    let past1 = engine.n_past();
    assert!(past1 > 0, "turn 1 close must keep KV");

    let turn2 = short_fixture_utterance(0.5);
    let mut pcm = Vec::new();
    engine
        .generate(
            &turn2,
            &Cancel::new(),
            &mut |audio| {
                pcm.extend_from_slice(&audio.samples);
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
        .expect("turn 2 after barge");
    let turn2_text = engine.last_text().unwrap_or("").to_string();
    eprintln!("turn2 last_text={turn2_text:?}");
    assert!(engine.n_past() > past1);

    let wav = WavPcm {
        format: syllabix_native::audio::PcmFormat {
            sample_rate_hz: NATIVE_OUT_RATE as u32,
            channels: 1,
        },
        samples: pcm,
    };
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let whisper_path = cache
        .resolve(
            cache.manifest().asset("whisper-small").expect("whisper"),
            &HttpFetcher,
            &mut progress,
            &Cancel::new(),
        )
        .expect("whisper");
    let mut stt = WhisperStt::from_model_path(whisper_path, SttModel::Small).expect("whisper");
    let frames = record_fixture_to_frames(&wav).expect("reply frames");
    let transcript = stt
        .transcribe(
            &Utterance {
                turn: TurnId(2),
                frames,
            },
            &Cancel::new(),
        )
        .expect("whisper");
    let whisper = transcript.text.trim().to_string();
    let verdict = report_post_barge_reply(&turn2_text, &whisper);
    eprintln!("barge turn2 {verdict}; last_text={turn2_text:?}; whisper={whisper:?}");
    let primary = judge_barge_followup_text_delta(&turn2_text);
    assert!(
        primary != "empty" && primary != "short or unclear",
        "turn 2 last_text must be a coherent follow-up (got {primary}, {turn2_text:?})"
    );
}

#[test]
fn lfm3_audio_tiny_ctx_resets() {
    skip_unless_model!(LFM3_AUDIO_2B);
    let _native = native();
    let (model, mmproj) = resolve_lfm3_paths();
    // Raised cap inflates preflight gen_reserve in `syllabix_s2s_generate` so turn 1 + turn 2
    // cannot share KV; must stay below default so the post-reset turn still fits `n_ctx`.
    const TINY_CTX_MAX_OUT_SAMPLES: i32 = 600_000;
    let turn1 = fixture_utterance();
    let turn2 = short_fixture_utterance(0.4);
    let mut probe =
        Lfm3AudioEngine::from_paths_with_n_ctx(&model, &mmproj, N_THREADS, u32::MAX, u32::MAX)
            .expect("probe engine")
            .with_max_out_samples(CAP_OUT_SAMPLES);
    probe
        .generate(&turn1, &Cancel::new(), &mut |_| Ok(()), &mut |_, _| Ok(()))
        .expect("probe turn 1");
    let past1 = probe.n_past();
    assert!(past1 > 0);
    // llama pads `n_ctx` to 256; pick a window that fits one turn but not turn-1 KV + turn-2 preflight.
    let n_ctx = past1 as u32 + 300;
    eprintln!("lfm3-audio-2b: tiny_ctx n_ctx={n_ctx} past1={past1}");
    drop(probe);
    let mut engine =
        Lfm3AudioEngine::from_paths_with_n_ctx(&model, &mmproj, N_THREADS, u32::MAX, n_ctx)
            .expect("tiny ctx engine")
            .with_max_out_samples(TINY_CTX_MAX_OUT_SAMPLES);
    engine
        .generate(&turn1, &Cancel::new(), &mut |_| Ok(()), &mut |_, _| Ok(()))
        .expect("turn 1");
    let past1 = engine.n_past();
    assert!(past1 > 0);
    engine
        .generate(&turn2, &Cancel::new(), &mut |_| Ok(()), &mut |_, _| Ok(()))
        .expect("turn 2 after context reset");
    let past2 = engine.n_past();
    eprintln!("lfm3-audio-2b: tiny_ctx past1={past1} past2={past2} n_ctx={n_ctx}");
    assert!(
        past2 < past1,
        "context-full turn 2 must reset (past1={past1} past2={past2})"
    );
    assert!(engine.last_text().map_or(false, |t| !t.trim().is_empty()));
}

#[test]
#[ignore = "manual native: SYLLABIX_NATIVE_MODELS=lfm3-audio-2b,pocket-tts,whisper-small cargo test -p syllabix-core --features native-inference --test native_inference lfm3_audio_multi_turn_memory -- --ignored --nocapture"]
fn lfm3_audio_multi_turn_memory() {
    skip_unless_model!(LFM3_AUDIO_2B);
    skip_unless_model!("pocket-tts");
    skip_unless_model!("whisper-small");
    let _native = native();
    reset_shared_engine();

    let jfk = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stt/jfk.wav");
    let turn1 = wav_fixture_utterance(&jfk, TurnId(0));
    let mut engine = shared_engine();
    engine
        .generate(&turn1, &Cancel::new(), &mut |_| Ok(()), &mut |_, _| Ok(()))
        .expect("turn 1");
    let past1 = engine.n_past();
    let turn1_text = engine.last_text().unwrap_or("").to_string();
    eprintln!("turn1 last_text={turn1_text:?}");

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let mut tts =
        PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &cancel).expect("pocket tts");
    let question = "What did I just say? Repeat it in a few words.";
    let pcm = tts
        .synthesize_chunk(
            &TokenChunk {
                turn: TurnId(1),
                generation: GenerationId(0),
                index: 0,
                text: question.into(),
                is_last: true,
            },
            &Cancel::new(),
        )
        .expect("tts")
        .into_iter()
        .flat_map(|c| c.samples)
        .collect::<Vec<_>>();
    let wav = WavPcm {
        format: syllabix_native::audio::PcmFormat::v0(),
        samples: pcm,
    };
    let turn2 = Utterance {
        turn: TurnId(1),
        frames: record_fixture_to_frames(&wav).expect("turn2 frames"),
    };
    let mut reply_pcm = Vec::new();
    engine
        .generate(
            &turn2,
            &Cancel::new(),
            &mut |audio| {
                reply_pcm.extend_from_slice(&audio.samples);
                Ok(())
            },
            &mut |_, _| Ok(()),
        )
        .expect("turn 2");
    let past2 = engine.n_past();
    assert!(
        past2 > past1,
        "turn 2 must extend KV (past1={past1} past2={past2})"
    );
    let turn2_text = engine.last_text().unwrap_or("").to_string();
    eprintln!("turn2 last_text={turn2_text:?}");

    let out_path = std::path::Path::new("/opt/cursor/artifacts/lfm3_s2s_turn2_reply.wav");
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let reply_wav = WavPcm {
        format: syllabix_native::audio::PcmFormat {
            sample_rate_hz: NATIVE_OUT_RATE as u32,
            channels: 1,
        },
        samples: reply_pcm.clone(),
    };
    write_wav(fs::File::create(out_path).expect("create wav"), &reply_wav).expect("write wav");

    let whisper_path = cache
        .resolve(
            cache.manifest().asset("whisper-small").expect("whisper"),
            &HttpFetcher,
            &mut progress,
            &Cancel::new(),
        )
        .expect("whisper");
    let mut stt = WhisperStt::from_model_path(whisper_path, SttModel::Small).expect("whisper");
    let frames = record_fixture_to_frames(&reply_wav).expect("reply frames");
    let transcript = stt
        .transcribe(
            &Utterance {
                turn: TurnId(2),
                frames,
            },
            &Cancel::new(),
        )
        .expect("whisper");
    let combined = format!("{} {}", turn2_text, transcript.text);
    eprintln!("whisper turn2 transcript={:?}", transcript.text.trim());
    let lower = combined.to_lowercase();
    assert!(
        ["country", "americans", "ask"]
            .iter()
            .any(|w| lower.contains(w)),
        "turn 2 reply should echo JFK (got {combined:?})"
    );
}
