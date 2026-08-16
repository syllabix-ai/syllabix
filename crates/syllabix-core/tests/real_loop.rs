//! PR 13 merge gate: six real-provider turns, queue bounds, shutdown, RSS ceiling.

use std::fs;
use std::io::Cursor;
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, DrainingPlayback, FixtureCapture, PcmFormat},
    load_real_providers, process_rss_bytes, run_loop_captured, BuiltinDefaults, Cancel,
    HttpFetcher, LoopConfig, LoopMode, ModelCache, PipelineStages, StderrProgress,
    END_SILENCE_FRAMES, FRAME_SAMPLES, LOOP_RSS_GROWTH_CEILING_BYTES,
};

const SPEECH_SHA256: &str = "e72f1ccf42dc827252141e927a0969793169fbe8039392e207285aa306f09daa";
const REAL_TURNS: usize = 6;

fn hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn speech_frames() -> Vec<syllabix_core::AudioFrame> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vad/speech.wav");
    let bytes = fs::read(&path).expect("speech.wav");
    assert_eq!(hex(Sha256::digest(&bytes)), SPEECH_SHA256);
    let wav = read_wav(Cursor::new(bytes)).expect("wav");
    record_fixture_to_frames(&wav).expect("frames")
}

fn silence_frame(seq: u64) -> syllabix_core::AudioFrame {
    syllabix_core::AudioFrame::new(
        seq,
        syllabix_core::DEFAULT_SAMPLE_RATE_HZ,
        syllabix_core::DEFAULT_CHANNELS,
        vec![0; FRAME_SAMPLES],
    )
    .expect("silence")
}

fn six_turn_frames() -> Vec<syllabix_core::AudioFrame> {
    let speech = speech_frames();
    let mut out = Vec::new();
    let mut seq = 0_u64;
    let pad = END_SILENCE_FRAMES + 2;
    for _ in 0..REAL_TURNS {
        for mut frame in speech.clone() {
            frame.seq = seq;
            seq += 1;
            out.push(frame);
        }
        for _ in 0..pad {
            out.push(silence_frame(seq));
            seq += 1;
        }
    }
    out
}

fn real_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn load_stages() -> (
    syllabix_core::SileroVad,
    syllabix_core::WhisperStt,
    syllabix_core::LlamaLlm,
    syllabix_core::KokoroTts,
) {
    static CELL: OnceLock<(
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
        std::path::PathBuf,
    )> = OnceLock::new();
    let _ = CELL.get_or_init(|| {
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let cancel = Cancel::new();
        let silero = cache.manifest().asset("silero").unwrap();
        let whisper = cache.manifest().asset("whisper-small").unwrap();
        let llama = cache.manifest().asset("llama-3.2-1b").unwrap();
        let kokoro = cache.manifest().asset("kokoro").unwrap();
        let voice = cache.manifest().asset("kokoro-voice").unwrap();
        (
            cache
                .resolve(silero, &HttpFetcher, &mut progress, &cancel)
                .expect("silero"),
            cache
                .resolve(whisper, &HttpFetcher, &mut progress, &cancel)
                .expect("whisper"),
            cache
                .resolve(llama, &HttpFetcher, &mut progress, &cancel)
                .expect("llama"),
            cache
                .resolve(kokoro, &HttpFetcher, &mut progress, &cancel)
                .expect("kokoro"),
            cache
                .resolve(voice, &HttpFetcher, &mut progress, &cancel)
                .expect("voice"),
        )
    });
    let cache = ModelCache::v0();
    load_real_providers(
        &cache,
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load real providers")
}

#[test]
fn six_real_turns_complete_within_queue_and_memory_bounds() {
    let _guard = real_lock();
    let (vad, stt, llm, tts) = load_stages();
    let warmup_rss = process_rss_bytes();

    let sink = DrainingPlayback::new(PcmFormat {
        sample_rate_hz: 48_000,
        channels: 2,
    })
    .expect("playback converter");
    let drain = sink.stats();
    let capture = FixtureCapture::from_frames(six_turn_frames());

    let report = run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
        },
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
        "expected {REAL_TURNS} real turns, got {:?}",
        report
            .turns
            .iter()
            .map(|t| (t.id, t.user_text.clone(), t.assistant_text.clone()))
            .collect::<Vec<_>>()
    );
    for (i, turn) in report.turns.iter().enumerate() {
        assert_eq!(turn.id, syllabix_core::TurnId(i as u64));
        assert!(
            !turn.user_text.trim().is_empty(),
            "turn {i} missing STT text"
        );
        assert!(
            !turn.assistant_text.trim().is_empty(),
            "turn {i} missing LLM text"
        );
        assert_ne!(
            turn.assistant_text,
            format!("echo:{}", turn.user_text),
            "must not still be the fake echo LLM"
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

#[test]
fn six_turns_config_matches_native_gate() {
    assert_eq!(LoopConfig::six_turns().mode, LoopMode::StopAfterTurns(6));
}
