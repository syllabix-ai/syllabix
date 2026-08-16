//! PR 13 merge gate: six real-provider turns, queue bounds, shutdown, RSS ceiling.

use std::fs;
use std::io::Cursor;

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, DrainingPlayback, FixtureCapture, PcmFormat},
    process_rss_bytes, run_loop_captured, Cancel, HttpFetcher, LoopConfig, ModelCache,
    PipelineStages, SileroVad, StderrProgress, END_SILENCE_FRAMES, FRAME_SAMPLES,
    LOOP_RSS_GROWTH_CEILING_BYTES,
};

use crate::hex;
use crate::native;

const SPEECH_SHA256: &str = "e72f1ccf42dc827252141e927a0969793169fbe8039392e207285aa306f09daa";
const REAL_TURNS: usize = 6;

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

#[test]
fn six_real_turns_complete_within_queue_and_memory_bounds() {
    let n = native();
    let warmup_rss = process_rss_bytes();
    // Fresh Silero session: GRU hangover on the shared instance would stall later turns.
    let vad = SileroVad::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Silero for six-turn");

    let sink = DrainingPlayback::new(PcmFormat {
        sample_rate_hz: 48_000,
        channels: 2,
    })
    .expect("playback converter");
    let drain = sink.stats();
    let capture = FixtureCapture::from_frames(six_turn_frames());

    let report = run_loop_captured(
        LoopConfig::default(),
        PipelineStages {
            vad,
            stt: n.stt.clone(),
            llm: n.llm.clone(),
            tts: n.tts.clone(),
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
