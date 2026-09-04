//! Silero classification over versioned silence, speech, and noise fixtures.

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    BlockedFetcher, Cancel, HttpFetcher, ModelCache, SileroVad, StderrProgress, Vad, VadEvent,
    END_SILENCE_FRAMES, MIN_SPEECH, MIN_SPEECH_FRAMES,
};

struct VadFixture {
    file: &'static str,
    sha256: &'static str,
}

const SPEECH: VadFixture = VadFixture {
    file: "speech.wav",
    sha256: "e72f1ccf42dc827252141e927a0969793169fbe8039392e207285aa306f09daa",
};
const SILENCE: VadFixture = VadFixture {
    file: "silence.wav",
    sha256: "643f8a8dc8bd9c19225afffad2becfec5426180b3749cb208abdf1a6c8354efc",
};
const NOISE: VadFixture = VadFixture {
    file: "noise.wav",
    sha256: "68a384b67befcdcdf53a3da1d52960062554684b5d9ded02b933fc6041ae454c",
};

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

fn fixture_bytes(file: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vad")
        .join(file);
    fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn fixture_frames(fix: &VadFixture) -> Vec<syllabix_core::AudioFrame> {
    let bytes = fixture_bytes(fix.file);
    assert_eq!(hex(Sha256::digest(&bytes)), fix.sha256, "{} hash", fix.file);
    let wav = read_wav(Cursor::new(bytes)).expect("wav");
    record_fixture_to_frames(&wav).expect("frames")
}

fn silero_lock() -> std::sync::MutexGuard<'static, SileroVad> {
    static CELL: OnceLock<Mutex<SileroVad>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cache = ModelCache::v0();
        Mutex::new(
            SileroVad::from_cache(
                &cache,
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
            )
            .expect("load Silero ONNX"),
        )
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn collect_events(vad: &mut SileroVad, frames: Vec<syllabix_core::AudioFrame>) -> Vec<VadEvent> {
    vad.reset();
    let mut events = Vec::new();
    for frame in frames {
        events.extend(vad.push_frame(frame).expect("silero"));
    }
    events.extend(vad.flush().expect("flush"));
    events
}

fn collect_ends(vad: &mut SileroVad, frames: Vec<syllabix_core::AudioFrame>) -> usize {
    collect_events(vad, frames)
        .into_iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnd { .. }))
        .count()
}

#[test]
fn fixture_hashes_are_stable() {
    for fix in [SPEECH, SILENCE, NOISE] {
        let bytes = fixture_bytes(fix.file);
        assert_eq!(hex(Sha256::digest(&bytes)), fix.sha256, "{}", fix.file);
    }
}

#[test]
fn speech_fixture_emits_one_utterance() {
    let mut vad = silero_lock();
    let ends = collect_ends(&mut vad, fixture_frames(&SPEECH));
    assert_eq!(ends, 1, "speech.wav must be one Silero turn");
}

#[test]
fn silence_fixture_emits_no_turn() {
    let mut vad = silero_lock();
    let ends = collect_ends(&mut vad, fixture_frames(&SILENCE));
    assert_eq!(ends, 0, "silence.wav must not start a turn");
}

#[test]
fn noise_fixture_emits_no_turn() {
    let mut vad = silero_lock();
    let ends = collect_ends(&mut vad, fixture_frames(&NOISE));
    assert_eq!(ends, 0, "noise.wav must not start a turn");
}

#[test]
fn populated_cache_reuses_silero_offline() {
    let _vad = silero_lock();
    let cached = ModelCache::v0();
    let asset = cached.manifest().asset("silero").unwrap();
    cached
        .resolve(
            asset,
            &BlockedFetcher::default(),
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .expect("populated cache must not need the network");
}

#[test]
fn turn_timing_matches_launch_params() {
    assert_eq!(MIN_SPEECH, std::time::Duration::from_millis(100));
    assert_eq!(END_SILENCE_FRAMES, 11);
    assert_eq!(MIN_SPEECH_FRAMES, 4);
}

#[test]
fn speech_fixture_has_speech_frames() {
    let mut vad = silero_lock();
    vad.reset();
    let mut above = 0_u32;
    for frame in fixture_frames(&SPEECH) {
        if vad.debug_probability(&frame).expect("score") >= 0.5 {
            above += 1;
        }
    }
    assert!(
        above >= MIN_SPEECH_FRAMES as u32,
        "8 kHz Silero scored {above} speech frames on speech.wav (need >={MIN_SPEECH_FRAMES})"
    );
}
