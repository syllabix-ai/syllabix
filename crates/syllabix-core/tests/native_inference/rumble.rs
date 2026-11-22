//! Rumble-2-Flash transcription through the shared Qwen3-ASR mtmd path.
//!
//! Weights test: gated on `SYLLABIX_NATIVE_MODELS=rumble-2-flash`, same as
//! the Qwen3-ASR sibling, so no model is downloaded unless explicitly
//! selected.

use std::fs;
use std::io::Cursor;

use sha2::{Digest, Sha256};
use syllabix_core::defaults::SttModel;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::qwen_asr::QwenAsrStt;
use syllabix_core::stt::{word_match_ratio, LIBRISPEECH_MIN_WORD_MATCH};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    Cancel, ModelCache, Stt, TurnId, Utterance,
};

use crate::{hex, skip_unless_model};

struct RecordedFixture {
    file: &'static str,
    sha256: &'static str,
    expected: &'static [&'static str],
}

const JFK: RecordedFixture = RecordedFixture {
    file: "jfk.wav",
    sha256: "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
    expected: &[
        "and",
        "so",
        "my",
        "fellow",
        "americans",
        "ask",
        "not",
        "what",
        "your",
        "country",
        "can",
        "do",
        "for",
        "you",
        "ask",
        "what",
        "you",
        "can",
        "do",
        "for",
        "your",
        "country",
    ],
};

fn fixture_utterance(fixture: &RecordedFixture) -> Utterance {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(fixture.file);
    let bytes = fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(
        hex(Sha256::digest(&bytes)),
        fixture.sha256,
        "{} fixture hash",
        fixture.file
    );
    let wav =
        read_wav(Cursor::new(bytes)).unwrap_or_else(|err| panic!("{} wav: {err}", fixture.file));
    let mut frames = record_fixture_to_frames(&wav)
        .unwrap_or_else(|err| panic!("{} frames: {err}", fixture.file));
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

#[test]
fn rumble_transcribes_a_recorded_english_fixture() {
    skip_unless_model!("rumble-2-flash");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut stt = QwenAsrStt::from_cache(
        &cache,
        &HttpFetcher,
        &mut progress,
        &Cancel::new(),
        SttModel::Rumble2FlashQ8,
        syllabix_core::defaults::TtsCompute::Auto,
    )
    .expect("load rumble-2-flash")
    .with_language("en")
    .expect("en");

    let transcript = stt
        .transcribe(&fixture_utterance(&JFK), &Cancel::new())
        .unwrap_or_else(|err| panic!("{}: {err}", JFK.file));
    let ratio = word_match_ratio(&transcript.text, JFK.expected);
    assert!(
        ratio >= LIBRISPEECH_MIN_WORD_MATCH,
        "{} ratio {ratio:.3} text {:?} expected {:?}",
        JFK.file,
        transcript.text,
        JFK.expected
    );
    assert_eq!(transcript.language, "en");
}
