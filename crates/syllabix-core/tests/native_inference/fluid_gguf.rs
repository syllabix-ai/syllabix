//! Fluid Speech Pro GGUF transcription over the same STT fixtures as Whisper.

use std::fs;
use std::io::Cursor;

use sha2::{Digest, Sha256};
use syllabix_core::defaults::SttModel;
use syllabix_core::fluid_gguf::FluidGgufStt;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::stt::{contains_words_in_order, word_match_ratio, LIBRISPEECH_MIN_WORD_MATCH};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    Cancel, ModelCache, Stt, TurnId, Utterance,
};

use crate::{hex, skip_unless_model};

const JFK_EXPECTED: &[&str] = &[
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
];

fn fixture_bytes(file: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(file);
    fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn jfk_utterance() -> Utterance {
    const SHA: &str = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e";
    let bytes = fixture_bytes("jfk.wav");
    assert_eq!(hex(Sha256::digest(&bytes)), SHA);
    let wav = read_wav(Cursor::new(bytes)).expect("jfk wav");
    let mut frames = record_fixture_to_frames(&wav).expect("jfk frames");
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

fn fluid_engine() -> FluidGgufStt {
    static ENGINE: std::sync::OnceLock<FluidGgufStt> = std::sync::OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            FluidGgufStt::from_cache(
                &cache,
                &HttpFetcher,
                &mut progress,
                &Cancel::new(),
                SttModel::FluidSpeechProGgufQ8,
            )
            .expect("load fluid-speech-pro-gguf-q8_0")
        })
        .clone()
}

#[test]
fn fluid_gguf_transcribes_jfk() {
    skip_unless_model!("fluid-speech-pro-gguf-q8_0");
    let mut stt = fluid_engine();
    let transcript = stt
        .transcribe(&jfk_utterance(), &Cancel::new())
        .expect("jfk transcription");
    let ratio = word_match_ratio(&transcript.text, JFK_EXPECTED);
    assert!(
        ratio >= LIBRISPEECH_MIN_WORD_MATCH
            || contains_words_in_order(&transcript.text, JFK_EXPECTED),
        "jfk ratio {ratio:.3} text {:?}",
        transcript.text
    );
}
