//! Fluid Speech Pro ONNX int8 over the shared STT fixtures.

use std::fs;
use std::io::Cursor;

use sha2::{Digest, Sha256};
use syllabix_core::defaults::SttModel;
use syllabix_core::fluid_onnx::FluidOnnxStt;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::stt::{contains_words_in_order, word_match_ratio, LIBRISPEECH_MIN_WORD_MATCH};
use syllabix_core::{audio::read_wav, Cancel, ModelCache, Stt};

use crate::{hex, skip_unless_model};

fn qwen_asr_style_jfk() -> (&'static str, &'static [&'static str]) {
    (
        "jfk.wav",
        &[
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
    )
}

fn fixture_utterance(file: &str, sha256: &str) -> syllabix_core::Utterance {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(file);
    let bytes = fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(hex(Sha256::digest(&bytes)), sha256, "{file} fixture hash");
    let wav = read_wav(Cursor::new(bytes)).unwrap_or_else(|err| panic!("{file} wav: {err}"));
    let frames = syllabix_core::audio::record_fixture_to_frames(&wav)
        .unwrap_or_else(|err| panic!("{file} frames: {err}"));
    syllabix_core::Utterance {
        turn: syllabix_core::TurnId(0),
        frames,
    }
}

fn fluid_engine() -> std::sync::MutexGuard<'static, FluidOnnxStt> {
    use std::sync::Mutex;
    static ENGINE: std::sync::OnceLock<Mutex<FluidOnnxStt>> = std::sync::OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            Mutex::new(
                FluidOnnxStt::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
                    .expect("load fluid-speech-pro-onnx-int8")
                    .with_language("en")
                    .expect("en"),
            )
        })
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn fluid_onnx_transcribes_jfk() {
    skip_unless_model!("fluid-speech-pro-onnx-int8");
    let (file, expected) = qwen_asr_style_jfk();
    const SHA: &str = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e";
    let mut stt = fluid_engine();
    let utterance = fixture_utterance(file, SHA);
    let transcript = stt
        .transcribe(&utterance, &Cancel::new())
        .unwrap_or_else(|err| panic!("{file}: {err}"));
    const EXPECTED: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    assert_eq!(transcript.text, EXPECTED, "jfk transcript");
    assert!(
        contains_words_in_order(&transcript.text, expected)
            || word_match_ratio(&transcript.text, expected) >= LIBRISPEECH_MIN_WORD_MATCH,
        "text {:?}",
        transcript.text
    );
    assert_eq!(transcript.language, "en");
    assert_eq!(
        SttModel::FluidSpeechProOnnxInt8.asset_id(),
        "fluid-speech-pro-onnx-encoder"
    );
}
