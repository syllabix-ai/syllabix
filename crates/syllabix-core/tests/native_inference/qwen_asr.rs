//! Qwen3-ASR 0.6B transcription over the same STT fixtures as Whisper.

use std::fs;
use std::io::Cursor;
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    contains_words_in_order, word_match_ratio, Cancel, HttpFetcher, ModelCache, QwenAsrStt,
    StderrProgress, Stt, SttModel, TurnId, Utterance, LIBRISPEECH_MIN_WORD_MATCH,
};

use crate::{hex, skip_unless_model};

struct RecordedFixture {
    file: &'static str,
    sha256: &'static str,
    expected: &'static [&'static str],
    min_word_match: f64,
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
    min_word_match: LIBRISPEECH_MIN_WORD_MATCH,
};

const LIBRISPEECH_1089: RecordedFixture = RecordedFixture {
    file: "librispeech-1089-134686-0000.wav",
    sha256: "c6517d6052651b2ff22d19526f49fc7da5fc56de30790b9ed54238abd9c48c36",
    expected: &[
        "he", "hoped", "there", "would", "be", "stew", "for", "dinner", "turnips", "and",
        "carrots", "and", "bruised", "potatoes", "and", "fat", "mutton", "pieces", "to", "be",
        "ladled", "out", "in", "thick", "peppered", "flour", "fattened", "sauce",
    ],
    min_word_match: LIBRISPEECH_MIN_WORD_MATCH,
};

const LIBRISPEECH_121: RecordedFixture = RecordedFixture {
    file: "librispeech-121-127105-0009.wav",
    sha256: "d17c23c38c823c3f3507f0d3a832e476c1c1bfd32de35f3068db5fe4eae05bad",
    expected: &["she", "has", "been", "dead", "these", "twenty", "years"],
    min_word_match: LIBRISPEECH_MIN_WORD_MATCH,
};

const LIBRISPEECH_1995: RecordedFixture = RecordedFixture {
    file: "librispeech-1995-1837-0005.wav",
    sha256: "598278504ac5ce8b2e9bde93cbfe8ca0514b4678150b265a12cefcee32f9b452",
    expected: &[
        "she", "was", "so", "strange", "and", "human", "a", "creature",
    ],
    min_word_match: LIBRISPEECH_MIN_WORD_MATCH,
};

const RECORDED_FIXTURES: &[RecordedFixture] =
    &[JFK, LIBRISPEECH_1089, LIBRISPEECH_121, LIBRISPEECH_1995];

fn fixture_bytes(file: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stt")
        .join(file);
    fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn fixture_utterance(fixture: &RecordedFixture) -> Utterance {
    let bytes = fixture_bytes(fixture.file);
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

/// One shared engine for this binary so parallel tests do not race the
/// model cache download and so ggml stays a single context.
fn qwen_asr() -> QwenAsrStt {
    static ENGINE: std::sync::OnceLock<QwenAsrStt> = std::sync::OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            QwenAsrStt::from_cache(
                &cache,
                &HttpFetcher,
                &mut progress,
                &Cancel::new(),
                SttModel::QwenAsr06,
                syllabix_core::TtsCompute::Auto,
            )
            .expect("load qwen3-asr-0.6")
            .with_language("en")
            .expect("en")
        })
        .clone()
}

#[test]
fn qwen_asr_transcribes_recorded_english_fixtures() {
    skip_unless_model!("qwen3-asr-0.6");
    let mut stt = qwen_asr();
    for fixture in RECORDED_FIXTURES {
        let transcript = stt
            .transcribe(&fixture_utterance(fixture), &Cancel::new())
            .unwrap_or_else(|err| panic!("{}: {err}", fixture.file));
        let ratio = word_match_ratio(&transcript.text, fixture.expected);
        assert!(
            ratio >= fixture.min_word_match,
            "{} ratio {ratio:.3} text {:?} expected {:?}",
            fixture.file,
            transcript.text,
            fixture.expected
        );
        assert_eq!(transcript.language, "en");
    }
}

#[test]
fn qwen_asr_cancel_aborts_and_context_stays_usable() {
    skip_unless_model!("qwen3-asr-0.6");
    let mut stt = qwen_asr();
    let utterance = fixture_utterance(&JFK);
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(40));
        cancel_thread.shutdown();
    });
    let result = stt.transcribe(&utterance, &cancel);
    match result {
        Err(syllabix_core::Error::Cancelled) => {}
        Ok(transcript) => {
            assert!(
                contains_words_in_order(&transcript.text, JFK.expected)
                    || word_match_ratio(&transcript.text, JFK.expected)
                        >= LIBRISPEECH_MIN_WORD_MATCH
            );
        }
        Err(err) => panic!("unexpected STT error: {err}"),
    }
    let again = stt
        .transcribe(&utterance, &Cancel::new())
        .expect("qwen asr context remains usable after cancel");
    assert!(
        word_match_ratio(&again.text, JFK.expected) >= LIBRISPEECH_MIN_WORD_MATCH,
        "after cancel: {:?}",
        again.text
    );
}
