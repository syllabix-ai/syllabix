//! whisper.cpp `small` transcription over versioned recorded fixtures.
//!
//! JFK is the whisper.cpp sample. The three LibriSpeech `test-clean` clips are
//! converted 16 kHz mono PCM from https://openslr.trmal.net/resources/12/test-clean.tar.gz
//! (OpenSLR 12, CC BY 4.0). Expected words are the official LibriSpeech
//! transcripts; those three clips pass at ≥80% in-order word match so a
//! numeral like "20" for "twenty" is allowed. JFK still requires a full match.

use std::fs;
use std::io::Cursor;
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    contains_words_in_order, run_loop, word_match_ratio, BlockedFetcher, Cancel, CollectingSink,
    FakeLlm, FakeTts, FakeVad, LoopConfig, ModelCache, PipelineStages, StderrProgress, Stt,
    SttModel, TurnId, Utterance, WhisperStt, LIBRISPEECH_MIN_WORD_MATCH,
};

use crate::{hex, native, skip_unless_model};

struct RecordedFixture {
    file: &'static str,
    sha256: &'static str,
    expected: &'static [&'static str],
    /// 1.0 requires every expected word; LibriSpeech uses 80%.
    min_word_match: f64,
}

/// SHA-256 of `tests/fixtures/stt/jfk.wav` (16 kHz mono PCM from whisper.cpp samples).
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
    min_word_match: 1.0,
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

#[test]
fn fixture_hashes_are_stable() {
    for fixture in RECORDED_FIXTURES {
        let bytes = fixture_bytes(fixture.file);
        assert_eq!(
            hex(Sha256::digest(&bytes)),
            fixture.sha256,
            "{}",
            fixture.file
        );
    }
}

#[test]
fn recorded_fixtures_match_documented_transcripts() {
    skip_unless_model!("whisper-small");
    let mut n = native();
    for fixture in RECORDED_FIXTURES {
        let utterance = fixture_utterance(fixture);
        let transcript = n
            .stt_mut()
            .transcribe(&utterance, &Cancel::new())
            .unwrap_or_else(|err| panic!("{} stt: {err}", fixture.file));
        let ratio = word_match_ratio(&transcript.text, fixture.expected);
        assert!(
            ratio >= fixture.min_word_match,
            "{} transcript {:?} matched {:.1}% of {:?} (need {:.0}%)",
            fixture.file,
            transcript.text,
            ratio * 100.0,
            fixture.expected,
            fixture.min_word_match * 100.0
        );
    }

    let cached = ModelCache::v0();
    let asset = cached.manifest().asset("whisper-small").unwrap();
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
fn whisper_replaces_fake_stt_in_the_loop() {
    skip_unless_model!("whisper-small");
    let mut n = native();
    let utterance = fixture_utterance(&JFK);
    let frames = utterance.frames.clone();
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: n.stt().clone(),
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        frames,
        Cancel::new(),
    )
    .expect("loop");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.queues.within_capacity());
    assert!(!report.turns.is_empty());
    let spoken: String = report
        .turns
        .iter()
        .map(|t| t.user_text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        contains_words_in_order(&spoken, JFK.expected),
        "pipeline transcripts {:?} did not contain the documented JFK quote",
        spoken
    );
}

#[test]
fn cancel_aborts_native_decode_and_context_stays_usable() {
    skip_unless_model!("whisper-small");
    let mut n = native();
    let utterance = fixture_utterance(&JFK);
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(40));
        cancel_thread.shutdown();
    });
    let result = n.stt_mut().transcribe(&utterance, &cancel);
    match result {
        Err(syllabix_core::Error::Cancelled) => {}
        Ok(transcript) => {
            // A very fast host may finish before abort; the adapter must still drop cleanly.
            assert!(contains_words_in_order(&transcript.text, JFK.expected));
        }
        Err(err) => panic!("unexpected STT error: {err}"),
    }
    let again = n
        .stt_mut()
        .transcribe(&utterance, &Cancel::new())
        .expect("whisper context remains usable after cancel");
    assert!(contains_words_in_order(&again.text, JFK.expected));
}

#[test]
fn auto_language_detects_english_and_pins_the_code() {
    skip_unless_model!("whisper-small");
    let cache = ModelCache::v0();
    let asset = cache.manifest().asset(SttModel::Small.asset_id()).unwrap();
    let path = cache
        .resolve(
            asset,
            &BlockedFetcher::default(),
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .expect("populated whisper-small cache");
    let mut stt = WhisperStt::from_model_path(path, SttModel::Small)
        .expect("load small for auto run")
        .with_language("auto")
        .expect("auto is a valid yaml language");
    let transcript = stt
        .transcribe(&fixture_utterance(&JFK), &Cancel::new())
        .expect("auto decode");
    assert_eq!(
        transcript.language, "en",
        "detected code must ride on the transcript"
    );
    assert!(contains_words_in_order(&transcript.text, JFK.expected));
}
