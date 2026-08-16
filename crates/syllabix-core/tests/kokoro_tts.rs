//! Merge gate for PR 12: Kokoro first-sentence audio, Markdown cleanup, loop swap.
//!
//! Intelligibility is a text → TTS → Whisper → text round-trip at ≥80%
//! in-order word match (same matcher as the LibriSpeech STT fixtures).

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use syllabix_core::{
    audio::FrameSplitter, run_loop, scripted_frames, strip_markdown_for_speech, transcript_words,
    word_match_ratio, BlockedFetcher, BuiltinDefaults, Cancel, CollectingSink, FakeLlm, FakeStt,
    FakeVad, GenerationId, HttpFetcher, KokoroTts, LoopConfig, LoopMode, ModelCache,
    PipelineStages, StderrProgress, Stt, SttModel, TokenChunk, Tts, TurnId, Utterance, WhisperStt,
    KOKORO_ASSET, KOKORO_VOICE_ASSET, TTS_ASR_MIN_WORD_MATCH, WHISPER_SMALL_ASSET,
};

fn kokoro_lock() -> std::sync::MutexGuard<'static, KokoroTts> {
    static CELL: OnceLock<Mutex<KokoroTts>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cache = ModelCache::v0();
        Mutex::new(
            KokoroTts::from_cache(
                &cache,
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
            )
            .expect("load Kokoro ONNX + af_heart"),
        )
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn token(text: &str, index: u32, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index,
        text: text.into(),
        is_last,
    }
}

fn has_energy(samples: &[i16]) -> bool {
    samples.iter().any(|s| s.abs() > 32)
}

fn whisper_lock() -> std::sync::MutexGuard<'static, WhisperStt> {
    static CELL: OnceLock<Mutex<WhisperStt>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cache = ModelCache::v0();
        let asset = cache
            .manifest()
            .asset(WHISPER_SMALL_ASSET)
            .expect("whisper-small asset");
        let path = cache
            .resolve(
                asset,
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
            )
            .expect("resolve whisper.cpp small");
        Mutex::new(WhisperStt::from_model_path(path, SttModel::Small).expect("load whisper small"))
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn pcm_to_utterance(samples: &[i16]) -> Utterance {
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(samples).expect("frame split");
    frames.extend(splitter.flush().expect("frame flush"));
    assert!(
        !frames.is_empty(),
        "TTS PCM must fill at least one 16 kHz frame"
    );
    Utterance {
        turn: TurnId(0),
        frames,
    }
}

#[test]
fn markdown_fixtures_remove_headings_lists_and_emphasis() {
    let spoken = strip_markdown_for_speech(
        "## Hello\n- **world**\n1. _list_\nDo **not** read this [link](https://x.test) aloud.",
    );
    assert!(!spoken.contains('#'), "{spoken}");
    assert!(!spoken.contains('*'), "{spoken}");
    assert!(!spoken.contains('_'), "{spoken}");
    assert!(!spoken.contains("https://"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("hello"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("world"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("list"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("link"), "{spoken}");
}

#[test]
fn first_sentence_audio_arrives_before_full_completion() {
    let _guard = kokoro_lock();
    let mut tts = _guard.clone();
    assert_eq!(tts.name(), "kokoro");
    let first = tts
        .synthesize_chunk(&token("Hello world. ", 0, false), &Cancel::new())
        .expect("first sentence");
    assert_eq!(first.len(), 1, "first sentence must emit before is_last");
    assert!(!first[0].is_last);
    assert_eq!(first[0].turn, TurnId(0));
    assert!(
        first[0].samples.len() >= 1_000,
        "playable PCM should be more than a few samples, got {}",
        first[0].samples.len()
    );
    assert!(
        has_energy(&first[0].samples),
        "first-sentence audio must be non-silent"
    );

    let rest = tts
        .synthesize_chunk(&token("More later.", 1, true), &Cancel::new())
        .expect("remainder");
    assert_eq!(rest.len(), 1);
    assert!(rest[0].is_last);
    assert!(has_energy(&rest[0].samples));
}

/// Spoken English in, Whisper `small` out. Extra ASR words are allowed; missed
/// reference words count against the 80% in-order ratio.
#[test]
fn spoken_text_round_trips_through_whisper_at_eighty_percent() {
    const TEXT: &str = "The children played outside in the garden after lunch.";
    let expected_owned = transcript_words(TEXT);
    let expected: Vec<&str> = expected_owned.iter().map(String::as_str).collect();
    assert!(
        expected.len() >= 8,
        "fixture needs enough words for an 80% gate, got {expected:?}"
    );

    let pcm = {
        let _tts_guard = kokoro_lock();
        let mut tts = _tts_guard.clone();
        let chunks = tts
            .synthesize_chunk(&token(TEXT, 0, true), &Cancel::new())
            .expect("kokoro synthesize");
        assert!(!chunks.is_empty());
        let mut pcm = Vec::new();
        for chunk in &chunks {
            assert!(
                has_energy(&chunk.samples),
                "round-trip audio must be voiced"
            );
            pcm.extend_from_slice(&chunk.samples);
        }
        pcm
    };

    let mut stt = whisper_lock();
    let transcript = stt
        .transcribe(&pcm_to_utterance(&pcm), &Cancel::new())
        .expect("whisper transcribe TTS audio");
    let ratio = word_match_ratio(&transcript.text, &expected);
    assert!(
        ratio >= TTS_ASR_MIN_WORD_MATCH,
        "TTS→ASR {:?} matched {:.1}% of {:?} (need {:.0}%)",
        transcript.text,
        ratio * 100.0,
        expected,
        TTS_ASR_MIN_WORD_MATCH * 100.0
    );
}

#[test]
fn kokoro_replaces_fake_tts_in_the_loop() {
    let _guard = kokoro_lock();
    let tts = _guard.clone();
    let frames = scripted_frames(1, 2, 1);
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: tts.clone(),
            sink: CollectingSink::default(),
        },
        frames,
        Cancel::new(),
    )
    .expect("loop");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.queues.within_capacity());
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "turn-000");
    assert_eq!(report.turns[0].assistant_text, "echo:turn-000");
    assert!(
        report.turns[0].audio_chunks > 0,
        "real Kokoro must play at least one audio chunk"
    );
    assert_ne!(
        report.turns[0].audio_chunks, report.turns[0].token_count,
        "Kokoro buffers tokens into sentences; must not still be FakeTts 1:1 chunks"
    );
}

#[test]
fn populated_cache_reuses_kokoro_offline() {
    let _tts = kokoro_lock();
    let cached = ModelCache::v0();
    for id in [KOKORO_ASSET, KOKORO_VOICE_ASSET] {
        let asset = cached.manifest().asset(id).unwrap();
        let path: PathBuf = cached
            .resolve(
                asset,
                &BlockedFetcher::default(),
                &mut StderrProgress::new(),
                &Cancel::new(),
            )
            .expect("populated cache must not need the network");
        assert!(path.exists());
    }
}
