//! Kokoro first-sentence audio, provider swapping, and TTS-to-ASR round trips.
//!
//! Opt-in via `SYLLABIX_NATIVE_MODELS=kokoro`. The shared `native().tts()` slot
//! follows the launch default (`AgentConfig::v0`); this suite loads Kokoro
//! directly.
//!
//! Intelligibility is a text → TTS → Whisper → text round-trip at ≥80%
//! in-order word match (same matcher as the LibriSpeech STT fixtures). Markdown
//! cleanup is a unit test in `kokoro_tts.rs` so coverage still sees it.

use std::time::Instant;

use syllabix_core::{
    audio::FrameSplitter, run_loop, scripted_frames, transcript_words, word_match_ratio, Cancel,
    CollectingSink, FakeLlm, FakeStt, FakeVad, GenerationId, HttpFetcher, KokoroTts, LoopConfig,
    ModelCache, PipelineStages, StderrProgress, Stt, TokenChunk, Tts, TurnId, Utterance,
    KOKORO_ASSET, KOKORO_VOICE_ASSET, TTS_ASR_MIN_WORD_MATCH,
};

use crate::{native, native_latency_enabled, skip_unless_model, TTS_LATENCY_SENTENCES};

fn load_kokoro() -> KokoroTts {
    KokoroTts::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Kokoro ONNX")
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
fn first_sentence_audio_arrives_before_full_completion() {
    skip_unless_model!("kokoro");
    let mut tts = load_kokoro();
    // `name()` describes where inference runs; the engine identity stays
    // in `model_id()`.
    assert_eq!(tts.name(), "local");
    assert_eq!(tts.model_id(), Some("kokoro"));
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
    skip_unless_model!("kokoro");
    const TEXT: &str = "The children played outside in the garden after lunch.";
    let expected_owned = transcript_words(TEXT);
    let expected: Vec<&str> = expected_owned.iter().map(String::as_str).collect();
    assert!(
        expected.len() >= 8,
        "fixture needs enough words for an 80% gate, got {expected:?}"
    );

    let mut n = native();
    let mut tts = load_kokoro();
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

    let transcript = n
        .stt_mut()
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
    skip_unless_model!("kokoro");
    let frames = scripted_frames(1, 2, 1);
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: load_kokoro(),
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
    skip_unless_model!("kokoro");
    let _ = load_kokoro();
    let cached = ModelCache::v0();
    for id in [KOKORO_ASSET, KOKORO_VOICE_ASSET] {
        let asset = cached.manifest().asset(id).unwrap();
        let path = cached
            .require_cached(asset)
            .expect("populated cache must not need the network");
        assert!(path.exists());
    }
}

/// Reproducible Kokoro latency baseline. First PCM is observed at the streaming
/// callback, so it remains comparable with Pocket and Qwen captures.
#[test]
fn kokoro_latency_capture() {
    if !native_latency_enabled() || !crate::native_model_selected("kokoro") {
        return;
    }
    let mut tts = load_kokoro();
    let mut ttfb_ms = Vec::new();
    let mut rtf = Vec::new();
    for text in TTS_LATENCY_SENTENCES {
        let started = Instant::now();
        let mut first_pcm = None;
        let mut samples = 0usize;
        tts.synthesize_chunk_into(&token(text, 0, true), &Cancel::new(), &mut |audio| {
            first_pcm.get_or_insert_with(|| started.elapsed());
            samples += audio.samples.len();
            Ok(())
        })
        .expect("Kokoro latency synthesis");
        let elapsed = started.elapsed().as_secs_f64();
        assert!(samples > 0, "Kokoro latency sample must be voiced");
        ttfb_ms.push(first_pcm.expect("first PCM").as_secs_f64() * 1_000.0);
        rtf.push(elapsed / (samples as f64 / 16_000.0));
    }
    let percentile = |mut values: Vec<f64>, p: f64| {
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite metric"));
        let rank = p / 100.0 * (values.len() - 1) as f64;
        let low = rank.floor() as usize;
        let high = rank.ceil() as usize;
        values[low] * (high as f64 - rank) + values[high] * (rank - low as f64)
    };
    println!(
        "tts latency [kokoro]: n={} ttfb_ms p50={:.0} p95={:.0}; rtf p50={:.2} p95={:.2}",
        ttfb_ms.len(),
        percentile(ttfb_ms.clone(), 50.0),
        percentile(ttfb_ms, 95.0),
        percentile(rtf.clone(), 50.0),
        percentile(rtf, 95.0),
    );
}
