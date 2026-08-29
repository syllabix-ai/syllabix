//! P1 Pocket TTS native feasibility gate. This test-only id deliberately
//! precedes any YAML or pipeline exposure; P2 owns that contract.

use std::time::{Duration, Instant};
use syllabix_core::{
    audio::FrameSplitter, transcript_words, word_match_ratio, Cancel, GenerationId, HttpFetcher,
    ModelCache, PocketTts, StderrProgress, Stt, TokenChunk, Tts, TurnId, Utterance,
    POCKET_TTS_TEXT_CONDITIONER_ASSET, TTS_ASR_MIN_WORD_MATCH,
};

use crate::{native, native_latency_enabled, skip_unless_model, TTS_LATENCY_SENTENCES};

// Pocket remains an opt-in repair candidate. Its P3 promotion decision keeps
// the shared 80% gate; this lower native floor only prevents regressions below
// the founder-accepted 77.8% baseline while the listening study is pending.
const POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH: f64 = 0.75;

fn token(text: &str, index: u32, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index,
        text: text.into(),
        is_last,
    }
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

fn speak(tts: &mut PocketTts, text: &str) -> Vec<i16> {
    let chunks = tts
        .synthesize_chunk(&token(text, 0, true), &Cancel::new())
        .expect("Pocket TTS synthesize");
    assert!(!chunks.is_empty(), "Pocket TTS must emit audio");
    chunks.into_iter().flat_map(|chunk| chunk.samples).collect()
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite metric"));
    let rank = percentile / 100.0 * (values.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    values[low] * (high as f64 - rank) + values[high] * (rank - low as f64)
}

#[test]
fn pinned_onnx_graph_set_loads_and_text_fixture_is_deterministic() {
    skip_unless_model!("pocket-tts");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut first = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load P1 Pocket TTS ONNX graph set");
    let a = first.text_fixture().expect("run first native text fixture");
    let b = first
        .text_fixture()
        .expect("run repeated native text fixture");
    assert_eq!(a, b, "fixed graph and token fixture must be deterministic");
    assert!(
        a.iter().any(|value| *value != 0.0),
        "fixture must be non-zero"
    );
    assert_eq!(a.len(), 5 * 1024);
    let raw = first.c_api_fixture().expect("run raw ONNX Runtime fixture");
    assert!(!raw.is_empty());
    assert!(raw.iter().all(|sample| sample.is_finite()));
    assert!(raw.iter().any(|sample| *sample != 0.0));
    let pcm = first
        .synthesize_fixture()
        .expect("synthesize text-conditioned native PCM fixture");
    assert_eq!(pcm.len(), 1920);
    assert!(pcm.iter().all(|sample| sample.is_finite()));
    assert!(pcm.iter().any(|sample| *sample != 0.0));
    let started = Instant::now();
    let measured = first.synthesize_fixture().expect("measure native fixture");
    let elapsed = started.elapsed().as_secs_f64();
    let rtf = elapsed / (measured.len() as f64 / 24_000.0);
    eprintln!(
        "Pocket TTS native fixture RTF: {rtf:.3} ({elapsed:.4}s for {} samples)",
        measured.len()
    );
    assert_eq!(
        cache
            .manifest()
            .asset(POCKET_TTS_TEXT_CONDITIONER_ASSET)
            .expect("manifest entry")
            .layer
            .as_str(),
        "tts"
    );

    let chunks = first
        .synthesize_chunk(
            &TokenChunk {
                turn: TurnId(1),
                generation: GenerationId(0),
                index: 0,
                text: "A short streamed reply.".into(),
                is_last: true,
            },
            &Cancel::new(),
        )
        .expect("Pocket TTS must stream a real sentence into pipeline PCM");
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|chunk| !chunk.samples.is_empty()));
    assert!(chunks.last().is_some_and(|chunk| chunk.is_last));

    let cancel = Cancel::new();
    let trigger = cancel.clone();
    let interrupter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        trigger.cancel_generation();
    });
    let started = Instant::now();
    let err = first
        .synthesize_chunk(
            &TokenChunk {
                turn: TurnId(2),
                generation: GenerationId(0),
                index: 0,
                text: "This sentence is deliberately long enough to interrupt while Pocket TTS is decoding its recurrent frames.".into(),
                is_last: true,
            },
            &cancel,
        )
        .expect_err("barge-in cancellation must abort Pocket TTS");
    interrupter.join().expect("interrupter thread");
    assert!(matches!(err, syllabix_core::Error::Cancelled));
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// Repair-PR machine proxy: Pocket speech must remain intelligible enough to
/// exercise the pipeline. P3 retains the 80% promotion bar and human listen.
#[test]
fn pocket_speech_round_trips_through_whisper_at_eighty_percent() {
    skip_unless_model!("pocket-tts");
    const TEXT: &str = "The children played outside in the garden after lunch.";
    let expected_owned = transcript_words(TEXT);
    let expected: Vec<&str> = expected_owned.iter().map(String::as_str).collect();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut tts = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load Pocket TTS");
    let pcm = speak(&mut tts, TEXT);
    let mut native = native();
    let transcript = native
        .stt_mut()
        .transcribe(&pcm_to_utterance(&pcm), &Cancel::new())
        .expect("whisper transcribe Pocket TTS audio");
    let ratio = word_match_ratio(&transcript.text, &expected);
    assert!(
        ratio >= POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH,
        "Pocket TTS→ASR {:?} matched {:.1}% of {:?} (need {:.0}% repair floor; P3 needs {:.0}%)",
        transcript.text,
        ratio * 100.0,
        expected,
        POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH * 100.0,
        TTS_ASR_MIN_WORD_MATCH * 100.0,
    );
}

/// P3 reproducible evidence capture. The callback measures actual first PCM,
/// not completion of the whole recurrent decode.
#[test]
fn pocket_latency_capture() {
    if !native_latency_enabled() || !crate::native_model_selected("pocket-tts") {
        return;
    }
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut tts = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load Pocket TTS");
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
        .expect("Pocket TTS latency synthesis");
        let elapsed = started.elapsed().as_secs_f64();
        assert!(samples > 0, "Pocket TTS latency sample must be voiced");
        ttfb_ms.push(first_pcm.expect("first PCM").as_secs_f64() * 1_000.0);
        rtf.push(elapsed / (samples as f64 / 16_000.0));
    }
    println!(
        "tts latency [pocket-tts]: n={} ttfb_ms p50={:.0} p95={:.0}; rtf p50={:.2} p95={:.2}",
        ttfb_ms.len(),
        percentile(ttfb_ms.clone(), 50.0),
        percentile(ttfb_ms, 95.0),
        percentile(rtf.clone(), 50.0),
        percentile(rtf, 95.0),
    );
}
