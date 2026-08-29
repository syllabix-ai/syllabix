//! P1 Pocket TTS native feasibility gate. This test-only id deliberately
//! precedes any YAML or pipeline exposure; P2 owns that contract.

use std::time::{Duration, Instant};
use syllabix_core::{
    Cancel, GenerationId, HttpFetcher, ModelCache, PocketTts, StderrProgress, TokenChunk, Tts,
    TurnId, POCKET_TTS_TEXT_CONDITIONER_ASSET,
};

use crate::skip_unless_model;

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
