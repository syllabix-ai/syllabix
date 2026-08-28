//! P1 Pocket TTS native feasibility gate. This test-only id deliberately
//! precedes any YAML or pipeline exposure; P2 owns that contract.

use std::time::Instant;
use syllabix_core::{
    Cancel, HttpFetcher, ModelCache, PocketTts, StderrProgress, POCKET_TTS_TEXT_CONDITIONER_ASSET,
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
}
