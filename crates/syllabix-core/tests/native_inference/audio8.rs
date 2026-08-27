//! A1 Audio8 native feasibility gate. It loads only the SHA-pinned ONNX
//! runtime assets and proves bounded, deterministic 44.1 kHz PCM without a
//! Python runtime, subprocess, service, or voice-registration encoder.

use std::time::Instant;

use sha2::{Digest, Sha256};
use syllabix_core::{
    process_rss_bytes, Audio8Native, Cancel, HttpFetcher, ModelCache, StderrProgress,
    AUDIO8_MAX_FRAMES, AUDIO8_SAMPLE_RATE_HZ,
};

use crate::skip_unless_model;

fn audio8() -> Audio8Native {
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    Audio8Native::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load pinned Audio8 SlowAR/FastAR/codec runtime")
}

fn pcm_hash(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for sample in samples {
        hash.update(sample.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn audio8_loads_and_emits_deterministic_native_pcm() {
    skip_unless_model!("audio8");
    let load_started = Instant::now();
    let engine = audio8();
    let load_elapsed = load_started.elapsed();
    let first_started = Instant::now();
    let first = engine
        .synthesize_greedy(
            "This is the Audio8 native feasibility fixture.",
            16,
            &Cancel::new(),
        )
        .expect("Audio8 first deterministic fixture");
    let first_elapsed = first_started.elapsed();
    let second_started = Instant::now();
    let second = engine
        .synthesize_greedy(
            "This is the Audio8 native feasibility fixture.",
            16,
            &Cancel::new(),
        )
        .expect("Audio8 second deterministic fixture");
    let second_elapsed = second_started.elapsed();
    assert!(!first.is_empty(), "Audio8 must emit PCM");
    assert!(first.iter().all(|sample| sample.is_finite()));
    assert_eq!(first, second, "greedy fixture PCM must reproduce exactly");
    assert_eq!(AUDIO8_SAMPLE_RATE_HZ, 44_100);
    const _: () = assert!(AUDIO8_MAX_FRAMES <= 256);
    let audio_seconds = first.len() as f64 / f64::from(AUDIO8_SAMPLE_RATE_HZ);
    let first_rtf = first_elapsed.as_secs_f64() / audio_seconds;
    let second_rtf = second_elapsed.as_secs_f64() / audio_seconds;
    eprintln!(
        "audio8 fixture: {} samples, {} Hz, sha256={}, load_ms={}, first_ms={}, second_ms={}, first_rtf={:.3}, second_rtf={:.3}, rss_bytes={:?}",
        first.len(),
        AUDIO8_SAMPLE_RATE_HZ,
        pcm_hash(&first),
        load_elapsed.as_millis(),
        first_elapsed.as_millis(),
        second_elapsed.as_millis(),
        first_rtf,
        second_rtf,
        process_rss_bytes(),
    );
}
