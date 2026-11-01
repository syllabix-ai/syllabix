//! Native inference tests for LiquidAI LFM3-1.7B Q8_0 (launch default):
//!
//! ```bash
//! SYLLABIX_NATIVE_MODELS=lfm3-1.7b cargo test -p syllabix-core --features native-inference --test native_inference lfm3
//! ```

use syllabix_core::llm::LFM3_1_7B_ASSET;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::{Cancel, Llm, ModelCache};
use syllabix_native::LlamaContext;

use crate::{native, skip_unless_model};

#[test]
fn q8_0_gguf_loads_co_resident_without_segfault() {
    skip_unless_model!(LFM3_1_7B_ASSET);
    let mut native = native();
    native.stt();
    native.tts();
    let llm = native.llm_mut();
    assert_eq!(llm.name(), "local");
    let window = llm.context_window().expect("native lfm3 context");
    assert!(window.0 > 0 && window.0 == window.1, "n_ctx={window:?}");
}

#[test]
fn q8_0_uses_gguf_sampling_metadata() {
    skip_unless_model!(LFM3_1_7B_ASSET);
    let cache = ModelCache::v0();
    let asset = cache
        .manifest()
        .asset(LFM3_1_7B_ASSET)
        .expect("manifest entry");
    let mut progress = StderrProgress::new();
    let path = cache
        .resolve(asset, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("cached LFM3 Q8_0");
    let ctx = LlamaContext::load(path, 1).expect("load LFM3");
    let preset = ctx
        .chat_lfm_sampler_preset()
        .expect("LFM3 chat template must use LFM sampler");
    assert!(
        (preset.temp - 0.1).abs() < f32::EPSILON,
        "temp={}",
        preset.temp
    );
    assert_eq!(preset.top_k, 50);
    assert!((preset.repeat_penalty - 1.1).abs() < f32::EPSILON);
}
