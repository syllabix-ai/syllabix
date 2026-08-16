//! Native inference suite (whisper.cpp, llama.cpp, Kokoro, six-turn loop).
//!
//! One integration binary so Whisper/Llama/Kokoro load once per `cargo test`
//! process. `cargo llvm-cov` sets `--cfg coverage` and skips these modules; the
//! 85% unit-coverage floor is restored in a follow-up PR.

#[cfg(not(coverage))]
mod kokoro;
#[cfg(not(coverage))]
mod llama;
#[cfg(not(coverage))]
mod real_loop;
#[cfg(not(coverage))]
mod whisper;

#[cfg(not(coverage))]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(not(coverage))]
use syllabix_core::{
    Cancel, HttpFetcher, KokoroTts, LlamaLlm, ModelCache, StderrProgress, WhisperStt,
};

#[cfg(not(coverage))]
pub(crate) struct Native {
    pub stt: WhisperStt,
    pub llm: LlamaLlm,
    pub tts: KokoroTts,
}

#[cfg(not(coverage))]
pub(crate) fn hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Shared ggml is process-global; serialize every native test.
#[cfg(not(coverage))]
pub(crate) fn native() -> MutexGuard<'static, Native> {
    static CELL: OnceLock<Mutex<Native>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let cancel = Cancel::new();
        let stt = WhisperStt::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
            .expect("load whisper.cpp small once");
        let llm = LlamaLlm::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
            .expect("load Llama Q4_K_M once");
        let tts = KokoroTts::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
            .expect("load Kokoro ONNX once");
        Mutex::new(Native { stt, llm, tts })
    })
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(coverage)]
#[test]
fn native_inference_skipped_under_llvm_cov() {}
