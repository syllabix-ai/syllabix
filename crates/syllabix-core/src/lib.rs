//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! The runtime connects Silero, whisper.cpp, llama.cpp, and text-to-speech
//! providers through bounded queues. Native inference tests share one binary
//! so large model weights load only once per test process. `SYLLABIX_NATIVE_MODELS`
//! selects specific model suites, while `cfg(coverage)` replaces live audio
//! loops with lightweight fakes. Model weights live in the first-run cache
//! rather than the executable.
//!
//! Hosts that embed this crate: see `docs/embed.md` (`Session` for Lane 3).
//! Only the Lane 2 and Lane 3 lists on that page are a stability promise.
//! Other public items live in named modules (`testing`, `g2p`, `sandbox`,
//! and the rest) and can change.

pub mod audio;
pub mod config;
pub mod defaults;
pub mod dist;
pub mod executor;
#[cfg(all(not(coverage), feature = "ggml", feature = "cloud"))]
mod goose;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod openai;
pub mod pipeline;
pub mod policy;
pub mod queue;
#[cfg(any(coverage, feature = "ggml"))]
pub mod real;
pub mod sandbox;
mod session_env;
pub mod skills;
pub mod stt;
pub mod testing;
pub mod trajectory;
pub mod tts;
pub mod turn_debug;
#[rustfmt::skip]
pub mod types { pub use syllabix_native::types::*; }
pub mod vad;

#[rustfmt::skip]
mod cancel { pub use syllabix_native::cancel::*; }
#[rustfmt::skip]
mod error { pub use syllabix_native::error::*; }
mod fake;
mod harness_trace;
#[cfg(all(feature = "ggml", feature = "device-audio"))]
mod live;
#[rustfmt::skip]
mod onnx { pub use syllabix_native::onnx::*; }
mod providers;
mod session;

#[rustfmt::skip]
pub mod moonshine { pub use crate::stt::moonshine::*; }
#[rustfmt::skip]
pub mod qwen_asr { pub use crate::stt::qwen_asr::*; }
#[cfg(feature = "cloud")]
#[rustfmt::skip]
pub mod remote_stt { pub use crate::stt::remote::*; }
pub(crate) use stt::language;
#[rustfmt::skip]
pub mod g2p { pub use crate::tts::g2p::*; }
#[rustfmt::skip]
pub mod speech_text { pub use crate::tts::speech_text::*; }
#[rustfmt::skip]
pub mod pocket_tts { pub use crate::tts::pocket::*; }
#[cfg(feature = "cloud")]
#[rustfmt::skip]
pub mod remote_tts { pub use crate::tts::remote::*; }

// Unit tests in this crate still write `crate::LlmProvider`. That name is
// not a crate-root `pub use`, so it stays off the freeze list.
#[cfg(test)]
pub(crate) use defaults::LlmProvider;

// Supported crate-root surface is the Lane 2 / Lane 3 lists in docs/embed.md.
// Do not add a new crate-root `pub use` unless that page documents it
// (supported freeze, or an explicit note that it is not frozen).
pub use cancel::Cancel;
pub use config::AgentConfig;
pub use error::{Error, Result};
#[cfg(all(feature = "ggml", feature = "device-audio"))]
pub use live::{run_live, run_live_with_controls};
pub use models::{cache_root, ModelCache};
pub use pipeline::{
    run_loop, run_loop_captured, LoopConfig, LoopEvent, LoopReport, PipelineStages, RuntimeControls,
};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
#[cfg(any(coverage, feature = "ggml"))]
pub use real::load_real_providers;
pub use session::Session;
pub use types::{
    AudioFrame, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId, Utterance, VadEvent,
};

#[cfg(all(test, feature = "ggml"))]
mod shared_ggml_tests {
    #[test]
    fn llama_and_whisper_frontends_share_one_ggml() {
        assert!(syllabix_native::frontends_linked());
        let info = syllabix_native::llama_system_info();
        assert!(!info.is_empty(), "llama.cpp frontend must be linked");
        #[cfg(not(target_vendor = "apple"))]
        {
            assert_eq!(syllabix_native::llama_n_gpu_layers(), 0);
            assert!(!syllabix_native::whisper_use_gpu());
        }
        #[cfg(target_vendor = "apple")]
        {
            assert_eq!(syllabix_native::llama_n_gpu_layers(), -1);
            assert!(syllabix_native::whisper_use_gpu());
        }
    }
}
