//! Shared `ggml` plus the whisper.cpp and llama.cpp frontends.

#[cfg(feature = "cpp")]
mod cpp;
#[cfg(feature = "cpp")]
pub use cpp::*;
