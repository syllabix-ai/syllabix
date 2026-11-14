//! Shared Syllabix types plus the whisper.cpp and llama.cpp frontends.
//!
//! Errors, cancel tokens, audio frames, and WAV/PCM helpers are always on.
//! Model cache and downloads sit behind `models`; ONNX helpers behind `onnx`.
//! The vendored C++ engines sit behind `cpp`.

pub mod audio;
pub mod cancel;
pub mod compute;
pub mod error;
#[cfg(feature = "models")]
pub mod models;
#[cfg(feature = "onnx")]
#[doc(hidden)]
pub mod onnx;
pub mod timeouts;
pub mod types;

pub use cancel::Cancel;
pub use error::{Error, Result};

#[cfg(feature = "cpp")]
mod cpp;
#[cfg(feature = "cpp")]
pub use cpp::*;
