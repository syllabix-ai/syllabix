//! Shared STFT helpers live in `syllabix-native` so TTS iSTFT and Hima STFT
//! reuse one Hann window, padding, and forward rFFT implementation.
#[cfg(feature = "hima")]
pub use syllabix_native::audio::stft::*;
