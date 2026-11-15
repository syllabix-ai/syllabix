//! PCM conversion and WAV helpers shared by the speech crates.

pub mod convert;
#[cfg(feature = "cpp")]
pub mod istft;
#[cfg(feature = "dsp")]
pub mod stft;
pub mod wav;

pub use convert::{
    f32_to_i16, i16_to_f32, live_buffer_ceiling_bytes, sine_i16, FrameSplitter, PcmConverter,
    PcmFormat, AUDIO_LIVE_BYTES_CEILING, FRAME_SPLITTER_MAX,
};
#[cfg(feature = "cpp")]
pub use istft::{Istft, HOP, N_FFT, N_FREQ, SAME_TRIM, SAMPLES_PER_TOKEN};
#[cfg(feature = "dsp")]
pub use stft::{
    padded_periodic_hann, periodic_hann, power_spectrogram, reflect_pad, PowerSpectrogram,
};
pub use wav::{read_wav, write_wav, WavPcm};
