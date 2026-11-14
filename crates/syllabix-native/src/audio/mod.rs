//! PCM conversion and WAV helpers shared by the speech crates.

pub mod convert;
pub mod wav;

pub use convert::{
    f32_to_i16, i16_to_f32, live_buffer_ceiling_bytes, sine_i16, FrameSplitter, PcmConverter,
    PcmFormat, AUDIO_LIVE_BYTES_CEILING, FRAME_SPLITTER_MAX,
};
pub use wav::{read_wav, write_wav, WavPcm};
