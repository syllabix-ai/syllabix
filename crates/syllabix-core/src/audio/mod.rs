//! Native mic/speaker I/O: device selection, PCM conversion, fixtures, cpal streams.
//!
//! Pipeline frames stay 16 kHz mono PCM16. Devices may run at another rate or
//! channel count; conversion is always on.

mod convert;
mod devices;
mod echo;
mod fixture;
mod native;
mod ring;
mod wav;

pub use convert::{
    f32_to_i16, i16_to_f32, live_buffer_ceiling_bytes, sine_i16, FrameSplitter, PcmConverter,
    PcmFormat, AUDIO_LIVE_BYTES_CEILING, FRAME_SPLITTER_MAX,
};
pub use devices::{
    input_privacy_hint, select_input, select_output, DeviceChoice, DeviceInfo, DeviceInventory,
};
pub use echo::{EchoCalibration, EchoController, EchoRecording, EchoReference, AEC_FRAME_SAMPLES};
pub use fixture::{
    play_fixture_to_device_pcm, record_fixture_to_frames, DrainStats, DrainingPlayback,
    FixtureCapture, FixturePlayback,
};
pub use native::{
    backend_name, probe_device_names, NativeCapture, NativePlayback, SelectedCpalDevice,
};
pub use ring::{device_ring_capacity_samples, SampleRing, DEVICE_RING_MS};
pub use wav::{read_wav, write_wav, WavPcm};

use crate::error::Result;

/// Convert a WAV at any supported rate and channel count into pipeline frames and back to
/// a typical laptop device layout (48 kHz stereo). Used by per-OS smoke tests.
pub fn record_and_play_fixture(wav: &WavPcm) -> Result<(Vec<crate::types::AudioFrame>, Vec<f32>)> {
    let frames = record_fixture_to_frames(wav)?;
    let device = PcmFormat {
        sample_rate_hz: 48_000,
        channels: 2,
    };
    let played = play_fixture_to_device_pcm(&frames, device)?;
    Ok((frames, played))
}
