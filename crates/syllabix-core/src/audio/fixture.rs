//! File/memory fixtures that exercise the same conversion as native I/O.

use crate::audio::convert::{f32_to_i16, i16_to_f32, FrameSplitter, PcmConverter, PcmFormat};
use crate::audio::wav::WavPcm;
use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::providers::{AudioCapture, AudioSink};
use crate::types::{AudioFrame, SynthesizedAudio};

/// Capture path: device-format WAV → 16 kHz mono frames.
pub fn record_fixture_to_frames(wav: &WavPcm) -> Result<Vec<AudioFrame>> {
    let mut conv = PcmConverter::new(wav.format, PcmFormat::v0())?;
    let mut split = FrameSplitter::new();
    let f32s = i16_to_f32(&wav.samples);
    let mut mono = conv.push(&f32s);
    mono.extend(conv.flush());
    let i16s = f32_to_i16(&mono);
    let mut frames = split.push(&i16s)?;
    frames.extend(split.flush()?);
    Ok(frames)
}

/// Playback path: v0 frames → device-format interleaved f32.
pub fn play_fixture_to_device_pcm(frames: &[AudioFrame], device: PcmFormat) -> Result<Vec<f32>> {
    let mut conv = PcmConverter::new(PcmFormat::v0(), device)?;
    let mut out = Vec::new();
    for frame in frames {
        frame.validate()?;
        let f32s = i16_to_f32(&frame.samples);
        out.extend(conv.push(&f32s));
    }
    out.extend(conv.flush());
    Ok(out)
}

/// [`AudioCapture`] over a preloaded WAV.
pub struct FixtureCapture {
    frames: Vec<AudioFrame>,
    index: usize,
}

impl FixtureCapture {
    /// Convert `wav` as if it were captured from a device.
    pub fn from_wav(wav: &WavPcm) -> Result<Self> {
        Ok(Self {
            frames: record_fixture_to_frames(wav)?,
            index: 0,
        })
    }
}

impl AudioCapture for FixtureCapture {
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if self.index >= self.frames.len() {
            return Ok(None);
        }
        let frame = self.frames[self.index].clone();
        self.index += 1;
        Ok(Some(frame))
    }
}

/// [`AudioSink`] that converts to a device layout and keeps the PCM for tests.
pub struct FixturePlayback {
    conv: PcmConverter,
    /// Interleaved device-format samples collected so far.
    pub pcm: Vec<f32>,
}

impl FixturePlayback {
    /// Play into `device` format.
    pub fn new(device: PcmFormat) -> Result<Self> {
        Ok(Self {
            conv: PcmConverter::new(PcmFormat::v0(), device)?,
            pcm: Vec::new(),
        })
    }
}

impl AudioSink for FixturePlayback {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            return Ok(());
        }
        let f32s = i16_to_f32(&audio.samples);
        self.pcm.extend(self.conv.push(&f32s));
        if audio.is_last {
            self.pcm.extend(self.conv.flush());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::convert::sine_i16;
    use crate::types::{GenerationId, TurnId};
    use std::time::Duration;

    #[test]
    fn capture_cancel_is_visible() {
        let wav = WavPcm {
            format: PcmFormat::v0(),
            samples: sine_i16(16_000, 1, 440.0, Duration::from_millis(100), 0.3),
        };
        let mut cap = FixtureCapture::from_wav(&wav).unwrap();
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = cap.next_frame(&cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn playback_cancel_is_visible() {
        let mut sink = FixturePlayback::new(PcmFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        })
        .unwrap();
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = sink
            .play(
                SynthesizedAudio {
                    turn: TurnId(0),
                    generation: GenerationId(0),
                    index: 0,
                    samples: vec![1; 64],
                    is_last: true,
                },
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }
}
