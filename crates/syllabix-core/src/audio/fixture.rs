//! File/memory fixtures that exercise the same conversion as native I/O.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

    /// Already-converted v0 frames (tests that concatenate speech + silence).
    pub fn from_frames(frames: Vec<AudioFrame>) -> Self {
        Self { frames, index: 0 }
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

    fn interrupt(&mut self) {
        self.conv.reset();
    }
}

/// Playback conversion that drops PCM after measuring live leftover.
///
/// The six-turn native fixture must not retain every synthesized sample.
#[derive(Clone)]
pub struct DrainStats {
    live_high_water: Arc<AtomicUsize>,
    chunks: Arc<AtomicUsize>,
    samples_played: Arc<AtomicUsize>,
}

impl DrainStats {
    fn new() -> Self {
        Self {
            live_high_water: Arc::new(AtomicUsize::new(0)),
            chunks: Arc::new(AtomicUsize::new(0)),
            samples_played: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Peak converter leftover plus the in-flight converted chunk, in bytes.
    pub fn live_high_water(&self) -> usize {
        self.live_high_water.load(Ordering::SeqCst)
    }

    /// Chunks accepted by [`AudioSink::play`].
    pub fn chunks(&self) -> usize {
        self.chunks.load(Ordering::SeqCst)
    }

    /// Device-layout samples converted, then discarded.
    pub fn samples_played(&self) -> usize {
        self.samples_played.load(Ordering::SeqCst)
    }
}

/// Playback conversion that drops PCM after measuring live leftover.
pub struct DrainingPlayback {
    conv: PcmConverter,
    stats: DrainStats,
}

impl DrainingPlayback {
    /// Play into `device` format without keeping the waveform.
    pub fn new(device: PcmFormat) -> Result<Self> {
        Ok(Self {
            conv: PcmConverter::new(PcmFormat::v0(), device)?,
            stats: DrainStats::new(),
        })
    }

    /// Clone occupancy counters before moving the sink into the pipeline.
    pub fn stats(&self) -> DrainStats {
        self.stats.clone()
    }
}

impl AudioSink for DrainingPlayback {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            return Ok(());
        }
        let f32s = i16_to_f32(&audio.samples);
        let mut converted = self.conv.push(&f32s);
        if audio.is_last {
            converted.extend(self.conv.flush());
        }
        let live = self.conv.leftover_bytes();
        let hw = self.stats.live_high_water.load(Ordering::SeqCst);
        if live > hw {
            self.stats.live_high_water.store(live, Ordering::SeqCst);
        }
        self.stats.chunks.fetch_add(1, Ordering::SeqCst);
        self.stats
            .samples_played
            .fetch_add(converted.len(), Ordering::SeqCst);
        drop(converted);
        Ok(())
    }

    fn interrupt(&mut self) {
        self.conv.reset();
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

    #[test]
    fn draining_playback_does_not_retain_pcm() {
        let mut sink = DrainingPlayback::new(PcmFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        })
        .unwrap();
        sink.play(
            SynthesizedAudio {
                turn: TurnId(0),
                generation: GenerationId(0),
                index: 0,
                samples: vec![100; 512],
                is_last: true,
            },
            &Cancel::new(),
        )
        .unwrap();
        assert!(sink.stats().chunks() > 0);
        assert!(sink.stats().samples_played() > 0);
        assert!(sink.stats().live_high_water() <= crate::audio::AUDIO_LIVE_BYTES_CEILING);
    }
}
