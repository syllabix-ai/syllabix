//! Sample-rate and channel conversion. Internal working format is f32.

use std::collections::VecDeque;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::types::{AudioFrame, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};

/// After warm-up, live converter + splitter + declared queues must stay under this.
/// 32 frame slots + 16 audio slots + leftover PCM, with headroom.
pub const AUDIO_LIVE_BYTES_CEILING: usize = 256 * 1024;

/// Frame splitter leftover is always fewer than one v0 frame.
pub const FRAME_SPLITTER_MAX: usize = FRAME_SAMPLES - 1;

/// Interleaved PCM layout at a device or pipeline edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmFormat {
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count.
    pub channels: u16,
}

impl PcmFormat {
    /// v0 pipeline PCM (16 kHz mono).
    pub const fn v0() -> Self {
        Self {
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
        }
    }

    fn validate(self) -> Result<()> {
        if self.sample_rate_hz == 0 || self.sample_rate_hz > 192_000 {
            return Err(Error::InvalidAudio {
                message: format!(
                    "sample rate {} Hz is not usable (need 1–192000 Hz)",
                    self.sample_rate_hz
                ),
            });
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(Error::InvalidAudio {
                message: format!(
                    "channel count {} is not usable (need 1–8 channels)",
                    self.channels
                ),
            });
        }
        Ok(())
    }
}

/// Converts interleaved f32 PCM from `src` to `dst`.
///
/// Always downmixes to mono, linearly resamples, then duplicates to `dst.channels`.
/// Leftover state is a few samples — not a growing buffer.
#[derive(Debug)]
pub struct PcmConverter {
    src: PcmFormat,
    dst: PcmFormat,
    partial_frame: Vec<f32>,
    mono: VecDeque<f32>,
    phase: f64,
}

impl PcmConverter {
    /// Build a converter after checking rates and channel counts.
    pub fn new(src: PcmFormat, dst: PcmFormat) -> Result<Self> {
        src.validate()?;
        dst.validate()?;
        Ok(Self {
            src,
            dst,
            partial_frame: Vec::new(),
            mono: VecDeque::new(),
            phase: 0.0,
        })
    }

    /// Source layout.
    pub fn src(&self) -> PcmFormat {
        self.src
    }

    /// Destination layout.
    pub fn dst(&self) -> PcmFormat {
        self.dst
    }

    /// Samples held for the next call (incomplete frames + resampler history).
    pub fn leftover_samples(&self) -> usize {
        self.partial_frame.len() + self.mono.len()
    }

    /// Approximate live bytes in converter state.
    pub fn leftover_bytes(&self) -> usize {
        self.leftover_samples() * std::mem::size_of::<f32>()
    }

    /// Convert one interleaved chunk. Output is interleaved at `dst`.
    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        self.partial_frame.extend_from_slice(input);
        let ch = self.src.channels as usize;
        let complete = self.partial_frame.len() / ch * ch;
        if complete == 0 {
            return Vec::new();
        }
        let rest = self.partial_frame.split_off(complete);
        let chunk = std::mem::replace(&mut self.partial_frame, rest);
        for frame in chunk.chunks_exact(ch) {
            let mono = if ch == 1 {
                frame[0]
            } else {
                frame.iter().sum::<f32>() / ch as f32
            };
            self.mono.push_back(mono);
        }
        let mono_out = resample_linear(
            &mut self.mono,
            &mut self.phase,
            self.src.sample_rate_hz,
            self.dst.sample_rate_hz,
        );
        expand_channels(&mono_out, self.dst.channels)
    }

    /// Pad the resampler so the last input samples become output.
    pub fn flush(&mut self) -> Vec<f32> {
        if !self.partial_frame.is_empty() {
            let ch = self.src.channels as usize;
            while !self.partial_frame.len().is_multiple_of(ch) {
                self.partial_frame.push(0.0);
            }
            let tail = std::mem::take(&mut self.partial_frame);
            for frame in tail.chunks_exact(ch) {
                let mono = frame.iter().sum::<f32>() / ch as f32;
                self.mono.push_back(mono);
            }
        }
        // Two extra zeros so linear interpolation can emit the last real sample.
        self.mono.push_back(0.0);
        self.mono.push_back(0.0);
        let mono_out = resample_linear(
            &mut self.mono,
            &mut self.phase,
            self.src.sample_rate_hz,
            self.dst.sample_rate_hz,
        );
        self.mono.clear();
        self.phase = 0.0;
        expand_channels(&mono_out, self.dst.channels)
    }
}

fn resample_linear(
    mono: &mut VecDeque<f32>,
    phase: &mut f64,
    in_rate: u32,
    out_rate: u32,
) -> Vec<f32> {
    if in_rate == out_rate {
        *phase = 0.0;
        return mono.drain(..).collect();
    }
    let step = f64::from(in_rate) / f64::from(out_rate);
    let mut out = Vec::new();
    loop {
        let i0 = phase.floor() as usize;
        let i1 = i0 + 1;
        if i1 >= mono.len() {
            break;
        }
        let frac = (*phase - i0 as f64) as f32;
        let a = mono[i0];
        let b = mono[i1];
        out.push(a + (b - a) * frac);
        *phase += step;
    }
    let consume = phase.floor() as usize;
    if consume > 0 {
        let consume = consume.min(mono.len().saturating_sub(1));
        for _ in 0..consume {
            mono.pop_front();
        }
        *phase -= consume as f64;
        if *phase < 0.0 {
            *phase = 0.0;
        }
    }
    out
}

fn expand_channels(mono: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return mono.to_vec();
    }
    let ch = channels as usize;
    let mut out = Vec::with_capacity(mono.len() * ch);
    for sample in mono {
        for _ in 0..ch {
            out.push(*sample);
        }
    }
    out
}

/// PCM16 → f32 in `-1.0..1.0`.
pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|s| f32::from(*s) / 32768.0).collect()
}

/// f32 → PCM16 with clipping.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| {
            let scaled = s.clamp(-1.0, 1.0) * 32767.0;
            scaled.round() as i16
        })
        .collect()
}

/// Pack converted 16 kHz mono i16 into [`FRAME_SAMPLES`] frames.
pub struct FrameSplitter {
    buf: Vec<i16>,
    next_seq: u64,
}

impl FrameSplitter {
    /// Empty splitter starting at sequence 0.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            next_seq: 0,
        }
    }

    /// Samples waiting for a full frame.
    pub fn leftover_samples(&self) -> usize {
        self.buf.len()
    }

    /// Push PCM16 mono at the v0 rate; emit complete frames.
    pub fn push(&mut self, samples: &[i16]) -> Result<Vec<AudioFrame>> {
        self.buf.extend_from_slice(samples);
        let mut frames = Vec::new();
        while self.buf.len() >= FRAME_SAMPLES {
            let rest = self.buf.split_off(FRAME_SAMPLES);
            let samples = std::mem::replace(&mut self.buf, rest);
            let frame = AudioFrame::new(
                self.next_seq,
                DEFAULT_SAMPLE_RATE_HZ,
                DEFAULT_CHANNELS,
                samples,
            )?;
            self.next_seq += 1;
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Pad a trailing partial frame with silence so short fixtures still emit.
    pub fn flush(&mut self) -> Result<Vec<AudioFrame>> {
        if self.buf.is_empty() {
            return Ok(Vec::new());
        }
        let pad = FRAME_SAMPLES - self.buf.len();
        let zeros = vec![0; pad];
        self.push(&zeros)
    }
}

impl Default for FrameSplitter {
    fn default() -> Self {
        Self::new()
    }
}

/// A-law-ish bound used by the 30-minute (simulated) soak.
pub fn live_buffer_ceiling_bytes() -> usize {
    AUDIO_LIVE_BYTES_CEILING
}

/// Interleaved sine at `hz` for fixture WAV files and hardware smokes.
pub fn sine_i16(
    rate_hz: u32,
    channels: u16,
    hz: f32,
    duration: Duration,
    amplitude: f32,
) -> Vec<i16> {
    let n = (u64::from(rate_hz) * duration.as_millis() as u64 / 1000) as usize;
    let mut out = Vec::with_capacity(n * channels as usize);
    let two_pi = std::f32::consts::PI * 2.0;
    for i in 0..n {
        let t = i as f32 / rate_hz as f32;
        let s = (two_pi * hz * t).sin() * amplitude;
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        for _ in 0..channels {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_rate_or_channels() {
        let err = PcmConverter::new(
            PcmFormat {
                sample_rate_hz: 0,
                channels: 1,
            },
            PcmFormat::v0(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
    }

    #[test]
    fn stereo_48k_to_mono_16k_shortens_by_about_six() {
        let mut conv = PcmConverter::new(
            PcmFormat {
                sample_rate_hz: 48_000,
                channels: 2,
            },
            PcmFormat::v0(),
        )
        .unwrap();
        // 48000 stereo frames = 96000 samples → 16000 mono after convert.
        let input = vec![0.5_f32; 48_000 * 2];
        let mut out = conv.push(&input);
        out.extend(conv.flush());
        let expected = 16_000usize;
        let delta = (out.len() as i32 - expected as i32).unsigned_abs() as usize;
        assert!(
            delta < 32,
            "expected ~{expected} mono samples, got {}",
            out.len()
        );
        assert!(conv.leftover_samples() < 16);
    }

    #[test]
    fn mono_16k_to_stereo_48k_triples_rate_and_duplicates() {
        let mut conv = PcmConverter::new(
            PcmFormat::v0(),
            PcmFormat {
                sample_rate_hz: 48_000,
                channels: 2,
            },
        )
        .unwrap();
        let input = vec![0.25_f32; 16_000];
        let mut out = conv.push(&input);
        out.extend(conv.flush());
        assert_eq!(out.len() % 2, 0);
        let frames = out.len() / 2;
        let delta = (frames as i32 - 48_000).unsigned_abs();
        assert!(delta < 32, "got {frames} stereo frames");
        assert!((out[0] - out[1]).abs() < 1e-6);
    }

    #[test]
    fn leftover_stays_bounded_across_tiny_chunks() {
        let mut conv = PcmConverter::new(
            PcmFormat {
                sample_rate_hz: 44_100,
                channels: 1,
            },
            PcmFormat::v0(),
        )
        .unwrap();
        for _ in 0..400 {
            let _ = conv.push(&[0.1, 0.2, 0.3]);
            assert!(
                conv.leftover_samples() < 64,
                "leftover grew to {}",
                conv.leftover_samples()
            );
        }
    }

    #[test]
    fn frame_splitter_emits_512_and_keeps_tail() {
        let mut split = FrameSplitter::new();
        let frames = split.push(&vec![1; FRAME_SAMPLES + 7]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples.len(), FRAME_SAMPLES);
        assert_eq!(split.leftover_samples(), 7);
        assert!(split.leftover_samples() <= FRAME_SPLITTER_MAX);
    }

    #[test]
    fn sine_is_non_silent() {
        let s = sine_i16(16_000, 1, 440.0, Duration::from_millis(50), 0.4);
        assert!(s.iter().any(|v| *v != 0));
        assert_eq!(s.len(), 16_000 * 50 / 1000);
    }
}
