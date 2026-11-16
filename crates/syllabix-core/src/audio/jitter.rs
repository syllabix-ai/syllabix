//! Start / refill / cap policy in front of the cpal device ring (#265 / #373).

use std::collections::VecDeque;

/// Do not forward PCM until this much is buffered (start and after underrun refill).
pub const JITTER_START_MS: u32 = 90;
/// Same threshold after an underrun.
pub const JITTER_REFILL_MS: u32 = 90;
/// Drop oldest samples when buffered duration exceeds this.
pub const JITTER_CAP_MS: u32 = 240;

/// Buffers device-rate interleaved f32 PCM before it reaches [`super::SampleRing`].
pub struct JitterBuffer {
    buffer: VecDeque<f32>,
    rate_hz: u32,
    channels: u16,
    playing: bool,
}

impl JitterBuffer {
    pub fn new(rate_hz: u32, channels: u16) -> Self {
        assert!(rate_hz > 0);
        assert!(channels > 0);
        Self {
            buffer: VecDeque::new(),
            rate_hz,
            channels,
            playing: false,
        }
    }

    fn ms_to_samples(&self, ms: u32) -> usize {
        (self.rate_hz as usize) * (self.channels as usize) * (ms as usize) / 1000
    }

    fn start_samples(&self) -> usize {
        self.ms_to_samples(JITTER_START_MS)
    }

    fn cap_samples(&self) -> usize {
        self.ms_to_samples(JITTER_CAP_MS)
    }

    fn refill_samples(&self) -> usize {
        self.ms_to_samples(JITTER_REFILL_MS)
    }

    fn apply_cap(&mut self) {
        let cap = self.cap_samples();
        let refill = self.refill_samples();
        while self.buffer.len() > cap {
            let drop = self.buffer.len().saturating_sub(refill);
            for _ in 0..drop {
                self.buffer.pop_front();
            }
        }
    }

    /// Push samples; apply cap. Returns PCM that may be written to the device ring
    /// (empty while priming or refilling).
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.buffer.extend(samples);
        self.apply_cap();

        if !self.playing {
            if self.buffer.len() >= self.start_samples() {
                self.playing = true;
                return self.buffer.drain(..).collect();
            }
            return Vec::new();
        }

        self.buffer.drain(..).collect()
    }

    /// Drain buffered PCM without waiting for the start threshold (end of utterance).
    pub fn flush(&mut self) -> Vec<f32> {
        self.playing = false;
        self.buffer.drain(..).collect()
    }

    /// Playback underrun: hold output until refill threshold is met again.
    pub fn note_underrun(&mut self) {
        self.playing = false;
    }

    /// Barge-in: drop buffered PCM and require a fresh start.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.playing = false;
    }

    /// Wall-time duration of PCM still held in the jitter buffer.
    pub fn buffered_ms(&self) -> u32 {
        if self.buffer.is_empty() {
            return 0;
        }
        let denom = self.rate_hz as u64 * self.channels as u64;
        ((self.buffer.len() as u64) * 1000 / denom) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const CH: u16 = 1;
    const START: usize = (RATE as usize) * (CH as usize) * (JITTER_START_MS as usize) / 1000;
    const REFILL: usize = (RATE as usize) * (CH as usize) * (JITTER_REFILL_MS as usize) / 1000;
    const CAP: usize = (RATE as usize) * (CH as usize) * (JITTER_CAP_MS as usize) / 1000;

    fn seq(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    #[test]
    fn holds_until_start_threshold() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert!(jitter.push(&seq(START - 1, 0.0)).is_empty());
    }

    #[test]
    fn starts_and_returns_buffered_pcm_at_threshold() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert!(jitter.push(&seq(START - 1, 0.0)).is_empty());
        assert!(jitter.buffered_ms() >= 89 && jitter.buffered_ms() <= 90);
        let out = jitter.push(&[START as f32 - 1.0]);
        assert_eq!(out.len(), START);
        assert_eq!(jitter.buffered_ms(), 0);
    }

    #[test]
    fn playing_forwards_small_follow_up_immediately() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert_eq!(jitter.push(&seq(START, 0.0)).len(), START);
        let follow_up = 64;
        let out = jitter.push(&seq(follow_up, 100.0));
        assert_eq!(out.len(), follow_up);
        assert_eq!(out[0], 100.0);
        assert!(jitter.buffered_ms() <= 1);
    }

    #[test]
    fn underrun_requires_refill() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert_eq!(jitter.push(&seq(START, 0.0)).len(), START);
        jitter.note_underrun();
        assert!(jitter.push(&seq(START - 1, 10.0)).is_empty());
        let out = jitter.push(&[10.0 + START as f32 - 1.0]);
        assert_eq!(out.len(), START);
        assert_eq!(out[0], 10.0);
    }

    #[test]
    fn cap_drops_oldest_until_refill_and_forwards_retained_tail() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert_eq!(jitter.push(&seq(START, 0.0)).len(), START);
        let oldest_tag = 1_000.0;
        let extra = START + 1;
        let chunk = seq(CAP + extra, oldest_tag);
        let out = jitter.push(&chunk);
        assert_eq!(jitter.buffer.len(), 0);
        assert_eq!(out.len(), REFILL);
        let dropped = CAP + extra - REFILL;
        assert_eq!(out[0], oldest_tag + dropped as f32);
        assert_eq!(out[REFILL - 1], oldest_tag + (CAP + extra - 1) as f32);
    }

    #[test]
    fn flush_drains_under_start_threshold() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert!(jitter.push(&seq(START - 1, 5.0)).is_empty());
        let out = jitter.flush();
        assert_eq!(out.len(), START - 1);
        assert_eq!(out[0], 5.0);
        assert_eq!(jitter.buffered_ms(), 0);
        assert!(jitter.push(&seq(START - 1, 0.0)).is_empty());
    }

    #[test]
    fn clear_requires_fresh_start() {
        let mut jitter = JitterBuffer::new(RATE, CH);
        assert_eq!(jitter.push(&seq(START, 0.0)).len(), START);
        jitter.clear();
        assert_eq!(jitter.buffered_ms(), 0);
        assert!(jitter.push(&seq(START - 1, 0.0)).is_empty());
        assert_eq!(jitter.push(&[1.0]).len(), START);
    }
}
