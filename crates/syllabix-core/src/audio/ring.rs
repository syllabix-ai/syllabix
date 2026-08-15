//! Fixed-capacity sample ring for cpal callbacks (they must not block).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::error::{Error, Result};

/// Device-side ring holds this much real time of PCM.
pub const DEVICE_RING_MS: u32 = 200;

/// Interleaved sample capacity for a 200 ms device buffer.
pub fn device_ring_capacity_samples(rate_hz: u32, channels: u16) -> usize {
    (rate_hz as usize) * (channels as usize) * (DEVICE_RING_MS as usize) / 1000
}

/// Bounded f32 PCM queue with occupancy stats.
pub struct SampleRing {
    cap: usize,
    inner: Mutex<VecDeque<f32>>,
    not_empty: Condvar,
    not_full: Condvar,
    high_water: AtomicUsize,
    overruns: AtomicU64,
    underruns: AtomicU64,
    closed: AtomicBool,
}

impl SampleRing {
    /// `cap` is interleaved samples, not frames.
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0);
        Self {
            cap,
            inner: Mutex::new(VecDeque::with_capacity(cap)),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            high_water: AtomicUsize::new(0),
            overruns: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        }
    }

    /// Interleaved sample capacity.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Samples currently queued.
    pub fn occupancy(&self) -> usize {
        self.inner.lock().expect("sample ring").len()
    }

    /// Peak occupancy.
    pub fn high_water(&self) -> usize {
        self.high_water.load(Ordering::SeqCst)
    }

    /// Samples dropped because the ring was full (capture overrun).
    pub fn overruns(&self) -> u64 {
        self.overruns.load(Ordering::SeqCst)
    }

    /// Silent samples inserted because the ring was empty (playback underrun).
    pub fn underruns(&self) -> u64 {
        self.underruns.load(Ordering::SeqCst)
    }

    /// Wake waiters and refuse further pushes.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    fn note_len(&self, len: usize) {
        let mut hw = self.high_water.load(Ordering::SeqCst);
        while len > hw {
            match self
                .high_water
                .compare_exchange(hw, len, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(actual) => hw = actual,
            }
        }
    }

    /// Callback-safe: write what fits, count the rest as overrun.
    pub fn try_push_slice(&self, samples: &[f32]) -> usize {
        if self.closed.load(Ordering::SeqCst) {
            return 0;
        }
        let mut q = self.inner.lock().expect("sample ring");
        let mut written = 0;
        for s in samples {
            if q.len() >= self.cap {
                break;
            }
            q.push_back(*s);
            written += 1;
        }
        self.note_len(q.len());
        let dropped = samples.len() - written;
        if dropped > 0 {
            self.overruns.fetch_add(dropped as u64, Ordering::SeqCst);
        }
        self.not_empty.notify_all();
        written
    }

    /// Pipeline-thread push. Blocks while the ring is full unless cancelled.
    pub fn push_slice_cancellable(&self, samples: &[f32], cancel: &Cancel) -> Result<()> {
        let mut offset = 0;
        while offset < samples.len() {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if self.closed.load(Ordering::SeqCst) {
                return Err(Error::Disconnected { stage: "sink" });
            }
            let mut q = self.inner.lock().expect("sample ring");
            while q.len() >= self.cap
                && !self.closed.load(Ordering::SeqCst)
                && !cancel.is_shutdown()
            {
                let (guard, wait) = self
                    .not_full
                    .wait_timeout(q, Duration::from_millis(5))
                    .expect("sample ring condvar");
                q = guard;
                if wait.timed_out() {
                    drop(q);
                    if cancel.is_shutdown() {
                        return Err(Error::Cancelled);
                    }
                    q = self.inner.lock().expect("sample ring");
                }
            }
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if self.closed.load(Ordering::SeqCst) {
                return Err(Error::Disconnected { stage: "sink" });
            }
            while offset < samples.len() && q.len() < self.cap {
                q.push_back(samples[offset]);
                offset += 1;
            }
            self.note_len(q.len());
            self.not_empty.notify_all();
        }
        Ok(())
    }

    /// Callback-safe pop. Short reads increment underrun by the missing count.
    pub fn try_pop_slice(&self, dest: &mut [f32]) -> usize {
        let mut q = self.inner.lock().expect("sample ring");
        let mut n = 0;
        while n < dest.len() {
            match q.pop_front() {
                Some(s) => {
                    dest[n] = s;
                    n += 1;
                }
                None => break,
            }
        }
        if n < dest.len() {
            self.underruns
                .fetch_add((dest.len() - n) as u64, Ordering::SeqCst);
            for slot in dest.iter_mut().skip(n) {
                *slot = 0.0;
            }
        }
        self.not_full.notify_all();
        n
    }

    /// Block until at least one sample, then copy up to `dest.len()`.
    pub fn pop_slice_cancellable(&self, dest: &mut [f32], cancel: &Cancel) -> Result<usize> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let mut q = self.inner.lock().expect("sample ring");
            if q.is_empty() {
                if self.closed.load(Ordering::SeqCst) {
                    return Ok(0);
                }
                let (guard, _) = self
                    .not_empty
                    .wait_timeout(q, Duration::from_millis(5))
                    .expect("sample ring condvar");
                drop(guard);
                continue;
            }
            let mut n = 0;
            while n < dest.len() {
                match q.pop_front() {
                    Some(s) => {
                        dest[n] = s;
                        n += 1;
                    }
                    None => break,
                }
            }
            self.not_full.notify_all();
            return Ok(n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_200ms() {
        assert_eq!(device_ring_capacity_samples(48_000, 2), 48_000 * 2 / 5);
        assert_eq!(device_ring_capacity_samples(16_000, 1), 3_200);
    }

    #[test]
    fn try_push_does_not_exceed_cap() {
        let ring = SampleRing::new(4);
        assert_eq!(ring.try_push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]), 4);
        assert_eq!(ring.occupancy(), 4);
        assert_eq!(ring.high_water(), 4);
        assert!(ring.high_water() <= ring.capacity());
        assert_eq!(ring.overruns(), 1);
        let mut buf = [0.0; 4];
        assert_eq!(ring.try_pop_slice(&mut buf), 4);
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn underrun_fills_silence() {
        let ring = SampleRing::new(4);
        let mut buf = [9.0; 3];
        assert_eq!(ring.try_pop_slice(&mut buf), 0);
        assert_eq!(buf, [0.0, 0.0, 0.0]);
        assert_eq!(ring.underruns(), 3);
    }

    #[test]
    fn cancellable_push_stops_on_shutdown() {
        let ring = SampleRing::new(1);
        ring.try_push_slice(&[1.0]);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = ring.push_slice_cancellable(&[2.0], &cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }
}
