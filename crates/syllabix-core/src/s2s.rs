//! Blocking speech-to-speech duplex backend.
//!
//! This module is not re-exported at the crate root and is not on the
//! supported-API lists. The duplex loop, jitter buffer, and Moshi client
//! are later spike PRs.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::types::AudioFrame;

const FAKE_S2S_DELAY: Duration = Duration::from_millis(20);

/// One event from a duplex S2S backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum S2sEvent {
    /// Decoded PCM at the pipeline rate (16 kHz mono).
    Audio(AudioFrame),
    /// Text the model emitted alongside audio.
    Text(String),
    /// Backend failure that should stop the loop.
    Error {
        /// Human-readable reason.
        message: String,
    },
}

/// Full-duplex speech-to-speech backend with a blocking push/poll API.
///
/// Push sends post-AEC capture frames. Poll waits for the next model event
/// (audio for the sink and AEC far-end, plus text or errors). Both methods
/// check [`Cancel`] and return [`Error::Cancelled`] on shutdown.
pub trait DuplexS2s: Send + Sync {
    /// Backend name (`fake-s2s`, later `moshi`).
    fn name(&self) -> &'static str;

    /// Send one post-AEC capture frame. Blocks until the backend accepts it.
    fn push(&self, frame: AudioFrame, cancel: &Cancel) -> Result<()>;

    /// Block until the next event is available.
    fn poll(&self, cancel: &Cancel) -> Result<S2sEvent>;
}

/// In-memory backend that echoes pushed audio back after a delay.
///
/// Used by tests. Push and poll may run on different threads.
pub struct FakeS2s {
    delay: Duration,
    inner: Mutex<VecDeque<(Instant, S2sEvent)>>,
    ready: Condvar,
}

impl FakeS2s {
    /// Echo each pushed frame after `delay`.
    pub fn new(delay: Duration) -> Self {
        Self {
            delay,
            inner: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(Instant, S2sEvent)>> {
        self.inner.lock().expect("fake s2s")
    }

    fn enqueue(&self, event: S2sEvent, delay: Duration, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut queue = self.lock();
        queue.push_back((Instant::now() + delay, event));
        self.ready.notify_one();
        Ok(())
    }
}

impl Default for FakeS2s {
    fn default() -> Self {
        Self::new(FAKE_S2S_DELAY)
    }
}

impl DuplexS2s for FakeS2s {
    fn name(&self) -> &'static str {
        "fake-s2s"
    }

    fn push(&self, frame: AudioFrame, cancel: &Cancel) -> Result<()> {
        frame.validate()?;
        self.enqueue(S2sEvent::Audio(frame), self.delay, cancel)
    }

    fn poll(&self, cancel: &Cancel) -> Result<S2sEvent> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let mut queue = self.lock();
            match queue.front() {
                Some((ready_at, _)) if Instant::now() >= *ready_at => {
                    return Ok(queue.pop_front().expect("queued s2s event").1);
                }
                Some((ready_at, _)) => {
                    let timeout = ready_at.saturating_duration_since(Instant::now());
                    let timeout = timeout.min(Duration::from_millis(10));
                    drop(self.ready.wait_timeout(queue, timeout).expect("fake s2s"));
                }
                None => {
                    drop(
                        self.ready
                            .wait_timeout(queue, Duration::from_millis(10))
                            .expect("fake s2s"),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};
    use std::sync::Arc;
    use std::thread;

    fn frame(seq: u64, sample: i16) -> AudioFrame {
        AudioFrame::new(
            seq,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![sample; FRAME_SAMPLES],
        )
        .expect("test frame")
    }

    fn audio(event: S2sEvent) -> AudioFrame {
        match event {
            S2sEvent::Audio(frame) => frame,
            other => panic!("expected audio, got {other:?}"),
        }
    }

    #[test]
    fn fake_name_and_default_delay() {
        let fake = FakeS2s::default();
        assert_eq!(DuplexS2s::name(&fake), "fake-s2s");
        assert_eq!(fake.delay, FAKE_S2S_DELAY);
    }

    #[test]
    fn echoes_pushed_audio_after_delay() {
        let delay = Duration::from_millis(15);
        let fake = FakeS2s::new(delay);
        let cancel = Cancel::new();
        let input = frame(3, 42);
        let start = Instant::now();
        fake.push(input.clone(), &cancel).expect("push");
        let got = audio(fake.poll(&cancel).expect("poll"));
        assert!(
            start.elapsed() >= delay,
            "echo returned too early: {:?}",
            start.elapsed()
        );
        assert_eq!(got, input);
    }

    #[test]
    fn echoes_frames_in_push_order() {
        let fake = FakeS2s::new(Duration::ZERO);
        let cancel = Cancel::new();
        let first = frame(0, 1);
        let second = frame(1, 2);
        fake.push(first.clone(), &cancel).expect("push first");
        fake.push(second.clone(), &cancel).expect("push second");
        assert_eq!(audio(fake.poll(&cancel).expect("poll first")), first);
        assert_eq!(audio(fake.poll(&cancel).expect("poll second")), second);
    }

    #[test]
    fn push_rejects_invalid_audio_and_shutdown() {
        let fake = FakeS2s::new(Duration::ZERO);
        let cancel = Cancel::new();
        let err = fake
            .push(
                AudioFrame {
                    seq: 0,
                    sample_rate_hz: 8_000,
                    channels: 1,
                    samples: vec![0; FRAME_SAMPLES],
                    capture_pcm: None,
                },
                &cancel,
            )
            .expect_err("bad rate");
        assert!(matches!(err, Error::InvalidAudio { .. }));

        cancel.shutdown();
        let err = fake.push(frame(0, 1), &cancel).expect_err("shutdown push");
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn poll_returns_cancelled_while_waiting() {
        let fake = Arc::new(FakeS2s::new(Duration::from_secs(30)));
        let cancel = Cancel::new();
        let poller = Arc::clone(&fake);
        let poll_cancel = cancel.clone();
        let handle = thread::spawn(move || poller.poll(&poll_cancel));
        thread::sleep(Duration::from_millis(5));
        cancel.shutdown();
        let err = handle.join().expect("poll thread").expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn poll_returns_cancelled_during_echo_delay() {
        let fake = Arc::new(FakeS2s::new(Duration::from_secs(30)));
        let cancel = Cancel::new();
        fake.push(frame(0, 7), &cancel).expect("push");
        let poller = Arc::clone(&fake);
        let poll_cancel = cancel.clone();
        let handle = thread::spawn(move || poller.poll(&poll_cancel));
        thread::sleep(Duration::from_millis(5));
        cancel.shutdown();
        let err = handle.join().expect("poll thread").expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn concurrent_push_and_poll() {
        let fake = Arc::new(FakeS2s::new(Duration::from_millis(5)));
        let cancel = Cancel::new();
        let expected = frame(9, 11);
        let poller = Arc::clone(&fake);
        let poll_cancel = cancel.clone();
        let want = expected.clone();
        let handle = thread::spawn(move || audio(poller.poll(&poll_cancel).expect("poll")));
        fake.push(expected, &cancel).expect("push");
        assert_eq!(handle.join().expect("join"), want);
    }

    #[test]
    fn event_variants_are_distinct() {
        let pcm = frame(0, 3);
        let audio = S2sEvent::Audio(pcm.clone());
        let text = S2sEvent::Text(String::from("hi"));
        let err = S2sEvent::Error {
            message: String::from("boom"),
        };
        assert_ne!(audio, text);
        assert_ne!(text, err);
        assert!(format!("{audio:?}").contains("Audio"));
        assert!(format!("{text:?}").contains("hi"));
        assert!(format!("{err:?}").contains("boom"));
        assert_eq!(audio, S2sEvent::Audio(pcm));
    }
}
