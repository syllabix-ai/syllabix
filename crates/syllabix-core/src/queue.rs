//! Bounded queues with occupancy statistics. Senders block instead of growing without limit.

use std::sync::mpsc::{self, RecvError, RecvTimeoutError, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::error::{Error, Result};

/// Occupancy snapshot for one queue. High-water must stay `<= capacity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy {
    /// Queue name (`frames`, `utterances`, …).
    pub name: &'static str,
    /// Declared bound.
    pub capacity: usize,
    /// Reserved slots (in the channel or about to enter it).
    pub current: usize,
    /// Max `current` observed.
    pub high_water: usize,
    /// Successful sends into the channel.
    pub sent: usize,
    /// Successful receives from the channel.
    pub received: usize,
}

impl Occupancy {
    /// True when the implementation never exceeded the bound.
    pub fn within_capacity(&self) -> bool {
        self.high_water <= self.capacity && self.current <= self.capacity
    }
}

#[derive(Debug, Default)]
struct Counts {
    current: usize,
    high_water: usize,
    sent: usize,
    received: usize,
}

/// Live counters shared by both ends of a bounded queue.
#[derive(Debug)]
pub struct QueueStats {
    name: &'static str,
    capacity: usize,
    counts: Mutex<Counts>,
}

impl QueueStats {
    fn new(name: &'static str, capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            name,
            capacity,
            counts: Mutex::new(Counts::default()),
        })
    }

    /// Take a slot before the matching channel send so occupancy cannot race past `capacity`.
    fn try_reserve(&self) -> bool {
        let mut counts = self.counts.lock().expect("queue stats");
        if counts.current >= self.capacity {
            return false;
        }
        counts.current += 1;
        counts.high_water = counts.high_water.max(counts.current);
        true
    }

    fn release_slot(&self) {
        let mut counts = self.counts.lock().expect("queue stats");
        counts.current = counts.current.saturating_sub(1);
    }

    fn note_sent(&self) {
        self.counts.lock().expect("queue stats").sent += 1;
    }

    fn note_recv(&self) {
        let mut counts = self.counts.lock().expect("queue stats");
        counts.received += 1;
        counts.current = counts.current.saturating_sub(1);
    }

    /// Point-in-time occupancy.
    pub fn snapshot(&self) -> Occupancy {
        let counts = self.counts.lock().expect("queue stats");
        Occupancy {
            name: self.name,
            capacity: self.capacity,
            current: counts.current,
            high_water: counts.high_water,
            sent: counts.sent,
            received: counts.received,
        }
    }
}

/// Sending end. Clone to share among producers of the same stage.
#[derive(Clone)]
pub struct BoundedSender<T> {
    inner: mpsc::SyncSender<T>,
    stats: Arc<QueueStats>,
    stage: &'static str,
}

/// Receiving end. Not cloneable; one consumer per stage.
pub struct BoundedReceiver<T> {
    inner: mpsc::Receiver<T>,
    stats: Arc<QueueStats>,
}

/// Create a bounded queue. `capacity` must be at least 1.
pub fn bounded<T>(
    name: &'static str,
    capacity: usize,
) -> (BoundedSender<T>, BoundedReceiver<T>, Arc<QueueStats>) {
    assert!(capacity > 0, "queue {name} capacity must be > 0");
    let (tx, rx) = mpsc::sync_channel(capacity);
    let stats = QueueStats::new(name, capacity);
    (
        BoundedSender {
            inner: tx,
            stats: Arc::clone(&stats),
            stage: name,
        },
        BoundedReceiver {
            inner: rx,
            stats: Arc::clone(&stats),
        },
        stats,
    )
}

impl<T> BoundedSender<T> {
    fn reserve_slot(&self, cancel: Option<&Cancel>) -> Result<()> {
        loop {
            if cancel.is_some_and(Cancel::is_shutdown) {
                return Err(Error::Cancelled);
            }
            if self.stats.try_reserve() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn finish_send(&self, item: T) -> Result<()> {
        match self.inner.send(item) {
            Ok(()) => {
                self.stats.note_sent();
                Ok(())
            }
            Err(_) => {
                self.stats.release_slot();
                Err(Error::Disconnected { stage: self.stage })
            }
        }
    }

    /// Blocking send. Prefer [`send_cancellable`](Self::send_cancellable) in workers.
    pub fn send(&self, item: T) -> Result<()> {
        self.reserve_slot(None)?;
        self.finish_send(item)
    }

    /// Non-blocking send.
    pub fn try_send(&self, item: T) -> std::result::Result<(), TrySendError<T>> {
        if !self.stats.try_reserve() {
            return Err(TrySendError::Full(item));
        }
        match self.inner.try_send(item) {
            Ok(()) => {
                self.stats.note_sent();
                Ok(())
            }
            Err(err) => {
                self.stats.release_slot();
                Err(err)
            }
        }
    }

    /// Send with backpressure, aborting on shutdown.
    pub fn send_cancellable(&self, mut item: T, cancel: &Cancel) -> Result<()> {
        self.reserve_slot(Some(cancel))?;
        loop {
            if cancel.is_shutdown() {
                self.stats.release_slot();
                return Err(Error::Cancelled);
            }
            match self.inner.try_send(item) {
                Ok(()) => {
                    self.stats.note_sent();
                    return Ok(());
                }
                Err(TrySendError::Full(returned)) => {
                    item = returned;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.stats.release_slot();
                    return Err(Error::Disconnected { stage: self.stage });
                }
            }
        }
    }
}

impl<T> BoundedReceiver<T> {
    /// Blocking receive.
    pub fn recv(&self) -> std::result::Result<T, RecvError> {
        let item = self.inner.recv()?;
        self.stats.note_recv();
        Ok(item)
    }

    /// Receive with timeout so workers can poll [`Cancel`].
    pub fn recv_timeout(&self, timeout: Duration) -> std::result::Result<T, RecvTimeoutError> {
        let item = self.inner.recv_timeout(timeout)?;
        self.stats.note_recv();
        Ok(item)
    }

    /// Non-blocking receive.
    pub fn try_recv(&self) -> std::result::Result<T, TryRecvError> {
        let item = self.inner.try_recv()?;
        self.stats.note_recv();
        Ok(item)
    }

    /// Receive until an item arrives, the producer disconnects, or shutdown is set.
    pub fn recv_cancellable(&self, cancel: &Cancel) -> Result<Option<T>> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            match self.recv_timeout(Duration::from_millis(5)) {
                Ok(item) => return Ok(Some(item)),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Ok(None),
            }
        }
    }
}

/// Occupancy for every pipeline queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueReport {
    /// Capture frames into VAD.
    pub frames: Occupancy,
    /// VAD utterances into STT.
    pub utterances: Occupancy,
    /// Transcripts into LLM.
    pub transcripts: Occupancy,
    /// LLM tokens into TTS.
    pub tokens: Occupancy,
    /// TTS chunks into playback.
    pub audio: Occupancy,
}

impl QueueReport {
    /// True when every queue respected its bound.
    pub fn within_capacity(&self) -> bool {
        self.frames.within_capacity()
            && self.utterances.within_capacity()
            && self.transcripts.within_capacity()
            && self.tokens.within_capacity()
            && self.audio.within_capacity()
    }

    /// Occupancy snapshots in stable order for assertions.
    pub fn all(&self) -> [Occupancy; 5] {
        [
            self.frames,
            self.utterances,
            self.transcripts,
            self.tokens,
            self.audio,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn occupancy_never_exceeds_capacity() {
        let (tx, rx, stats) = bounded::<u32>("frames", 2);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        assert!(tx.try_send(3).is_err());
        let snap = stats.snapshot();
        assert_eq!(snap.high_water, 2);
        assert!(snap.within_capacity());
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        let snap = stats.snapshot();
        assert_eq!(snap.current, 0);
        assert_eq!(snap.sent, 2);
        assert_eq!(snap.received, 2);
    }

    #[test]
    fn occupancy_stays_within_cap_under_concurrent_send_recv() {
        let (tx, rx, stats) = bounded::<u32>("audio", 4);
        let producer = thread::spawn(move || {
            for i in 0..200 {
                tx.send(i).unwrap();
            }
        });
        let consumer = thread::spawn(move || {
            for _ in 0..200 {
                rx.recv().unwrap();
            }
        });
        producer.join().expect("producer");
        consumer.join().expect("consumer");
        let snap = stats.snapshot();
        assert!(
            snap.within_capacity(),
            "occupancy raced past the bound: {snap:?}"
        );
        assert_eq!(snap.current, 0);
        assert_eq!(snap.sent, 200);
        assert_eq!(snap.received, 200);
    }

    #[test]
    fn send_cancellable_stops_on_shutdown() {
        let (tx, _rx, _) = bounded::<u32>("tokens", 1);
        tx.send(1).unwrap();
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = tx.send_cancellable(2, &cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }
}
