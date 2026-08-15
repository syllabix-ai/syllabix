//! Pipeline shutdown and generation cancellation.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::types::GenerationId;

/// Shared cancel state for every pipeline stage.
#[derive(Debug, Clone)]
pub struct Cancel {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    shutdown: AtomicBool,
    generation: AtomicU64,
}

impl Cancel {
    /// Fresh cancel state at generation 0, not shut down.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                shutdown: AtomicBool::new(false),
                generation: AtomicU64::new(0),
            }),
        }
    }

    /// Stop the whole loop. Stages must exit even if queues still have work.
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
    }

    /// True after [`shutdown`](Self::shutdown).
    pub fn is_shutdown(&self) -> bool {
        self.inner.shutdown.load(Ordering::SeqCst)
    }

    /// Current assistant generation. Stages stamp work with this id.
    pub fn generation(&self) -> GenerationId {
        GenerationId(self.inner.generation.load(Ordering::SeqCst))
    }

    /// Drop queued LLM/TTS audio for the current generation and keep the user turn path open.
    pub fn cancel_generation(&self) -> GenerationId {
        let next = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        GenerationId(next)
    }

    /// True when `generation` is no longer the live assistant generation, or the loop is stopping.
    pub fn is_stale(&self, generation: GenerationId) -> bool {
        self.is_shutdown() || self.generation() != generation
    }
}

impl Default for Cancel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_is_visible_to_clones() {
        let cancel = Cancel::new();
        let clone = cancel.clone();
        assert!(!cancel.is_shutdown());
        clone.shutdown();
        assert!(cancel.is_shutdown());
        assert!(cancel.is_stale(GenerationId(0)));
    }

    #[test]
    fn cancel_generation_invalidates_prior_id() {
        let cancel = Cancel::new();
        let first = cancel.generation();
        assert_eq!(first, GenerationId(0));
        let next = cancel.cancel_generation();
        assert_eq!(next, GenerationId(1));
        assert!(cancel.is_stale(first));
        assert!(!cancel.is_stale(next));
    }
}
