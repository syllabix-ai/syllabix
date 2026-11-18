//! Turn-based speech-to-speech engine trait.

use crate::cancel::Cancel;
use crate::error::Result;
use crate::types::{SynthesizedAudio, Utterance};

/// One VAD-committed user utterance in, streamed assistant PCM out.
pub trait S2sEngine: Send {
    /// Config / backend name.
    fn name(&self) -> &'static str;

    /// Stream assistant audio for one user utterance.
    ///
    /// `on_audio` is invoked in chunk order. The last invocation must have `is_last = true`.
    /// Honor `cancel.is_shutdown()` and `cancel.is_stale(generation)`.
    fn generate(
        &mut self,
        utterance: &Utterance,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::cancel::Cancel;
    use crate::fake::FakeS2s;
    use crate::types::{TurnId, Utterance};

    use super::S2sEngine;

    #[test]
    fn trait_object_accepts_fake_s2s() {
        let engine: Box<dyn S2sEngine> = Box::new(FakeS2s::new());
        assert_eq!(engine.name(), "fake-s2s");
    }

    #[test]
    fn generate_through_trait_echoes_pcm() {
        let mut engine: Box<dyn S2sEngine> = Box::new(FakeS2s::new());
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let utterance = Utterance {
            turn: TurnId(3),
            frames: crate::fake::scripted_frames(1, 1, 0),
        };
        let expected = utterance.pcm();
        let mut chunks = Vec::new();
        engine
            .generate(&utterance, &cancel, &mut |audio| {
                chunks.push(audio);
                Ok(())
            })
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples, expected);
        assert!(chunks[0].is_last);
        assert_eq!(chunks[0].turn, TurnId(3));
        assert_eq!(chunks[0].generation, generation);
    }

    #[test]
    fn delayed_generate_is_cancellable_mid_wait() {
        let mut engine = FakeS2s::with_delay(Duration::from_millis(200));
        let log = engine.call_log();
        let cancel = Arc::new(Cancel::new());
        let cancel_worker = Arc::clone(&cancel);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            cancel_worker.cancel_generation();
        });
        let utterance = Utterance {
            turn: TurnId(0),
            frames: crate::fake::scripted_frames(1, 1, 0),
        };
        let err = engine
            .generate(&utterance, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::Cancelled));
        assert!(log.lock().unwrap().contains(&TurnId(0)));
    }
}
