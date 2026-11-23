//! Turn-based speech-to-speech engine trait.

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::types::{GenerationId, SynthesizedAudio, TurnId, Utterance};

/// Manifest / yaml id for LiquidAI LFM3-Audio-2B (`pipeline.s2s.model`).
pub const LFM3_AUDIO_2B: &str = "lfm3-audio-2b";

/// Manifest id for the LFM3-Audio multimodal projector GGUF.
pub const LFM3_AUDIO_2B_MMPROJ: &str = "lfm3-audio-2b-mmproj";

/// Voice-chat system prompt from the [LFM3-Audio-2B model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
/// (Voice chat: speech and text responses). It is not stored in the GGUF metadata.
pub const LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT: &str = "Respond with interleaved text and audio.";

/// Decoded into the KV cache on barge-in before end-of-turn. Must match `SYLLABIX_S2S_INTERRUPT_MARKER_TEXT` in `shim_s2s.c`.
pub const S2S_INTERRUPT_MARKER: &str = "(interrupted)";

/// One VAD-committed user utterance in, streamed assistant PCM out.
#[cfg(all(feature = "ggml", any(feature = "lfm3-s2s", coverage)))]
pub use crate::s2s_lfm3::Lfm3AudioEngine;

pub trait S2sEngine: Send {
    /// Config / backend name.
    fn name(&self) -> &'static str;

    /// Stream assistant audio for one user utterance.
    ///
    /// `on_audio` is invoked in chunk order. The last invocation must have `is_last = true`.
    /// `on_text` streams UTF-8 pieces in order during generation; `is_last = true` only when
    /// the turn's text is finished successfully, not on cancel.
    /// Honor `cancel.is_shutdown()` and `cancel.is_stale(generation)`.
    fn generate(
        &mut self,
        utterance: &Utterance,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
        on_text: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()>;

    /// Text sampled during the last `generate`, including a cancelled partial turn, when exposed.
    fn last_text(&self) -> Option<&str> {
        None
    }

    /// Offline ASR for one utterance ([`crate::config::S2sReplyWith::Llm`]).
    fn transcribe(
        &mut self,
        _utterance: &Utterance,
        _language: Option<&str>,
        _cancel: &Cancel,
    ) -> Result<String> {
        Err(Error::Provider {
            provider: self.name(),
            message: "does not support offline ASR".into(),
        })
    }

    /// TTS for one text segment ([`crate::config::S2sReplyWith::Llm`]).
    ///
    /// `on_audio` is invoked in chunk order at `PcmFormat::v0()`. The last chunk must have
    /// `is_last = true` only when the segment finishes successfully, not on cancel (same
    /// contract as `generate`'s `on_audio`).
    #[allow(clippy::too_many_arguments)]
    fn speak(
        &mut self,
        _turn: TurnId,
        _generation: GenerationId,
        _text: &str,
        _language: Option<&str>,
        _voice: Option<&str>,
        _cancel: &Cancel,
        _on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        Err(Error::Provider {
            provider: self.name(),
            message: "does not support offline TTS".into(),
        })
    }

    /// PCM sample cap one `speak` call enforces, in `PcmFormat::v0()` samples
    /// as delivered to `on_audio` (not native model-rate samples), when the
    /// backend has one.
    fn speak_sample_cap(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::cancel::Cancel;
    use crate::fake::{scripted_frames, FakeS2s};
    use crate::types::{SynthesizedAudio, TurnId, Utterance};

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
            .generate(
                &utterance,
                &cancel,
                &mut |audio| {
                    chunks.push(audio);
                    Ok(())
                },
                &mut |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples, expected);
        assert!(chunks[0].is_last);
        assert_eq!(chunks[0].turn, TurnId(3));
        assert_eq!(chunks[0].generation, generation);
    }

    #[test]
    fn default_transcribe_and_speak_are_provider_errors() {
        struct StubS2s;
        impl S2sEngine for StubS2s {
            fn name(&self) -> &'static str {
                "stub-s2s"
            }
            fn generate(
                &mut self,
                utterance: &Utterance,
                cancel: &Cancel,
                on_audio: &mut dyn FnMut(SynthesizedAudio) -> crate::error::Result<()>,
                on_text: &mut dyn FnMut(&str, bool) -> crate::error::Result<()>,
            ) -> crate::error::Result<()> {
                FakeS2s::new().generate(utterance, cancel, on_audio, on_text)
            }
        }
        let mut engine: Box<dyn S2sEngine> = Box::new(StubS2s);
        let utterance = Utterance {
            turn: TurnId(0),
            frames: scripted_frames(0, 1, 0),
        };
        let err = engine
            .transcribe(&utterance, None, &Cancel::new())
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::Provider { .. }));
        let err = engine
            .speak(
                TurnId(0),
                Cancel::new().generation(),
                "hi",
                None,
                None,
                &Cancel::new(),
                &mut |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::Provider { .. }));
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
            .generate(&utterance, &cancel, &mut |_| Ok(()), &mut |_, _| Ok(()))
            .unwrap_err();
        assert!(matches!(err, crate::error::Error::Cancelled));
        assert!(log.lock().unwrap().contains(&TurnId(0)));
    }
}
