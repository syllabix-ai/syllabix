//! Interfaces for pipeline stages and audio devices, with in-memory implementations for tests.

use crate::cancel::Cancel;
use crate::error::Result;
use crate::types::{
    AudioFrame, HistoryTurn, LlmDebugMeta, SynthesizedAudio, TokenChunk, ToolTurnEvent, Transcript,
    TurnId, Utterance, VadEvent,
};

/// Voice-activity detector. Consumes frames, emits speech-start and speech-stop.
pub trait Vad: Send {
    /// Config name (`silero`).
    fn name(&self) -> &'static str;

    /// Push one frame. May emit zero, one, or two events (start+end is unusual but allowed).
    fn push_frame(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>>;

    /// End of stream: close an in-flight utterance if the last frames were speech.
    fn flush(&mut self) -> Result<Vec<VadEvent>>;
}

/// Speech-to-text for a completed VAD utterance.
pub trait Stt: Send {
    /// Config name (`whisper.cpp`).
    fn name(&self) -> &'static str;

    /// Transcribe one utterance. Must check `cancel` and return [`crate::Error::Cancelled`].
    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript>;

    /// Whether this engine can produce live text while VAD owns an active
    /// turn. The default keeps the established utterance-final Whisper path.
    fn supports_partials(&self) -> bool {
        false
    }

    /// VAD opened `turn`. This is notification only: VAD remains the single
    /// owner of start/end timing and the shared `end_silence_ms` setting.
    fn start_turn(&mut self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
        Ok(())
    }

    /// One post-AEC frame from an active VAD turn. Returning text updates the
    /// provisional transcript; returning `None` means no visible change.
    fn push_frame(&mut self, _frame: &AudioFrame, _cancel: &Cancel) -> Result<Option<String>> {
        Ok(None)
    }

    /// Forget provisional state for a turn abandoned by cancellation.
    fn cancel_turn(&mut self, _turn: TurnId) {}
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::types::{AudioFrame, GenerationId, Utterance};

    struct StubStt;
    impl Stt for StubStt {
        fn name(&self) -> &'static str {
            "stub-stt"
        }

        fn transcribe(&mut self, _utterance: &Utterance, _cancel: &Cancel) -> Result<Transcript> {
            Ok(Transcript {
                turn: TurnId(1),
                text: String::from("hello"),
                language: String::from("en"),
            })
        }
    }

    struct StubLlm;
    impl Llm for StubLlm {
        fn name(&self) -> &'static str {
            "stub-llm"
        }

        fn generate(
            &mut self,
            _history: &[HistoryTurn],
            _user: &Transcript,
            _cancel: &Cancel,
            _on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
        ) -> Result<()> {
            Ok(())
        }
    }

    struct StubTts;
    impl Tts for StubTts {
        fn name(&self) -> &'static str {
            "stub-tts"
        }

        fn synthesize_chunk(
            &mut self,
            _token: &TokenChunk,
            _cancel: &Cancel,
        ) -> Result<Vec<SynthesizedAudio>> {
            Ok(Vec::new())
        }
    }

    struct StubSink {
        played: usize,
    }
    impl AudioSink for StubSink {
        fn play(&mut self, _audio: SynthesizedAudio, _cancel: &Cancel) -> Result<()> {
            self.played += 1;
            Ok(())
        }
    }

    fn frame() -> AudioFrame {
        AudioFrame {
            seq: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0; 256],
            capture_pcm: None,
        }
    }

    fn token() -> TokenChunk {
        TokenChunk {
            turn: TurnId(1),
            generation: GenerationId(1),
            index: 0,
            text: String::from("hi"),
            is_last: true,
        }
    }

    #[test]
    fn stt_defaults_keep_utterance_final_path() {
        let mut stt = StubStt;
        let cancel = Cancel::new();
        assert_eq!(stt.name(), "stub-stt");
        assert!(!stt.supports_partials());
        stt.start_turn(TurnId(1), &cancel).expect("start_turn");
        assert_eq!(stt.push_frame(&frame(), &cancel).expect("push"), None);
        let transcript = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: vec![frame()],
                },
                &cancel,
            )
            .expect("transcribe");
        assert_eq!(transcript.text, "hello");
        stt.cancel_turn(TurnId(1));
    }

    #[test]
    fn llm_and_tts_defaults_report_no_live_meta() {
        let llm = StubLlm;
        assert_eq!(llm.name(), "stub-llm");
        assert!(llm.debug_meta().is_none());
        let mut llm = llm;
        assert!(llm.take_tool_events().is_empty());
        llm.generate(
            &[],
            &Transcript {
                turn: TurnId(1),
                text: String::from("hello"),
                language: String::from("en"),
            },
            &Cancel::new(),
            &mut |_| Ok(()),
        )
        .expect("generate");

        let tts = StubTts;
        assert_eq!(tts.name(), "stub-tts");
        assert!(tts.model_id().is_none());
    }

    #[test]
    fn tts_chunk_into_fans_out_to_callback() {
        let mut tts = StubTts;
        let cancel = Cancel::new();
        let token = token();
        let mut seen = 0;
        tts.synthesize_chunk_into(&token, &cancel, &mut |_| {
            seen += 1;
            Ok(())
        })
        .expect("fan-out");
        assert_eq!(seen, 0);
    }

    #[test]
    fn sink_defaults_finish_and_interrupt_silently() {
        let mut sink = StubSink { played: 0 };
        let cancel = Cancel::new();
        sink.play(
            SynthesizedAudio {
                turn: TurnId(7),
                generation: GenerationId(1),
                index: 0,
                samples: vec![0],
                is_last: true,
            },
            &cancel,
        )
        .expect("play");
        sink.finish_turn(TurnId(7), &cancel).expect("finish");
        sink.interrupt();
        assert_eq!(sink.played, 1);
    }
}

/// Streaming language model.
pub trait Llm: Send {
    /// Config name (`llama.cpp`, `openai`).
    fn name(&self) -> &'static str;

    /// Provider facts for diagnostics sidecars. `None` means the fake and
    /// test providers; live engines report provider/model/endpoint/request-id.
    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        None
    }

    /// Structured API-tool evidence accumulated during the last generation.
    /// Local and existing fake providers have none.
    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        Vec::new()
    }

    /// Stream tokens for `user`. `history` is completed prior turns in order.
    ///
    /// `on_token` is invoked in index order. The last invocation must have `is_last = true`.
    fn generate(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()>;
}

/// Streaming text-to-speech.
pub trait Tts: Send {
    /// Config name (`kokoro`).
    fn name(&self) -> &'static str;

    /// Weight id for diagnostics sidecars. `None` means the fake and test
    /// providers; live engines report the loaded asset id.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Synthesize buffered tokens. May return no chunks until a sentence ends.
    ///
    /// Pipeline calls this per streamed token. The last chunk of a generation
    /// must set `is_last = true` so the loop can complete the turn.
    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>>;

    /// Streaming form of [`Self::synthesize_chunk`]. Existing providers keep
    /// their buffered implementation; native Qwen overrides this to release
    /// vocoder windows while generation is still running.
    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        for audio in self.synthesize_chunk(token, cancel)? {
            on_audio(audio)?;
        }
        Ok(())
    }
}

/// Playback destination implemented by native speakers or an in-memory collector.
pub trait AudioSink: Send {
    /// Play or collect one chunk. Must check `cancel`.
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()>;

    /// Wait until a completed turn has drained from the output device.
    ///
    /// Fixture sinks consume synchronously, so their default implementation is
    /// immediate. Native playback waits for the first silent callback after
    /// the final samples were submitted. The pipeline uses this boundary
    /// before default-mode VAD resumes, preventing the speaker tail from
    /// becoming the next user turn.
    fn finish_turn(&mut self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
        Ok(())
    }

    /// Duck/stop queued playback. Default is a no-op for collecting sinks.
    fn interrupt(&mut self) {}
}

/// Microphone or fixture source yielding pipeline PCM frames.
pub trait AudioCapture: Send {
    /// Config / backend name (`cpal`, `fixture`).
    fn name(&self) -> &'static str;

    /// Next 16 kHz mono frame, or `Ok(None)` at end of a fixture.
    ///
    /// Must return [`crate::Error::Cancelled`] when `cancel` is shut down.
    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>>;
}
