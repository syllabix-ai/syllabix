//! Provider traits. v0 has one live implementation per layer; fakes stand in until those land.

use crate::cancel::Cancel;
use crate::error::Result;
use crate::types::{
    AudioFrame, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, Utterance, VadEvent,
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
}

/// Streaming language model.
pub trait Llm: Send {
    /// Config name (`llama.cpp`).
    fn name(&self) -> &'static str;

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

    /// Synthesize `text` as one audio chunk. Pipeline calls this per streamed token.
    fn synthesize_chunk(&mut self, token: &TokenChunk, cancel: &Cancel)
        -> Result<SynthesizedAudio>;
}

/// Playback sink. Native speakers replace the collecting sink in a later PR.
pub trait AudioSink: Send {
    /// Play or collect one chunk. Must check `cancel`.
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()>;
}

/// Microphone (or fixture) source. Yields v0 PCM frames.
pub trait AudioCapture: Send {
    /// Config / backend name (`cpal`, `fixture`).
    fn name(&self) -> &'static str;

    /// Next 16 kHz mono frame, or `Ok(None)` at end of a fixture.
    ///
    /// Must return [`crate::Error::Cancelled`] when `cancel` is shut down.
    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>>;
}
