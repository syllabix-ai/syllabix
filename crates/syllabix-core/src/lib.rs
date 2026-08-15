//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! Pull request 2 lands runtime contracts and a fake VAD → STT → LLM → TTS
//! cascade. Native audio and real model adapters replace the fakes in later PRs.

mod cancel;
mod defaults;
mod error;
mod fake;
mod pipeline;
mod providers;
mod queue;
mod types;

pub use cancel::Cancel;
pub use defaults::{
    BuiltinDefaults, LlmProvider, QueueCaps, SttModel, SttProvider, TtsProvider, VadProvider,
};
pub use error::{Error, Result};
pub use fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad, LlmCall};
pub use pipeline::{run_loop, LoopConfig, LoopMode, LoopReport, PipelineStages};
pub use providers::{AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript,
    TurnId, Utterance, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
