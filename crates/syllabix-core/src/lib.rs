//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! Pull request 6 adds a versioned model manifest and a checksummed on-disk
//! cache. Fake providers still stand in for VAD/STT/LLM/TTS.

pub mod audio;
pub mod models;

mod cancel;
mod defaults;
mod error;
mod fake;
mod pipeline;
mod providers;
mod queue;
mod types;
mod vad;

pub use cancel::Cancel;
pub use defaults::{
    BuiltinDefaults, LlmProvider, QueueCaps, SttModel, SttProvider, TtsProvider, VadProvider,
};
pub use error::{Error, Result};
pub use fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad, LlmCall};
pub use models::{
    cache_root, format_progress_line, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset,
    ModelCache, ModelLayer, NoProgress, Progress, StderrProgress,
};
pub use pipeline::{run_loop, LoopConfig, LoopMode, LoopReport, PipelineStages};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript,
    TurnId, Utterance, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
pub use vad::{SileroVad, END_SILENCE_FRAMES, SPEECH_THRESHOLD};
