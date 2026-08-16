//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! Pull request 9 adds the in-process llama.cpp GGUF adapter. Fake LLM
//! remains available for the 30-turn in-memory pipeline tests until the
//! all-real loop lands.

pub mod audio;
pub mod models;

mod cancel;
mod defaults;
mod error;
mod fake;
mod llm;
mod pipeline;
mod providers;
mod queue;
mod stt;
mod types;
mod vad;

pub use cancel::Cancel;
pub use defaults::{
    BuiltinDefaults, LlmModel, LlmProvider, QueueCaps, SttModel, SttProvider, TtsProvider,
    VadProvider,
};
pub use error::{Error, Result};
pub use fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad, LlmCall};
pub use llm::{
    format_llama3_prompt, rolling_chat, ChatTurn, LlamaLlm, CANCEL_TIMEOUT, CONTEXT_TOKENS,
    LLM_ASSET, MAX_HISTORY_TURNS, MAX_NEW_TOKENS, SAMPLE_SEED, VOICE_SYSTEM_PROMPT,
};
pub use models::{
    cache_root, format_progress_line, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset,
    ModelCache, ModelLayer, NoProgress, Progress, StderrProgress,
};
pub use pipeline::{run_loop, LoopConfig, LoopMode, LoopReport, PipelineStages};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use stt::{
    contains_words_in_order, transcript_words, word_match_ratio, WhisperStt,
    LIBRISPEECH_MIN_WORD_MATCH, STT_LANGUAGE, WHISPER_SMALL_ASSET,
};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript,
    TurnId, Utterance, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
pub use vad::{SileroVad, END_SILENCE_FRAMES, SPEECH_THRESHOLD};
