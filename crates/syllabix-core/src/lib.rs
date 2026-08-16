//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! Pull request 13 runs Silero → whisper.cpp → llama.cpp → Kokoro → fixture
//! playback. Fake providers remain in the 30-turn in-memory tests. Native
//! inference tests share one binary (`native_inference`) and skip weight loads
//! under `cargo llvm-cov` (`cfg(coverage)`).

pub mod audio;
pub mod models;

mod cancel;
mod defaults;
mod error;
mod fake;
mod g2p;
mod llm;
mod memory;
mod pipeline;
mod providers;
mod queue;
mod real;
mod speech_text;
mod stt;
mod tts;
mod types;
mod vad;

pub use cancel::Cancel;
pub use defaults::{
    BuiltinDefaults, LlmProvider, QueueCaps, SttModel, SttProvider, TtsProvider, VadProvider,
};
pub use error::{Error, Result};
pub use fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeStt,
    FakeTts, FakeVad, LlmCall,
};
pub use g2p::{english_to_ipa, english_to_kokoro_ids, KOKORO_MAX_PHONEME_TOKENS};
pub use llm::{
    LlamaLlm, LLAMA_1B_ASSET, LLAMA_CANCEL_TIMEOUT, LLAMA_MAX_HISTORY_TURNS, LLAMA_N_CTX,
    LLAMA_N_PREDICT, VOICE_SYSTEM_PROMPT,
};
pub use memory::{process_rss_bytes, LOOP_RSS_GROWTH_CEILING_BYTES};
pub use models::{
    cache_root, format_progress_line, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset,
    ModelCache, ModelLayer, NoProgress, Progress, StderrProgress,
};
pub use pipeline::{run_loop, run_loop_captured, LoopConfig, LoopMode, LoopReport, PipelineStages};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use real::load_real_providers;
pub use speech_text::{strip_markdown_for_speech, take_sentences};
pub use stt::{
    contains_words_in_order, transcript_words, word_match_ratio, WhisperStt,
    LIBRISPEECH_MIN_WORD_MATCH, STT_LANGUAGE, WHISPER_SMALL_ASSET,
};
pub use tts::{
    KokoroTts, KOKORO_ASSET, KOKORO_NATIVE_RATE_HZ, KOKORO_VOICE_ASSET, TTS_ASR_MIN_WORD_MATCH,
};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript,
    TurnId, Utterance, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
pub use vad::{SileroVad, END_SILENCE_FRAMES, SPEECH_THRESHOLD};

/// Force-link both native frontends into `syllabix` (PR 10).
pub fn ensure_shared_ggml_frontends() -> bool {
    syllabix_native::frontends_linked()
}

#[cfg(test)]
mod shared_ggml_tests {
    #[test]
    fn llama_and_whisper_frontends_share_one_ggml() {
        assert!(super::ensure_shared_ggml_frontends());
        let info = syllabix_native::llama_system_info();
        assert!(!info.is_empty(), "llama.cpp frontend must be linked");
    }
}
