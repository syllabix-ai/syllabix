//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! Pull request 13 runs Silero → whisper.cpp → llama.cpp → Kokoro → fixture
//! playback. Fake providers remain in the 30-turn in-memory tests. Native
//! inference tests share one binary (`native_inference`) and skip weight loads
//! under `cargo llvm-cov` (`cfg(coverage)`). Sequence 22 ships a `dist`
//! profile executable; model weights stay in the first-run cache. Sequence 23
//! publishes those files as a GitHub Release.

pub mod audio;
pub mod models;

mod cancel;
mod config;
mod defaults;
mod dist;
mod error;
mod fake;
mod g2p;
mod language;
mod live;
mod llm;
mod memory;
mod openai;
mod pipeline;
mod providers;
mod queue;
mod real;
mod speech_text;
mod stt;
mod tts;
mod turn_debug;
mod types;
mod vad;

pub use cancel::Cancel;
pub use config::{AgentConfig, CONFIG_FILE_NAME};
pub use defaults::{
    BuiltinDefaults, LlmProvider, QueueCaps, SttModel, SttProvider, TtsModel, TtsProvider,
    VadProvider,
};
pub use dist::{
    artifact_name_for_target, artifact_name_for_uname, format_sha256sums_line,
    latest_release_download_url, parse_sha256sums, sha256sums_contains, ARTIFACT_FILE_NAMES,
    DIST_PROFILE, DIST_TARGETS, MAX_DIST_BINARY_BYTES, RELEASE_LATEST_DOWNLOAD_PREFIX,
    SHA256SUMS_FILE_NAME,
};
pub use error::{Error, Result};
pub use fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeStt,
    FakeTts, FakeVad, LlmCall, ScriptedStt,
};
pub use g2p::{english_to_ipa, english_to_kokoro_ids, KOKORO_MAX_PHONEME_TOKENS};
pub use live::run_live;
pub use llm::{
    is_v0_llm_model, system_prompt_for, LlamaLlm, LLAMA_32_1B_ASSET, LLAMA_CANCEL_TIMEOUT,
    LLAMA_MAX_HISTORY_TURNS, QWEN35_08B_ASSET, QWEN35_2B_ASSET, VOICE_SYSTEM_PROMPT,
};
pub use memory::{process_rss_bytes, LOOP_RSS_GROWTH_CEILING_BYTES};
pub use models::{
    cache_root, format_progress_line, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset,
    ModelCache, ModelLayer, NoProgress, Progress, StderrProgress,
};
pub use openai::{
    join_endpoint, resolve_api_key, validate_base_url, OpenAiLlm, OpenAiSettings, API_KEY_ENV,
    CLOUD_FALLBACK_TEXT, DEFAULT_LLM_BASE_URL,
};
pub use pipeline::{
    is_blank_stt, run_loop, run_loop_captured, LoopConfig, LoopEvent, LoopMode, LoopReport,
    PipelineStages,
};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use real::{build_llm, load_real_providers, LiveLlm};
pub use speech_text::{
    speak_text_for_tts, strip_markdown_for_speech, strip_think_for_speech, take_sentences,
    ThinkFilter,
};
pub use stt::{
    contains_words_in_order, transcript_words, word_match_ratio, WhisperStt,
    LIBRISPEECH_MIN_WORD_MATCH, STT_LANGUAGE,
};
pub use tts::{
    KokoroTts, QwenTts, KOKORO_ASSET, KOKORO_NATIVE_RATE_HZ, KOKORO_VOICE_ASSET,
    QWEN_TTS_06B_ASSET, QWEN_TTS_06B_MMPROJ_ASSET, QWEN_TTS_ASSET, QWEN_TTS_MMPROJ_ASSET,
    TTS_ASR_MIN_WORD_MATCH,
};
pub use turn_debug::{
    prepare_turn_debug_dir, resolve_turn_debug_dir, TurnDebug, TurnOutcome, DEFAULT_TURN_DEBUG_DIR,
    TURN_DEBUG_DIR_ENV,
};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, LlmDebugMeta, SynthesizedAudio,
    TokenChunk, Transcript, TurnId, TurnTimings, Utterance, VadEvent, DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
pub use vad::{
    SileroVad, VadSettings, END_SILENCE, END_SILENCE_FRAMES, FRAME_DURATION, MIN_SPEECH,
    MIN_SPEECH_FRAMES, PREROLL_SAMPLES, SPEECH_THRESHOLD, WHISPER_PREROLL,
};

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
        #[cfg(not(target_vendor = "apple"))]
        {
            assert_eq!(syllabix_native::llama_n_gpu_layers(), 0);
            assert!(!syllabix_native::whisper_use_gpu());
        }
        #[cfg(target_vendor = "apple")]
        {
            assert_eq!(syllabix_native::llama_n_gpu_layers(), -1);
            assert!(syllabix_native::whisper_use_gpu());
        }
    }
}
