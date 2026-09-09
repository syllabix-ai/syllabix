//! Shared types and the in-memory conversation loop for Syllabix.
//!
//! The runtime connects Silero, whisper.cpp, llama.cpp, and text-to-speech
//! providers through bounded queues. Native inference tests share one binary
//! so large model weights load only once per test process. `SYLLABIX_NATIVE_MODELS`
//! selects specific model suites, while `cfg(coverage)` replaces live audio
//! loops with lightweight fakes. Model weights live in the first-run cache
//! rather than the executable.

pub mod audio;
pub mod models;

mod cancel;
mod config;
mod defaults;
mod dist;
mod error;
mod executor;
mod fake;
mod g2p;
mod language;
mod live;
mod llm;
mod memory;
// Moonshine's production adapter dynamically loads three ONNX graphs and a
// tokenizer. Coverage tests inject scripted sessions instead of downloading
// weights, so the real adapter always compiles.
mod moonshine;
mod onnx;
mod openai;
mod pipeline;
mod policy;
// Pocket TTS native inference runs in the model suite; coverage tests inject
// scripted sessions, so the real adapter always compiles.
mod pocket_tts;
mod providers;
mod queue;
mod real;
mod sandbox;
mod skills;
mod speech_text;
mod stt;
mod tts;
mod turn_debug;
mod types;
mod vad;

pub use cancel::Cancel;
pub use config::{
    AgentConfig, DeveloperPermissions, FilesystemMode, NetworkMode, SecretPolicy, CONFIG_FILE_NAME,
    DEFAULT_AUTO_TIMEOUT_EXIT_MS, DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS,
};
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
pub use executor::{
    execute_skill, execute_with_permissions_and_skills, ShellRequest, SkillRequest, ValidatedCall,
};
pub use fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeStt,
    FakeTts, FakeVad, LlmCall, ScriptedStt,
};
pub use g2p::{english_to_ipa, english_to_kokoro_ids, KOKORO_MAX_PHONEME_TOKENS};
pub use live::{run_live, run_live_with_controls};
pub use llm::{
    is_v0_llm_model, render_system_prompt, render_system_prompt_with_skills, system_prompt_for,
    LlamaLlm, LFM25_230M_ASSET, LFM25_2_6B_ASSET, LFM25_350M_ASSET, LFM_TOOL_TURN_MAX_CHARS,
    LLAMA_32_1B_ASSET, LLAMA_CANCEL_TIMEOUT, LLAMA_MAX_HISTORY_TURNS, LOCAL_TOOL_FALLBACK_TEXT,
    QWEN35_08B_ASSET, QWEN35_2B_ASSET, V0_LLM_MODELS, VOICE_SYSTEM_PROMPT,
    VOICE_SYSTEM_PROMPT_TEMPLATE,
};
pub use memory::{process_rss_bytes, LOOP_RSS_GROWTH_CEILING_BYTES};
pub use models::{
    cache_root, format_progress_line, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset,
    ModelCache, ModelLayer, NoProgress, Progress, StderrProgress,
};
pub use moonshine::{
    MoonshineStt, MoonshineTokenizer, DECODER_ASSET as MOONSHINE_DECODER_ASSET,
    DECODER_PAST_ASSET as MOONSHINE_DECODER_PAST_ASSET, ENCODER_ASSET as MOONSHINE_ENCODER_ASSET,
    MEDIUM_DECODER_ASSET as MOONSHINE_MEDIUM_DECODER_ASSET,
    MEDIUM_DECODER_PAST_ASSET as MOONSHINE_MEDIUM_DECODER_PAST_ASSET,
    MEDIUM_ENCODER_ASSET as MOONSHINE_MEDIUM_ENCODER_ASSET,
    MEDIUM_TOKENIZER_ASSET as MOONSHINE_MEDIUM_TOKENIZER_ASSET,
    TOKENIZER_ASSET as MOONSHINE_TOKENIZER_ASSET,
};
pub use openai::{
    join_endpoint, resolve_api_key, validate_base_url, OpenAiLlm, OpenAiSettings, API_KEY_ENV,
    CLOUD_FALLBACK_TEXT, DEFAULT_LLM_BASE_URL, MAX_TOOL_CALLS_PER_TURN, TOOL_LIMIT_TEXT,
};
pub use pipeline::{
    is_blank_stt, run_loop, run_loop_captured, AutoTimeoutAction, IdleClock, LoopConfig, LoopEvent,
    LoopMode, LoopReport, PipelineStages, RuntimeControls,
};
pub use pocket_tts::{
    PocketTts, POCKET_TTS_BOS_ASSET, POCKET_TTS_BUNDLE_ASSET, POCKET_TTS_FLOW_ASSET,
    POCKET_TTS_FLOW_MAIN_ASSET, POCKET_TTS_MIMI_DECODER_ASSET, POCKET_TTS_TEXT_CONDITIONER_ASSET,
    POCKET_TTS_TOKENIZER_ASSET, POCKET_TTS_VOICE_ASSET,
};
pub use policy::{resolve_effective, Enforcement, ExecutionPlan, PolicyDeny, Provenance};
pub use providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
pub use queue::{bounded, BoundedReceiver, BoundedSender, Occupancy, QueueReport, QueueStats};
pub use real::{build_llm, build_stt, build_tts, load_real_providers, LiveLlm, LiveStt, LiveTts};
#[cfg(all(target_os = "macos", not(coverage)))]
pub use sandbox::MacOsSandboxProvider;
pub use sandbox::{
    current_provider, SandboxError, SandboxProvider, SandboxRequest, UnavailableSandboxProvider,
};
#[cfg(all(target_os = "linux", not(coverage)))]
pub use sandbox::{
    BubblewrapSandboxProvider, LandlockSandboxProvider, LinuxSandboxProvider, LinuxSandboxRunner,
};
pub use skills::{
    resolve_entrypoint_argv, validate_inputs, DiscoveredSkill, SkillDiagnostic, SkillDiscovery,
    SkillEntrypoint, SkillInput, SkillManifest, SkillPermissions, SkillPin, SkillRoot, SkillSource,
    SkillsConfig, DEFAULT_SKILL_TIMEOUT, MAX_SKILL_ARGV_VALUES, MAX_SKILL_ARGV_VALUE_BYTES,
    MAX_SKILL_DESCRIPTION_BYTES, MAX_SKILL_FILE_BYTES, MAX_SKILL_TIMEOUT_SECONDS,
};
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
    prepare_turn_debug_dir, PlaybackWatch, TimelineAnchor, TurnDebug, TurnOutcome,
    DEFAULT_TURN_DEBUG_DIR,
};
pub use types::{
    AudioFrame, CompletedTurn, GenerationId, HistoryTurn, LlmDebugMeta, SynthesizedAudio,
    TokenChunk, ToolCall, ToolResult, ToolTurnEvent, Transcript, TurnId, TurnTimings, Utterance,
    VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
pub use vad::{
    SileroVad, VadSettings, END_SILENCE, END_SILENCE_FRAMES, FRAME_DURATION, MIN_SPEECH,
    MIN_SPEECH_FRAMES, PREROLL_SAMPLES, SPEECH_THRESHOLD, WHISPER_PREROLL,
};

/// Force-link the whisper.cpp and llama.cpp frontends into the executable.
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
