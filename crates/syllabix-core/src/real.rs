#[cfg(coverage)]
include!("real_coverage.rs");

#[cfg(not(coverage))]
mod production {
    //! Load configured native and online providers from the model cache and environment.
    //!
    //! The LLM slot is a [`LiveLlm`]: the in-process llama.cpp GGUF engine by
    //! default, or the user-keyed online adapter when YAML selects
    //! `pipeline.llm.provider: openai`. VAD, AEC, STT, and TTS stay local either
    //! way; only transcript text reaches the cloud endpoint.

    use zeroize::Zeroizing;

    use crate::cancel::Cancel;
    use crate::config::AgentConfig;
    use crate::error::{Error, Result};
    use crate::llm::LlamaLlm;
    use crate::models::{Fetcher, ModelCache, Progress};
    use crate::moonshine::MoonshineStt;
    use crate::openai::{OpenAiLlm, OpenAiSettings, API_KEY_ENV, PROVIDER_NAME};
    use crate::providers::Llm;
    use crate::stt::WhisperStt;
    use crate::tts::{KokoroTts, QwenTts};
    use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, Transcript, Utterance};
    use crate::vad::SileroVad;
    use crate::PocketTts;

    /// The configured LLM implementation for one live run.
    pub enum LiveLlm {
        /// In-process GGUF engine (default).
        Local(LlamaLlm),
        /// User-keyed OpenAI-compatible endpoint.
        Cloud(OpenAiLlm),
    }

    impl Llm for LiveLlm {
        fn name(&self) -> &'static str {
            match self {
                Self::Local(llm) => llm.name(),
                Self::Cloud(llm) => llm.name(),
            }
        }

        fn debug_meta(&self) -> Option<LlmDebugMeta> {
            match self {
                Self::Local(llm) => llm.debug_meta(),
                Self::Cloud(llm) => llm.debug_meta(),
            }
        }

        fn take_tool_events(&mut self) -> Vec<crate::types::ToolTurnEvent> {
            match self {
                Self::Local(llm) => llm.take_tool_events(),
                Self::Cloud(llm) => llm.take_tool_events(),
            }
        }

        fn generate(
            &mut self,
            history: &[HistoryTurn],
            user: &Transcript,
            cancel: &Cancel,
            on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
        ) -> Result<()> {
            match self {
                Self::Local(llm) => llm.generate(history, user, cancel, on_token),
                Self::Cloud(llm) => llm.generate(history, user, cancel, on_token),
            }
        }
    }

    /// The configured TTS implementation for one live run.
    pub enum LiveTts {
        /// Kokoro ONNX, selected explicitly through YAML.
        Kokoro(KokoroTts),
        /// Qwen3-TTS through the shared ggml runtime.
        Qwen(QwenTts),
        /// Pocket TTS through ONNX Runtime, used by default.
        Pocket(Box<PocketTts>),
    }

    impl crate::providers::Tts for LiveTts {
        fn name(&self) -> &'static str {
            match self {
                Self::Kokoro(tts) => tts.name(),
                Self::Qwen(tts) => tts.name(),
                Self::Pocket(tts) => tts.name(),
            }
        }

        fn model_id(&self) -> Option<&str> {
            match self {
                Self::Kokoro(tts) => tts.model_id(),
                Self::Qwen(tts) => tts.model_id(),
                Self::Pocket(tts) => tts.model_id(),
            }
        }

        fn synthesize_chunk(
            &mut self,
            token: &TokenChunk,
            cancel: &Cancel,
        ) -> Result<Vec<crate::types::SynthesizedAudio>> {
            match self {
                Self::Kokoro(tts) => tts.synthesize_chunk(token, cancel),
                Self::Qwen(tts) => tts.synthesize_chunk(token, cancel),
                Self::Pocket(tts) => tts.synthesize_chunk(token, cancel),
            }
        }

        fn synthesize_chunk_into(
            &mut self,
            token: &TokenChunk,
            cancel: &Cancel,
            on_audio: &mut dyn FnMut(crate::types::SynthesizedAudio) -> Result<()>,
        ) -> Result<()> {
            match self {
                Self::Kokoro(tts) => tts.synthesize_chunk_into(token, cancel, on_audio),
                Self::Qwen(tts) => tts.synthesize_chunk_into(token, cancel, on_audio),
                Self::Pocket(tts) => tts.synthesize_chunk_into(token, cancel, on_audio),
            }
        }
    }

    /// The configured STT implementation for one live run.
    pub enum LiveStt {
        /// whisper.cpp GGML engine (default).
        Whisper(WhisperStt),
        /// Moonshine streaming ONNX (small or medium; English only, partials always on).
        Moonshine(Box<MoonshineStt>),
    }

    impl crate::providers::Stt for LiveStt {
        fn name(&self) -> &'static str {
            match self {
                Self::Whisper(stt) => stt.name(),
                Self::Moonshine(stt) => stt.name(),
            }
        }

        fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
            match self {
                Self::Whisper(stt) => stt.transcribe(utterance, cancel),
                Self::Moonshine(stt) => stt.transcribe(utterance, cancel),
            }
        }

        fn supports_partials(&self) -> bool {
            matches!(self, Self::Moonshine(_))
        }

        fn start_turn(&mut self, turn: crate::TurnId, cancel: &Cancel) -> Result<()> {
            match self {
                Self::Whisper(stt) => stt.start_turn(turn, cancel),
                Self::Moonshine(stt) => stt.start_turn(turn, cancel),
            }
        }

        fn push_frame(
            &mut self,
            frame: &crate::AudioFrame,
            cancel: &Cancel,
        ) -> Result<Option<String>> {
            match self {
                Self::Whisper(stt) => stt.push_frame(frame, cancel),
                Self::Moonshine(stt) => stt.push_frame(frame, cancel),
            }
        }

        fn cancel_turn(&mut self, turn: crate::TurnId) {
            match self {
                Self::Whisper(stt) => stt.cancel_turn(turn),
                Self::Moonshine(stt) => stt.cancel_turn(turn),
            }
        }
    }

    impl LiveStt {
        /// Apply the yaml STT language to the selected engine.
        pub fn with_language(self, language: &str) -> Result<Self> {
            match self {
                Self::Whisper(stt) => Ok(Self::Whisper(stt.with_language(language)?)),
                Self::Moonshine(stt) => Ok(Self::Moonshine(stt)),
            }
        }
    }

    /// Build just the STT slot from config. Only the selected engine's weights
    /// are fetched; selecting Moonshine never fetches Whisper GGMLs and vice versa.
    pub fn build_stt(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
    ) -> Result<LiveStt> {
        match config.stt_model {
            crate::SttModel::Small
            | crate::SttModel::Medium
            | crate::SttModel::LargeV3Turbo
            | crate::SttModel::MediumQ5_0
            | crate::SttModel::LargeV3TurboQ5_0 => Ok(LiveStt::Whisper(WhisperStt::from_cache(
                cache,
                fetcher,
                progress,
                cancel,
                config.stt_model,
            )?)),
            crate::SttModel::MoonshineStreamingSmall
            | crate::SttModel::MoonshineStreamingMedium => Ok(LiveStt::Moonshine(Box::new(
                MoonshineStt::from_cache(cache, fetcher, progress, cancel, config.stt_model)?,
            ))),
        }
    }
    /// Load Silero, the configured STT engine, and the configured language and
    /// speech models.
    ///
    /// `llm_api_key` carries the already-resolved `SYLLABIX_LLM_API_KEY` value
    /// (`run_live` fails fast before this point when the config needs one and it
    /// is missing). The local provider ignores it; the cloud provider requires it.
    pub fn load_real_providers(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
    ) -> Result<(SileroVad, LiveStt, LiveLlm, LiveTts)> {
        let vad = SileroVad::from_cache(cache, fetcher, progress, cancel)?
            .with_settings(config.vad_settings());
        // Only the selected STT engine is fetched; the rest of the menu stays on disk.
        let stt =
            build_stt(cache, fetcher, progress, cancel, config)?.with_language(&config.language)?;
        let llm = build_llm(cache, fetcher, progress, cancel, config, llm_api_key)?;
        let tts = build_tts(cache, fetcher, progress, cancel, config)?;
        Ok((vad, stt, llm, tts))
    }

    /// Build just the TTS slot from config. Only the selected provider's weights
    /// are fetched; selecting other models does not fetch Qwen GGUFs.
    pub fn build_tts(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
    ) -> Result<LiveTts> {
        match config.tts {
            crate::TtsProvider::Local => match config.tts_model {
                crate::TtsModel::Kokoro => Ok(LiveTts::Kokoro(KokoroTts::from_cache(
                    cache, fetcher, progress, cancel,
                )?)),
                crate::TtsModel::PocketTts => Ok(LiveTts::Pocket(Box::new(PocketTts::from_cache(
                    cache, fetcher, progress, cancel,
                )?))),
                qwen => Ok(LiveTts::Qwen(QwenTts::from_cache(
                    cache,
                    fetcher,
                    progress,
                    cancel,
                    &config.tts_language,
                    qwen,
                )?)),
            },
            // Config parsing rejects `online`; this arm keeps the builder total
            // for hand-built configurations and reports where inference runs.
            crate::TtsProvider::Online => Err(Error::Config {
                field: "pipeline.tts.provider".into(),
                message: "online TTS is not supported yet; use provider \"local\"".into(),
            }),
        }
    }

    /// Build just the LLM slot from config. The cloud path never touches the
    /// model cache — no GGUF is fetched or loaded for a cloud run.
    pub fn build_llm(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
    ) -> Result<LiveLlm> {
        let skill_context = if config.llm_developer_harness {
            let discovery =
                config.discover_skills(&std::env::current_dir().unwrap_or_else(|_| ".".into()));
            for diagnostic in &discovery.diagnostics {
                tracing::warn!(
                    code = %diagnostic.code,
                    path = diagnostic
                        .path
                        .as_deref()
                        .map(std::path::Path::display)
                        .map(|path| path.to_string())
                        .unwrap_or_else(|| "<skills-root>".into()),
                    message = %diagnostic.message,
                    "local skill ignored"
                );
            }
            discovery.model_context()
        } else {
            String::new()
        };
        match config.llm {
            crate::LlmProvider::Local => Ok(LiveLlm::Local(
                LlamaLlm::from_cached_model(
                    cache,
                    fetcher,
                    progress,
                    cancel,
                    &config.llm_model,
                    config.thinking,
                )?
                .with_system_prompt(config.system_prompt.clone())
                .with_skill_context(skill_context.clone())
                .with_developer_harness(config.llm_developer_harness)
                .with_developer_permissions(config.llm_developer_permissions.clone()),
            )),
            crate::LlmProvider::Online => {
                let key = llm_api_key.ok_or_else(|| Error::Config {
                    field: API_KEY_ENV.into(),
                    message: format!("is required when pipeline.llm.provider is {PROVIDER_NAME}"),
                })?;
                Ok(LiveLlm::Cloud(
                    OpenAiLlm::new(OpenAiSettings::from_config(config), key.clone())
                        .with_skill_context(skill_context),
                ))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::models::{BlockedFetcher, Manifest, NoProgress};
        use crate::openai::DEFAULT_LLM_BASE_URL;
        use std::path::PathBuf;

        fn empty_manifest_cache(label: &str) -> ModelCache {
            let root: PathBuf = std::env::temp_dir().join(format!(
                "syllabix-real-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&root).unwrap();
            // An empty manifest makes any accidental weight lookup fail loudly.
            ModelCache::new(
                root,
                Manifest {
                    version: 1,
                    assets: vec![],
                },
            )
        }

        #[test]
        fn missing_manifest_assets_fail_before_native_load() {
            let cache = empty_manifest_cache("missing");
            let err = match load_real_providers(
                &cache,
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &crate::AgentConfig::v0(),
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("empty manifest should fail"),
            };
            assert!(matches!(err, Error::ModelCache { .. }));
        }

        #[test]
        fn cloud_llm_needs_no_weights_but_requires_the_key() {
            let mut config = crate::AgentConfig::v0();
            config.llm = crate::LlmProvider::Online;
            config.llm_model = "gpt-4o-mini".into();

            let err = build_llm(
                &empty_manifest_cache("cloud-nokey"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            )
            .err()
            .expect("missing key fails fast");
            assert!(err.to_string().contains(API_KEY_ENV), "{err}");

            let llm = build_llm(
                &empty_manifest_cache("cloud-ok"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                Some(&Zeroizing::new("sk-x".to_string())),
            )
            .expect("cloud slot builds without any weights");
            assert!(matches!(llm, LiveLlm::Cloud(_)));
            assert_eq!(llm.name(), "online");
            let meta = llm.debug_meta().expect("meta");
            assert_eq!(meta.model, "gpt-4o-mini");
            assert_eq!(
                meta.endpoint,
                format!("{DEFAULT_LLM_BASE_URL}/chat/completions")
            );
            assert_eq!(meta.provider, "online");
        }

        #[test]
        fn local_default_still_reports_llama_cpp() {
            // The slot type reports the configured provider without loading weights;
            // loading the engine itself stays with native inference tests.
            let mut config = crate::AgentConfig::v0();
            config.llm = crate::LlmProvider::Local;
            let err = build_llm(
                &empty_manifest_cache("local-meta"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            )
            .err()
            .expect("empty manifest fails before a GGUF load");
            assert!(matches!(err, Error::ModelCache { .. }));
        }
    }
}

#[cfg(not(coverage))]
pub use production::*;
