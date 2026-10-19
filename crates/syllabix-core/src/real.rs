#[cfg(coverage)]
include!("real_coverage.rs");

#[cfg(all(not(coverage), feature = "ggml"))]
mod production {
    //! Load configured native and online providers from the model cache and environment.
    //!
    //! The LLM slot is a [`LiveLlm`]: the in-process llama.cpp GGUF engine by
    //! default, or the user-keyed online adapter when YAML selects
    //! `pipeline.llm.provider: online`. STT can be local or `provider: online`
    //! ([`crate::remote_stt::RemoteStt`]) and TTS can be local or `provider: online`
    //! ([`crate::remote_tts::RemoteTts`]). VAD and AEC stay local. When STT
    //! is online, utterance WAV leaves the machine and no STT weights are fetched.
    //! When TTS is online, reply text leaves the machine and no TTS weights are fetched.

    use zeroize::Zeroizing;

    use crate::cancel::Cancel;
    use crate::config::AgentConfig;
    use crate::error::{Error, Result};
    #[cfg(feature = "cloud")]
    use crate::goose::{skill_file_list, GooseAgent, GooseStart};
    use crate::llm::LlamaLlm;
    use crate::models::{Fetcher, ModelCache, Progress};
    use crate::moonshine::MoonshineStt;
    #[cfg(feature = "cloud")]
    use crate::openai::{OpenAiLlm, OpenAiSettings, API_KEY_ENV, PROVIDER_NAME};
    use crate::pocket_tts::PocketTts;
    use crate::providers::Llm;
    #[cfg(feature = "cloud")]
    use crate::remote_stt::{RemoteStt, RemoteSttSettings};
    #[cfg(feature = "cloud")]
    use crate::remote_tts::{RemoteTts, RemoteTtsSettings};
    use crate::stt::WhisperStt;
    use crate::tts::{KokoroTts, QwenTts};
    use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, Transcript, Utterance};
    use crate::vad::SileroVad;

    /// The configured LLM implementation for one live run.
    pub enum LiveLlm {
        /// In-process GGUF engine (default).
        Local(LlamaLlm),
        /// User-keyed OpenAI-compatible endpoint.
        #[cfg(feature = "cloud")]
        Cloud(OpenAiLlm),
        /// Goose runs the reply (`developer_harness.agent: goose`).
        #[cfg(feature = "cloud")]
        Goose(GooseAgent),
    }

    impl Llm for LiveLlm {
        fn name(&self) -> &'static str {
            match self {
                Self::Local(llm) => llm.name(),
                #[cfg(feature = "cloud")]
                Self::Cloud(llm) => llm.name(),
                #[cfg(feature = "cloud")]
                Self::Goose(llm) => llm.name(),
            }
        }

        fn debug_meta(&self) -> Option<LlmDebugMeta> {
            match self {
                Self::Local(llm) => llm.debug_meta(),
                #[cfg(feature = "cloud")]
                Self::Cloud(llm) => llm.debug_meta(),
                #[cfg(feature = "cloud")]
                Self::Goose(llm) => llm.debug_meta(),
            }
        }

        fn take_tool_events(&mut self) -> Vec<crate::types::ToolTurnEvent> {
            match self {
                Self::Local(llm) => llm.take_tool_events(),
                #[cfg(feature = "cloud")]
                Self::Cloud(llm) => llm.take_tool_events(),
                #[cfg(feature = "cloud")]
                Self::Goose(llm) => llm.take_tool_events(),
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
                #[cfg(feature = "cloud")]
                Self::Cloud(llm) => llm.generate(history, user, cancel, on_token),
                #[cfg(feature = "cloud")]
                Self::Goose(llm) => llm.generate(history, user, cancel, on_token),
            }
        }
    }

    impl LiveLlm {
        /// Distinct ACP `toolCallId`s observed in the Goose session, or `None`
        /// for the other backends. The manual harness-quality run uses this with
        /// [`Self::goose_terminal_create_count`] to prove every Goose call
        /// reached Syllabix (a sandboxed `terminal/create` or an MCP web tool).
        /// Test/evidence seam; not frozen API.
        #[cfg(feature = "cloud")]
        pub fn goose_acp_tool_call_ids(&self) -> Option<Vec<String>> {
            match self {
                Self::Goose(llm) => Some(llm.acp_tool_call_ids()),
                _ => None,
            }
        }

        /// `terminal/create` requests answered by the Goose session, or `None`
        /// for the other backends. Test/evidence seam; not frozen API.
        #[cfg(feature = "cloud")]
        pub fn goose_terminal_create_count(&self) -> Option<u64> {
            match self {
                Self::Goose(llm) => Some(llm.terminal_create_count()),
                _ => None,
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
        /// OpenAI-compatible speech endpoint. No local TTS weights.
        #[cfg(feature = "cloud")]
        Remote(RemoteTts),
    }

    impl LiveTts {
        /// Compute backend selected by Qwen's startup probe. Other TTS engines
        /// do not have an adaptive placement decision to report.
        pub fn backend_id(&self) -> Option<&str> {
            match self {
                Self::Qwen(tts) => tts.backend_id(),
                Self::Kokoro(_) | Self::Pocket(_) => None,
                #[cfg(feature = "cloud")]
                Self::Remote(_) => None,
            }
        }
    }

    impl crate::providers::Tts for LiveTts {
        fn name(&self) -> &'static str {
            match self {
                Self::Kokoro(tts) => tts.name(),
                Self::Qwen(tts) => tts.name(),
                Self::Pocket(tts) => tts.name(),
                #[cfg(feature = "cloud")]
                Self::Remote(tts) => tts.name(),
            }
        }

        fn model_id(&self) -> Option<&str> {
            match self {
                Self::Kokoro(tts) => tts.model_id(),
                Self::Qwen(tts) => tts.model_id(),
                Self::Pocket(tts) => tts.model_id(),
                #[cfg(feature = "cloud")]
                Self::Remote(tts) => tts.model_id(),
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
                #[cfg(feature = "cloud")]
                Self::Remote(tts) => tts.synthesize_chunk(token, cancel),
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
                #[cfg(feature = "cloud")]
                Self::Remote(tts) => tts.synthesize_chunk_into(token, cancel, on_audio),
            }
        }
    }

    /// The configured STT implementation for one live run.
    pub enum LiveStt {
        /// whisper.cpp GGML engine (default).
        Whisper(WhisperStt),
        /// Moonshine streaming ONNX (small or medium; English only, partials always on).
        Moonshine(Box<MoonshineStt>),
        /// Qwen3-ASR 0.6B through the shared ggml (finalize only).
        Qwen(Box<crate::qwen_asr::QwenAsrStt>),
        /// OpenAI-compatible transcriptions endpoint. No local STT weights.
        #[cfg(feature = "cloud")]
        Remote(RemoteStt),
    }

    impl crate::providers::Stt for LiveStt {
        fn name(&self) -> &'static str {
            match self {
                Self::Whisper(stt) => stt.name(),
                Self::Moonshine(stt) => stt.name(),
                Self::Qwen(stt) => stt.name(),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => stt.name(),
            }
        }

        fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
            match self {
                Self::Whisper(stt) => stt.transcribe(utterance, cancel),
                Self::Moonshine(stt) => stt.transcribe(utterance, cancel),
                Self::Qwen(stt) => stt.transcribe(utterance, cancel),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => stt.transcribe(utterance, cancel),
            }
        }

        fn supports_partials(&self) -> bool {
            matches!(self, Self::Moonshine(_))
        }

        fn start_turn(&mut self, turn: crate::TurnId, cancel: &Cancel) -> Result<()> {
            match self {
                Self::Whisper(stt) => stt.start_turn(turn, cancel),
                Self::Moonshine(stt) => stt.start_turn(turn, cancel),
                Self::Qwen(stt) => stt.start_turn(turn, cancel),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => stt.start_turn(turn, cancel),
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
                Self::Qwen(stt) => stt.push_frame(frame, cancel),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => stt.push_frame(frame, cancel),
            }
        }

        fn cancel_turn(&mut self, turn: crate::TurnId) {
            match self {
                Self::Whisper(stt) => stt.cancel_turn(turn),
                Self::Moonshine(stt) => stt.cancel_turn(turn),
                Self::Qwen(stt) => stt.cancel_turn(turn),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => stt.cancel_turn(turn),
            }
        }
    }

    impl LiveStt {
        /// Compute backend confirmed after load for Qwen ASR (`cpu` / `metal` /
        /// `vulkan`). Whisper and Moonshine return `None` (bench uses the
        /// fingerprint ggml backend instead).
        pub fn backend_id(&self) -> Option<&str> {
            match self {
                Self::Qwen(stt) => stt.backend_id(),
                Self::Whisper(_) | Self::Moonshine(_) => None,
                #[cfg(feature = "cloud")]
                Self::Remote(_) => None,
            }
        }

        /// Apply the yaml STT language to the selected engine.
        pub fn with_language(self, language: &str) -> Result<Self> {
            match self {
                Self::Whisper(stt) => Ok(Self::Whisper(stt.with_language(language)?)),
                Self::Moonshine(stt) => Ok(Self::Moonshine(stt)),
                Self::Qwen(stt) => Ok(Self::Qwen(Box::new(stt.with_language(language)?))),
                #[cfg(feature = "cloud")]
                Self::Remote(stt) => Ok(Self::Remote(stt.with_language(language))),
            }
        }
    }

    /// Build just the STT slot from config. Only the selected engine's weights
    /// are fetched; selecting Moonshine never fetches Whisper GGMLs and vice versa.
    /// Online STT never touches the model cache.
    pub fn build_stt(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        stt_api_key: Option<&Zeroizing<String>>,
    ) -> Result<LiveStt> {
        match config.stt {
            crate::defaults::SttProvider::Local => match config.stt_model {
                crate::defaults::SttModel::Small
                | crate::defaults::SttModel::Medium
                | crate::defaults::SttModel::LargeV3Turbo
                | crate::defaults::SttModel::MediumQ5_0
                | crate::defaults::SttModel::LargeV3TurboQ5_0 => Ok(LiveStt::Whisper(
                    WhisperStt::from_cache(cache, fetcher, progress, cancel, config.stt_model)?,
                )),
                crate::defaults::SttModel::MoonshineStreamingSmall
                | crate::defaults::SttModel::MoonshineStreamingMedium => {
                    Ok(LiveStt::Moonshine(Box::new(MoonshineStt::from_cache(
                        cache,
                        fetcher,
                        progress,
                        cancel,
                        config.stt_model,
                    )?)))
                }
                crate::defaults::SttModel::QwenAsr06 => Ok(LiveStt::Qwen(Box::new(
                    crate::qwen_asr::QwenAsrStt::from_cache(
                        cache,
                        fetcher,
                        progress,
                        cancel,
                        config.stt_model,
                        config.stt_compute,
                    )?,
                ))),
            },
            crate::defaults::SttProvider::Online => {
                #[cfg(feature = "cloud")]
                {
                    let _ = (cache, fetcher, progress, cancel);
                    let key = stt_api_key.ok_or_else(|| Error::Config {
                        field: crate::config::STT_API_KEY_ENV.into(),
                        message: format!(
                            "is required when pipeline.stt.provider is {}",
                            crate::remote_stt::PROVIDER_NAME
                        ),
                    })?;
                    Ok(LiveStt::Remote(RemoteStt::new(
                        RemoteSttSettings::from_config(config)?,
                        key.clone(),
                    )))
                }
                #[cfg(not(feature = "cloud"))]
                {
                    let _ = (cache, fetcher, progress, cancel, stt_api_key);
                    Err(Error::Config {
                        field: "pipeline.stt.provider".into(),
                        message: "online STT requires the `cloud` feature".into(),
                    })
                }
            }
        }
    }
    /// Load Silero, the configured STT engine, and the configured language and
    /// speech models.
    ///
    /// `llm_api_key` / `stt_api_key` / `tts_api_key` carry already-resolved
    /// environment keys (`run_live` fails fast before this point when the config
    /// needs one and it is missing). Local providers ignore them; online
    /// providers require them. Online STT and TTS never touch the model cache.
    #[allow(clippy::too_many_arguments)]
    pub fn load_real_providers(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
        stt_api_key: Option<&Zeroizing<String>>,
        tts_api_key: Option<&Zeroizing<String>>,
    ) -> Result<(SileroVad, LiveStt, LiveLlm, LiveTts)> {
        let vad = SileroVad::from_cache(cache, fetcher, progress, cancel)?
            .with_settings(config.vad_settings());
        // Only the selected local STT engine is fetched; online STT skips the cache.
        let stt = build_stt(cache, fetcher, progress, cancel, config, stt_api_key)?
            .with_language(&config.language)?;
        let llm = build_llm(cache, fetcher, progress, cancel, config, llm_api_key)?;
        let tts = build_tts(cache, fetcher, progress, cancel, config, tts_api_key)?;
        #[cfg(feature = "cloud")]
        if let LiveLlm::Cloud(llm) = &llm {
            match llm.warm_up(cancel) {
                Ok(()) => eprintln!("cloud: endpoint ready"),
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    tracing::warn!(provider = PROVIDER_NAME, error = %err, "cloud warm-up failed");
                    eprintln!(
                        "cloud: could not verify endpoint; continuing — the first reply may fail"
                    );
                }
            }
        }
        #[cfg(feature = "cloud")]
        if let LiveStt::Remote(stt) = &stt {
            match stt.warm_up(cancel) {
                Ok(()) => eprintln!("cloud: STT endpoint ready"),
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    tracing::warn!(
                        provider = crate::remote_stt::PROVIDER_NAME,
                        error = %err,
                        "cloud STT warm-up failed"
                    );
                    eprintln!(
                        "cloud: could not verify STT endpoint; continuing — the first transcript may fail"
                    );
                }
            }
        }
        #[cfg(feature = "cloud")]
        if let LiveTts::Remote(tts) = &tts {
            match tts.warm_up(cancel) {
                Ok(()) => eprintln!("cloud: TTS endpoint ready"),
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    tracing::warn!(
                        provider = crate::remote_tts::PROVIDER_NAME,
                        error = %err,
                        "cloud TTS warm-up failed"
                    );
                    eprintln!(
                        "cloud: could not verify TTS endpoint; continuing — the first spoken reply may fail"
                    );
                }
            }
        }
        Ok((vad, stt, llm, tts))
    }

    /// Build just the TTS slot from config. Only the selected provider's weights
    /// are fetched; selecting other models does not fetch Qwen GGUFs.
    /// Online TTS never touches the model cache.
    pub fn build_tts(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        tts_api_key: Option<&Zeroizing<String>>,
    ) -> Result<LiveTts> {
        match config.tts {
            crate::defaults::TtsProvider::Local => match config.tts_model {
                crate::defaults::TtsModel::Kokoro => Ok(LiveTts::Kokoro(KokoroTts::from_cache(
                    cache, fetcher, progress, cancel,
                )?)),
                crate::defaults::TtsModel::PocketTts => Ok(LiveTts::Pocket(Box::new(
                    PocketTts::from_cache(cache, fetcher, progress, cancel)?,
                ))),
                qwen => Ok(LiveTts::Qwen(QwenTts::from_cache(
                    cache,
                    fetcher,
                    progress,
                    cancel,
                    &config.tts_language,
                    qwen,
                    config.tts_compute,
                )?)),
            },
            crate::defaults::TtsProvider::Online => {
                #[cfg(feature = "cloud")]
                {
                    let _ = (cache, fetcher, progress, cancel);
                    let key = tts_api_key.ok_or_else(|| Error::Config {
                        field: crate::config::TTS_API_KEY_ENV.into(),
                        message: format!(
                            "is required when pipeline.tts.provider is {}",
                            crate::remote_tts::PROVIDER_NAME
                        ),
                    })?;
                    Ok(LiveTts::Remote(RemoteTts::new(
                        RemoteTtsSettings::from_config(config)?,
                        key.clone(),
                    )))
                }
                #[cfg(not(feature = "cloud"))]
                {
                    let _ = (cache, fetcher, progress, cancel, tts_api_key);
                    Err(Error::Config {
                        field: "pipeline.tts.provider".into(),
                        message: "online TTS requires the `cloud` feature".into(),
                    })
                }
            }
        }
    }

    /// Build just the LLM slot from config. The cloud path never touches the
    /// model cache — no GGUF is fetched or loaded for a cloud run.
    /// `developer_harness.agent: goose` is online-only: after the pinned Goose CLI is on
    /// disk, this starts the Goose backend for replies.
    pub fn build_llm(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
    ) -> Result<LiveLlm> {
        build_llm_at(cache, fetcher, progress, cancel, config, llm_api_key, None)
    }

    /// [`build_llm`] with an explicit workspace. `None` runs against the
    /// current directory, matching every host path. The manual Goose
    /// harness-quality run passes each fixture's own workspace so a harness
    /// write can never touch the source checkout. Test/evidence seam; not
    /// frozen API.
    pub fn build_llm_at(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
        workspace: Option<std::path::PathBuf>,
    ) -> Result<LiveLlm> {
        let workspace = workspace.unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
        });
        if config.agent == crate::defaults::AgentBackend::Goose {
            if config.llm != crate::defaults::LlmProvider::Online {
                return Err(Error::Config {
                    field: "developer_harness.agent".into(),
                    message: "goose requires pipeline.llm.provider: online".into(),
                });
            }
            #[cfg(feature = "cloud")]
            {
                let key = llm_api_key.ok_or_else(|| Error::Config {
                    field: API_KEY_ENV.into(),
                    message: format!("is required when pipeline.llm.provider is {PROVIDER_NAME}"),
                })?;
                let binary = crate::goose::resolve_goose_exe(cache, fetcher, progress, cancel)?;
                let discovery = config.discover_skills(&workspace);
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
                return Ok(LiveLlm::Goose(GooseAgent::start(GooseStart {
                    binary,
                    cache: crate::goose::goose_path_root(cache),
                    config: config.clone(),
                    api_key: key.clone(),
                    workspace: workspace.clone(),
                    skill_list: skill_file_list(&discovery),
                })?));
            }
            #[cfg(not(feature = "cloud"))]
            {
                let _ = (cache, fetcher, progress, cancel, llm_api_key);
                return Err(Error::Config {
                    field: "developer_harness.agent".into(),
                    message: "goose requires the `cloud` feature".into(),
                });
            }
        }
        let skill_context = if config.llm_developer_harness {
            let discovery = config.discover_skills(&workspace);
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
            crate::defaults::LlmProvider::Local => Ok(LiveLlm::Local(
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
                .with_developer_permissions(config.llm_developer_permissions.clone())
                .with_workspace(workspace.clone()),
            )),
            crate::defaults::LlmProvider::Online => {
                #[cfg(feature = "cloud")]
                {
                    let key = llm_api_key.ok_or_else(|| Error::Config {
                        field: API_KEY_ENV.into(),
                        message: format!(
                            "is required when pipeline.llm.provider is {PROVIDER_NAME}"
                        ),
                    })?;
                    Ok(LiveLlm::Cloud(
                        OpenAiLlm::new(OpenAiSettings::from_config(config), key.clone())
                            .with_skill_context(skill_context)
                            .with_workspace(workspace.clone()),
                    ))
                }
                #[cfg(not(feature = "cloud"))]
                {
                    let _ = (llm_api_key, skill_context);
                    Err(Error::Config {
                        field: "pipeline.llm.provider".into(),
                        message: "online LLM requires the `cloud` feature".into(),
                    })
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::models::{BlockedFetcher, Manifest, NoProgress};
        #[cfg(feature = "cloud")]
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
                None,
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("empty manifest should fail"),
            };
            assert!(matches!(err, Error::ModelCache { .. }));
        }

        #[cfg(feature = "cloud")]
        #[test]
        fn cloud_needs_no_weights_but_requires_the_key() {
            let mut config = crate::AgentConfig::v0();
            config.llm = crate::defaults::LlmProvider::Online;
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
        fn goose_online_fetches_the_pinned_binary() {
            let mut config = crate::AgentConfig::v0();
            config.agent = crate::defaults::AgentBackend::Goose;
            config.llm = crate::defaults::LlmProvider::Online;
            config.llm_model = "gpt-4o-mini".into();
            let fetcher = BlockedFetcher::default();
            let err = build_llm(
                &empty_manifest_cache("goose-online"),
                &fetcher,
                &mut NoProgress,
                &Cancel::new(),
                &config,
                Some(&Zeroizing::new("sk-x".to_string())),
            )
            .err()
            .expect("goose fetch is attempted");
            #[cfg(feature = "cloud")]
            {
                assert!(matches!(err, Error::ModelCache { .. }), "{err:?}");
                assert!(err.to_string().contains("network blocked"), "{err}");
                assert!(fetcher.hits.load(std::sync::atomic::Ordering::SeqCst) >= 1);
            }
            #[cfg(not(feature = "cloud"))]
            {
                assert!(err.to_string().contains("developer_harness.agent"), "{err}");
                assert!(err.to_string().contains("cloud"), "{err}");
                assert_eq!(fetcher.hits.load(std::sync::atomic::Ordering::SeqCst), 0);
            }
        }

        #[test]
        fn goose_local_is_a_config_error() {
            let mut config = crate::AgentConfig::v0();
            config.agent = crate::defaults::AgentBackend::Goose;
            let fetcher = BlockedFetcher::default();
            let err = build_llm(
                &empty_manifest_cache("goose-local"),
                &fetcher,
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            )
            .err()
            .expect("goose with a local LLM is a config error");
            assert!(err.to_string().contains("developer_harness.agent"), "{err}");
            assert!(err.to_string().contains("online"), "{err}");
            assert!(
                !err.to_string().contains("not implemented yet"),
                "local goose must not look like a missing backend: {err}"
            );
            assert_eq!(
                fetcher.hits.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "local goose must not download the Goose binary"
            );
        }

        #[cfg(feature = "cloud")]
        #[test]
        fn goose_online_build_llm_returns_the_goose_slot() {
            let binary = {
                let dir = std::env::temp_dir().join(format!(
                    "syllabix-real-goose-bin-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let path = dir.join("goose");
                std::fs::write(
                    &path,
                    concat!(
                        "#!/usr/bin/env python3\n",
                        "import json,sys\n",
                        "def send(o):\n",
                        "    sys.stdout.write(json.dumps(o)+'\\n'); sys.stdout.flush()\n",
                        "for raw in sys.stdin:\n",
                        "    raw=raw.strip()\n",
                        "    if not raw: continue\n",
                        "    m=json.loads(raw); method=m.get('method'); i=m.get('id')\n",
                        "    if method=='initialize': send({'jsonrpc':'2.0','id':i,'result':{'protocolVersion':1}})\n",
                        "    elif method=='session/new': send({'jsonrpc':'2.0','id':i,'result':{'sessionId':'s'}})\n",
                        "    elif method and i is not None: send({'jsonrpc':'2.0','id':i,'result':{}})\n",
                    ),
                )
                .unwrap();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = std::fs::metadata(&path).unwrap().permissions();
                    perms.set_mode(0o755);
                    std::fs::set_permissions(&path, perms).unwrap();
                }
                path
            };
            let mut config = crate::AgentConfig::v0();
            config.agent = crate::defaults::AgentBackend::Goose;
            config.llm = crate::defaults::LlmProvider::Online;
            config.llm_model = "gpt-4o-mini".into();
            let _guard = crate::goose::GooseBinGuard::set(binary);
            let llm = build_llm(
                &empty_manifest_cache("goose-slot"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                Some(&Zeroizing::new("sk-x".to_string())),
            )
            .expect("live goose slot");
            assert!(matches!(llm, LiveLlm::Goose(_)));
            assert_eq!(llm.name(), "goose");
            let meta = llm.debug_meta().expect("meta");
            assert_eq!(meta.provider, "goose");
            assert_eq!(meta.model, "gpt-4o-mini");
        }

        #[cfg(feature = "cloud")]
        #[test]
        fn online_tts_needs_no_weights_but_requires_the_key() {
            use crate::providers::Tts;

            let mut config = crate::AgentConfig::v0();
            config.tts = crate::defaults::TtsProvider::Online;
            config.tts_base_url = Some("http://127.0.0.1:8092/v1".into());

            let err = match build_tts(
                &empty_manifest_cache("online-tts-nokey"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("missing key fails fast"),
            };
            assert!(
                err.to_string().contains(crate::config::TTS_API_KEY_ENV),
                "{err}"
            );

            let tts = build_tts(
                &empty_manifest_cache("online-tts-ok"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                Some(&Zeroizing::new("sk-tts".to_string())),
            )
            .expect("online TTS slot builds without any weights");
            assert!(matches!(tts, LiveTts::Remote(_)));
            assert_eq!(tts.name(), crate::remote_tts::PROVIDER_NAME);
            assert_eq!(tts.model_id(), Some("pocket-tts"));
        }

        #[cfg(not(feature = "cloud"))]
        #[test]
        fn online_tts_requires_the_cloud_feature() {
            let mut config = crate::AgentConfig::v0();
            config.tts = crate::defaults::TtsProvider::Online;
            config.tts_base_url = Some("http://127.0.0.1:8092/v1".into());
            let err = match build_tts(
                &empty_manifest_cache("online-tts"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("online TTS needs cloud"),
            };
            assert!(err.to_string().contains("`cloud` feature"), "{err}");
        }

        #[cfg(feature = "cloud")]
        #[test]
        fn online_stt_needs_no_weights_but_requires_the_key() {
            use crate::providers::Stt;

            let mut config = crate::AgentConfig::v0();
            config.stt = crate::defaults::SttProvider::Online;
            config.stt_base_url = Some("http://127.0.0.1:8091/v1".into());

            let err = match build_stt(
                &empty_manifest_cache("online-stt-nokey"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("missing key fails fast"),
            };
            assert!(
                err.to_string().contains(crate::config::STT_API_KEY_ENV),
                "{err}"
            );

            let stt = build_stt(
                &empty_manifest_cache("online-stt-ok"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                Some(&Zeroizing::new("sk-stt".to_string())),
            )
            .expect("online STT slot builds without any weights");
            assert!(matches!(stt, LiveStt::Remote(_)));
            assert_eq!(stt.name(), crate::remote_stt::PROVIDER_NAME);
            let stt = stt.with_language("fr").expect("language");
            assert_eq!(stt.name(), crate::remote_stt::PROVIDER_NAME);
        }

        #[cfg(not(feature = "cloud"))]
        #[test]
        fn online_stt_requires_the_cloud_feature() {
            let mut config = crate::AgentConfig::v0();
            config.stt = crate::defaults::SttProvider::Online;
            config.stt_base_url = Some("http://127.0.0.1:8091/v1".into());
            let err = match build_stt(
                &empty_manifest_cache("online-stt"),
                &BlockedFetcher::default(),
                &mut NoProgress,
                &Cancel::new(),
                &config,
                None,
            ) {
                Err(err) => err,
                Ok(_) => panic!("online STT needs cloud"),
            };
            assert!(err.to_string().contains("`cloud` feature"), "{err}");
        }

        #[test]
        fn local_default_still_reports_llama_cpp() {
            // The slot type reports the configured provider without loading weights;
            // loading the engine itself stays with native inference tests.
            let mut config = crate::AgentConfig::v0();
            config.llm = crate::defaults::LlmProvider::Local;
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

#[cfg(all(not(coverage), feature = "ggml"))]
pub use production::*;
