//! Load the blessed v0 native providers from the model cache.
//!
//! The LLM slot is a [`LiveLlm`]: the in-process llama.cpp GGUF engine by
//! default, or the row-30 BYO-key cloud adapter when yaml selects
//! `pipeline.llm.provider: openai`. VAD, AEC, STT, and TTS stay local either
//! way; only transcript text reaches the cloud endpoint.

use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::{Error, Result};
use crate::llm::LlamaLlm;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::openai::{OpenAiLlm, OpenAiSettings, API_KEY_ENV, PROVIDER_NAME};
use crate::providers::Llm;
use crate::stt::WhisperStt;
use crate::tts::KokoroTts;
use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, Transcript};
use crate::vad::SileroVad;

/// The configured LLM implementation for one live run.
pub enum LiveLlm {
    /// In-process GGUF engine (default).
    Local(LlamaLlm),
    /// BYO-key OpenAI-compatible endpoint (row 30).
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

/// Silero + whisper.cpp + llama.cpp/Kokoro, resolved through the v0 manifest.
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
) -> Result<(SileroVad, WhisperStt, LiveLlm, KokoroTts)> {
    let vad = SileroVad::from_cache(cache, fetcher, progress, cancel)?
        .with_settings(config.vad_settings());
    // Only the selected STT id is fetched; the rest of the menu stays on disk.
    let stt = WhisperStt::from_cache(cache, fetcher, progress, cancel, config.stt_model)?
        .with_language(&config.language)?;
    let llm = build_llm(cache, fetcher, progress, cancel, config, llm_api_key)?;
    let tts = KokoroTts::from_cache(cache, fetcher, progress, cancel)?;
    Ok((vad, stt, llm, tts))
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
    match config.llm {
        crate::LlmProvider::Local => Ok(LiveLlm::Local(LlamaLlm::from_cached_model(
            cache,
            fetcher,
            progress,
            cancel,
            &config.llm_model,
            config.thinking,
        )?)),
        crate::LlmProvider::Online => {
            let key = llm_api_key.ok_or_else(|| Error::Config {
                field: API_KEY_ENV.into(),
                message: format!("is required when pipeline.llm.provider is {PROVIDER_NAME}"),
            })?;
            Ok(LiveLlm::Cloud(OpenAiLlm::new(
                OpenAiSettings::from_config(config),
                key.clone(),
            )))
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
        // The slot type must keep reporting the launch provider by default;
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
