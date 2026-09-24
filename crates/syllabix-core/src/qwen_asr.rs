//! Qwen3-ASR via llama.cpp `mtmd` (qwen3a encoder + Qwen3 decoder GGUF).
//!
//! Batch finalize only: one completed VAD [`Utterance`] in, one [`Transcript`]
//! out — the same contract as [`WhisperStt`](crate::stt::WhisperStt). Partials
//! stay off until a later phase measures a streaming recipe.

use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(not(coverage))]
use syllabix_native::{DecodeError, QwenAsrContext, QwenTtsBackend};

use crate::defaults::TtsCompute;
use crate::error::{Error, Result};
use crate::language::{is_supported, language_name, LANGUAGE_AUTO};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Stt;
use crate::stt::STT_LANGUAGE;
use crate::types::{Transcript, Utterance};
use crate::Cancel;
use crate::SttModel;

/// Config / log name for this engine.
pub const PROVIDER_NAME: &str = "qwen-asr";
/// Decoder GGUF asset id.
pub const ASR_ASSET: &str = "qwen3-asr-0.6";
/// Audio-encoder mmproj asset id.
pub const ASR_MMPROJ_ASSET: &str = "qwen3-asr-0.6-mmproj";

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

/// In-process Qwen3-ASR adapter.
pub struct QwenAsrStt {
    decoder: Arc<Mutex<Box<dyn Decoder>>>,
    language: String,
    backend: Option<&'static str>,
}

impl Clone for QwenAsrStt {
    fn clone(&self) -> Self {
        Self {
            decoder: Arc::clone(&self.decoder),
            language: self.language.clone(),
            backend: self.backend,
        }
    }
}

impl QwenAsrStt {
    /// Load decoder GGUF + mmproj from disk with default Auto placement.
    #[cfg(not(coverage))]
    pub fn from_paths(model: impl AsRef<Path>, mmproj: impl AsRef<Path>) -> Result<Self> {
        Self::from_paths_with_compute(model, mmproj, TtsCompute::Auto)
    }

    /// Load with an explicit [`TtsCompute`] placement (same ids as Qwen TTS).
    #[cfg(not(coverage))]
    pub fn from_paths_with_compute(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        compute: TtsCompute,
    ) -> Result<Self> {
        let decoder = NativeDecoder::load(model.as_ref(), mmproj.as_ref(), compute)?;
        let backend = Some(decoder.backend.as_str());
        Ok(Self {
            decoder: Arc::new(Mutex::new(Box::new(decoder))),
            language: STT_LANGUAGE.to_string(),
            backend,
        })
    }

    /// Coverage never opens GGUF weights; keep the constructors as callable
    /// seams for cache and provider-error tests.
    #[cfg(coverage)]
    pub fn from_paths(model: impl AsRef<Path>, mmproj: impl AsRef<Path>) -> Result<Self> {
        Self::from_paths_with_compute(model, mmproj, TtsCompute::Auto)
    }

    #[cfg(coverage)]
    pub fn from_paths_with_compute(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        compute: TtsCompute,
    ) -> Result<Self> {
        let _ = (model, mmproj, compute);
        Err(provider(
            "Qwen3-ASR inference is not loaded in coverage tests.",
        ))
    }

    /// Resolve the 0.6B pair from the manifest cache, then load. Only these
    /// two assets are fetched.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        model: SttModel,
        compute: TtsCompute,
    ) -> Result<Self> {
        if model != SttModel::QwenAsr06 {
            return Err(provider(format!(
                "QwenAsrStt::from_cache requires qwen3-asr-0.6, got {}",
                model.as_str()
            )));
        }
        let backbone = cache
            .manifest()
            .asset(ASR_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {ASR_ASSET} asset"),
            })?;
        let mmproj = cache
            .manifest()
            .asset(ASR_MMPROJ_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {ASR_MMPROJ_ASSET} asset"),
            })?;
        let model_path = cache.resolve(backbone, fetcher, progress, cancel)?;
        let mmproj_path = cache.resolve(mmproj, fetcher, progress, cancel)?;
        Self::from_paths_with_compute(model_path, mmproj_path, compute)
    }

    /// Compute backend confirmed after load (`cpu` / `metal` / `vulkan`).
    pub fn backend_id(&self) -> Option<&str> {
        self.backend
    }

    /// Configured or last-detected STT language code.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Set the yaml language: any whisper-supported code or `auto`.
    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        if !is_supported(&language) {
            return Err(Error::Config {
                field: "pipeline.stt.language".into(),
                message: format!("unsupported value {language:?} (allowed: ISO code or \"auto\")"),
            });
        }
        self.language = language;
        Ok(self)
    }

    #[cfg(test)]
    fn with_decoder(decoder: Box<dyn Decoder>) -> Self {
        Self {
            decoder: Arc::new(Mutex::new(decoder)),
            language: STT_LANGUAGE.to_string(),
            backend: None,
        }
    }
}

impl Stt for QwenAsrStt {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(provider("utterance has no frames"));
        }
        let pcm = utterance.pcm();
        if pcm.is_empty() {
            return Err(provider("utterance has no samples"));
        }
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let prompt_lang = prompt_language(&self.language);
        let raw = self
            .decoder
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .decode(&audio, &prompt_lang, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let (text, language) = split_asr_output(&raw, &self.language);
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language,
        })
    }
}

trait Decoder: Send {
    fn decode(&mut self, pcm: &[f32], language: &str, cancel: &Cancel) -> Result<String>;
}

/// Native model loading lives behind `cfg(not(coverage))` so llvm-cov never
/// counts unhit ggml FFI. Native inference tests still compile this path.
#[cfg(not(coverage))]
struct NativeDecoder {
    ctx: QwenAsrContext,
    backend: QwenTtsBackend,
}

#[cfg(not(coverage))]
impl NativeDecoder {
    fn load(model: &Path, mmproj: &Path, compute: TtsCompute) -> Result<Self> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4) as i32)
            .unwrap_or(1);
        let load =
            |backend| QwenAsrContext::load(model, mmproj, n_threads, backend).map_err(provider);
        let (ctx, _) = load_first_backend(backend_attempts(compute)?, load)?;
        let backend = ctx.backend();
        Ok(Self { ctx, backend })
    }
}

#[cfg(not(coverage))]
fn backend_attempts(compute: TtsCompute) -> Result<&'static [QwenTtsBackend]> {
    const CPU: &[QwenTtsBackend] = &[QwenTtsBackend::Cpu];
    const METAL: &[QwenTtsBackend] = &[QwenTtsBackend::Metal];
    const METAL_THEN_CPU: &[QwenTtsBackend] = &[QwenTtsBackend::Metal, QwenTtsBackend::Cpu];
    const VULKAN: &[QwenTtsBackend] = &[QwenTtsBackend::Vulkan];
    const VULKAN_THEN_CPU: &[QwenTtsBackend] = &[QwenTtsBackend::Vulkan, QwenTtsBackend::Cpu];
    Ok(match compute {
        TtsCompute::Cpu => CPU,
        TtsCompute::Auto if cfg!(all(target_os = "macos", target_arch = "aarch64")) => {
            METAL_THEN_CPU
        }
        TtsCompute::Auto
            if syllabix_native::ggml_vulkan_compiled()
                && syllabix_native::vulkan_device_count() > 0 =>
        {
            VULKAN_THEN_CPU
        }
        TtsCompute::Auto => CPU,
        TtsCompute::Metal if cfg!(target_os = "macos") => METAL,
        TtsCompute::Metal => {
            return Err(provider("Metal compute requires macOS"));
        }
        TtsCompute::Vulkan if !syllabix_native::ggml_vulkan_compiled() => {
            return Err(provider(
                "Vulkan compute requires a Linux build with SYLLABIX_GGML_VULKAN=1",
            ));
        }
        TtsCompute::Vulkan if syllabix_native::vulkan_device_count() > 0 => VULKAN,
        TtsCompute::Vulkan => {
            return Err(provider("Vulkan compute found no usable Vulkan device"));
        }
    })
}

#[cfg(not(coverage))]
fn load_first_backend<T, E>(
    attempts: &[QwenTtsBackend],
    mut load: impl FnMut(QwenTtsBackend) -> std::result::Result<T, E>,
) -> std::result::Result<(T, QwenTtsBackend), E> {
    let mut last_error = None;
    for &backend in attempts {
        match load(backend) {
            Ok(value) => return Ok((value, backend)),
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.expect("Qwen ASR backend attempt list is never empty"))
}

#[cfg(not(coverage))]
impl Decoder for NativeDecoder {
    fn decode(&mut self, pcm: &[f32], language: &str, cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort_user = cancel as *const Cancel as *mut std::ffi::c_void;
        match unsafe {
            self.ctx
                .decode(pcm, language, Some(abort_on_shutdown), abort_user)
        } {
            Ok(text) => {
                if cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                Ok(text)
            }
            Err(DecodeError::Cancelled) => Err(Error::Cancelled),
            Err(DecodeError::Failed(message)) => Err(provider(message)),
        }
    }
}

#[cfg(not(coverage))]
unsafe extern "C" fn abort_on_shutdown(user_data: *mut std::ffi::c_void) -> bool {
    unsafe { (*(user_data as *const Cancel)).is_shutdown() }
}

fn prompt_language(code: &str) -> String {
    if code == LANGUAGE_AUTO {
        return "auto".into();
    }
    language_name(code).unwrap_or("English").to_string()
}

/// Split `language English<asr_text>hello` into transcript + ISO code.
pub(crate) fn split_asr_output(raw: &str, configured: &str) -> (String, String) {
    const TAG: &str = "<asr_text>";
    let fallback = if configured == LANGUAGE_AUTO {
        STT_LANGUAGE
    } else {
        configured
    };
    if let Some(index) = raw.find(TAG) {
        let head = raw[..index].trim();
        let text = raw[index + TAG.len()..].trim().to_string();
        let language = head
            .strip_prefix("language ")
            .map(str::trim)
            .and_then(language_code_from_name)
            .unwrap_or(fallback)
            .to_string();
        return (text, language);
    }
    (raw.trim().to_string(), fallback.to_string())
}

fn language_code_from_name(name: &str) -> Option<&'static str> {
    crate::language::WHISPER_LANGUAGES
        .iter()
        .find(|(_, display)| display.eq_ignore_ascii_case(name))
        .map(|(code, _)| *code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioFrame, TurnId};

    struct ScriptedDecoder {
        replies: Vec<String>,
        index: usize,
        languages: std::sync::Arc<Mutex<Vec<String>>>,
        shutdown_after: bool,
    }

    impl ScriptedDecoder {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: replies.iter().map(|s| (*s).to_string()).collect(),
                index: 0,
                languages: std::sync::Arc::new(Mutex::new(Vec::new())),
                shutdown_after: false,
            }
        }
    }

    impl Decoder for ScriptedDecoder {
        fn decode(&mut self, _pcm: &[f32], language: &str, cancel: &Cancel) -> Result<String> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            self.languages
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(language.to_string());
            let text = self.replies.get(self.index).cloned().unwrap_or_default();
            self.index += 1;
            if self.shutdown_after {
                cancel.shutdown();
            }
            Ok(text)
        }
    }

    fn utterance() -> Utterance {
        Utterance {
            turn: TurnId(1),
            frames: vec![AudioFrame {
                seq: 0,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![100, -100],
                capture_pcm: None,
            }],
        }
    }

    #[test]
    fn strips_language_tag_and_maps_display_name() {
        let (text, language) =
            split_asr_output("language French<asr_text> bonjour le monde ", "en");
        assert_eq!(text, "bonjour le monde");
        assert_eq!(language, "fr");
    }

    #[test]
    fn plain_transcript_keeps_configured_language() {
        let (text, language) = split_asr_output("hello there", "de");
        assert_eq!(text, "hello there");
        assert_eq!(language, "de");
    }

    #[test]
    fn auto_without_tag_falls_back_to_english() {
        let (text, language) = split_asr_output("hello", LANGUAGE_AUTO);
        assert_eq!(text, "hello");
        assert_eq!(language, "en");
    }

    #[test]
    fn transcribe_uses_scripted_decoder() {
        let mut stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&[
            "language English<asr_text>hello from qwen",
        ])));
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.text, "hello from qwen");
        assert_eq!(transcript.language, "en");
        assert_eq!(stt.name(), PROVIDER_NAME);
        assert!(!stt.supports_partials());
    }

    #[test]
    fn empty_utterance_is_a_provider_error() {
        let mut stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])));
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: vec![],
                },
                &Cancel::new(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("no frames"), "{err}");
    }

    #[test]
    fn shutdown_cancels_before_decode() {
        let mut stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])));
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = stt.transcribe(&utterance(), &cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn with_language_rejects_unknown_codes() {
        let stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&[])));
        let err = match stt.with_language("klingon") {
            Err(err) => err,
            Ok(_) => panic!("unknown language should fail"),
        };
        assert!(err.to_string().contains("pipeline.stt.language"), "{err}");
    }

    #[test]
    fn clone_exposes_language_and_name() {
        let stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&[])));
        assert_eq!(stt.language(), STT_LANGUAGE);
        let cloned = stt.clone();
        assert_eq!(cloned.language(), STT_LANGUAGE);
        assert_eq!(cloned.name(), PROVIDER_NAME);
    }

    #[test]
    fn with_language_auto_prompts_auto() {
        let decoder = ScriptedDecoder::new(&["language English<asr_text>hi"]);
        let languages = std::sync::Arc::clone(&decoder.languages);
        let mut stt = QwenAsrStt::with_decoder(Box::new(decoder))
            .with_language(LANGUAGE_AUTO)
            .expect("auto");
        assert_eq!(stt.language(), LANGUAGE_AUTO);
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.text, "hi");
        assert_eq!(transcript.language, "en");
        assert_eq!(languages.lock().expect("languages").as_slice(), ["auto"]);
    }

    #[test]
    fn with_language_french_prompts_display_name() {
        let decoder = ScriptedDecoder::new(&["language French<asr_text>bonjour"]);
        let languages = std::sync::Arc::clone(&decoder.languages);
        let mut stt = QwenAsrStt::with_decoder(Box::new(decoder))
            .with_language("fr")
            .expect("fr");
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.language, "fr");
        assert_eq!(transcript.text, "bonjour");
        assert_eq!(languages.lock().expect("languages").as_slice(), ["French"]);
    }

    #[test]
    fn empty_samples_are_a_provider_error() {
        let mut stt = QwenAsrStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])));
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: vec![AudioFrame {
                        seq: 0,
                        sample_rate_hz: 16_000,
                        channels: 1,
                        samples: vec![],
                        capture_pcm: None,
                    }],
                },
                &Cancel::new(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("no samples"), "{err}");
    }

    #[test]
    fn shutdown_after_decode_cancels_the_transcript() {
        let mut decoder = ScriptedDecoder::new(&["hello"]);
        decoder.shutdown_after = true;
        let mut stt = QwenAsrStt::with_decoder(Box::new(decoder));
        let err = stt.transcribe(&utterance(), &Cancel::new()).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn unknown_language_name_keeps_configured_code() {
        let (text, language) = split_asr_output("language Klingon<asr_text> qapla ", "en");
        assert_eq!(text, "qapla");
        assert_eq!(language, "en");
    }

    #[test]
    fn tag_without_language_prefix_keeps_fallback() {
        let (text, language) = split_asr_output("French<asr_text>bonjour", LANGUAGE_AUTO);
        assert_eq!(text, "bonjour");
        assert_eq!(language, "en");
    }

    #[test]
    fn from_cache_rejects_non_qwen_models() {
        let cache = ModelCache::v0();
        let err = match QwenAsrStt::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::Small,
            TtsCompute::Auto,
        ) {
            Err(err) => err,
            Ok(_) => panic!("whisper model should not load as qwen asr"),
        };
        assert!(err.to_string().contains("qwen3-asr-0.6"), "{err}");
    }

    #[test]
    fn from_cache_requires_both_assets() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-qwen-asr-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(
            root,
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match QwenAsrStt::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::QwenAsr06,
            TtsCompute::Auto,
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains(ASR_ASSET), "{err}");
    }

    #[test]
    fn from_cache_requires_mmproj_asset() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-qwen-asr-mmproj-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let backbone = crate::models::Manifest::v0()
            .asset(ASR_ASSET)
            .cloned()
            .expect("qwen3-asr-0.6 in v0 manifest");
        let cache = ModelCache::new(
            root,
            crate::models::Manifest {
                version: 1,
                assets: vec![backbone],
            },
        );
        let err = match QwenAsrStt::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::QwenAsr06,
            TtsCompute::Auto,
        ) {
            Err(err) => err,
            Ok(_) => panic!("missing mmproj should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains(ASR_MMPROJ_ASSET), "{err}");
    }

    #[test]
    fn from_cache_does_not_fetch_when_network_is_blocked() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-qwen-asr-blocked-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(root, crate::models::Manifest::v0());
        let err = match QwenAsrStt::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::QwenAsr06,
            TtsCompute::Auto,
        ) {
            Err(err) => err,
            Ok(_) => panic!("blocked fetch should fail"),
        };
        assert!(err.to_string().contains("network blocked"), "{err}");
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_paths_does_not_load_weights() {
        let err = match QwenAsrStt::from_paths("fake-qwen-asr.gguf", "fake-mmproj.gguf") {
            Err(err) => err,
            Ok(_) => panic!("coverage must not load Qwen3-ASR"),
        };
        assert!(err.to_string().contains("coverage tests"), "{err}");
    }

    #[cfg(not(coverage))]
    #[test]
    fn missing_model_files_are_a_provider_error() {
        let err = match QwenAsrStt::from_paths("/no/such/qwen3-asr.gguf", "/no/such/mmproj.gguf") {
            Err(err) => err,
            Ok(_) => panic!("missing paths should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: PROVIDER_NAME,
                ..
            }
        ));
    }
}
