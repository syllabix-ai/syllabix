//! Fluid Speech Pro Q8_0 GGUF via vendored parakeet.cpp (finalize only).

use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(all(not(coverage), feature = "parakeet"))]
use syllabix_native::ParakeetContext;

use crate::language::LANGUAGE_AUTO;
use crate::{Stt, SttModel, STT_LANGUAGE};
use syllabix_native::error::{Error, Result};
use syllabix_native::models::{Fetcher, ModelCache, Progress};
use syllabix_native::types::{Transcript, Utterance};
use syllabix_native::Cancel;

/// Config / log name for this engine.
pub const PROVIDER_NAME: &str = "fluid-gguf";
/// Decoder GGUF asset id.
pub const ASSET: &str = "fluid-speech-pro-gguf-q8_0";

/// European language codes Fluid's card documents (plus `auto` in yaml).
pub const FLUID_LANGUAGE_CODES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];

/// True for `auto` or one of [`FLUID_LANGUAGE_CODES`].
pub fn is_fluid_language(code: &str) -> bool {
    code == LANGUAGE_AUTO || FLUID_LANGUAGE_CODES.contains(&code)
}

#[cfg(all(not(coverage), feature = "parakeet"))]
fn default_threads() -> i32 {
    #[cfg(feature = "whisper")]
    {
        crate::whisper::thread_count()
    }
    #[cfg(not(feature = "whisper"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get().min(4) as i32)
            .unwrap_or(1)
    }
}

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

/// In-process Fluid Speech Pro adapter.
pub struct FluidGgufStt {
    decoder: Arc<Mutex<Box<dyn Decoder>>>,
    language: String,
    threads: i32,
}

impl Clone for FluidGgufStt {
    fn clone(&self) -> Self {
        Self {
            decoder: Arc::clone(&self.decoder),
            language: self.language.clone(),
            threads: self.threads,
        }
    }
}

impl FluidGgufStt {
    #[cfg(all(not(coverage), feature = "parakeet"))]
    pub fn from_path(model: impl AsRef<Path>) -> Result<Self> {
        let threads = default_threads();
        let ctx = ParakeetContext::load(model.as_ref(), threads).map_err(provider)?;
        Ok(Self {
            decoder: Arc::new(Mutex::new(Box::new(NativeDecoder { ctx }))),
            language: STT_LANGUAGE.to_string(),
            threads,
        })
    }

    #[cfg(any(coverage, not(feature = "parakeet")))]
    pub fn from_path(model: impl AsRef<Path>) -> Result<Self> {
        let _ = model;
        Err(provider(if cfg!(coverage) {
            "Fluid Speech Pro inference is not loaded in coverage tests."
        } else {
            "Fluid Speech Pro inference requires the `ggml` feature with parakeet"
        }))
    }

    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        model: SttModel,
    ) -> Result<Self> {
        if model != SttModel::FluidSpeechProGgufQ8 {
            return Err(provider(format!(
                "FluidGgufStt::from_cache requires {}, got {}",
                SttModel::FluidSpeechProGgufQ8.as_str(),
                model.as_str()
            )));
        }
        let asset = cache
            .manifest()
            .asset(ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {ASSET} asset"),
            })?;
        let model_path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_path(model_path)
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        if !is_fluid_language(&language) {
            return Err(Error::Config {
                field: "pipeline.stt.language".into(),
                message: format!(
                    "unsupported value {language:?} (allowed: auto or {})",
                    FLUID_LANGUAGE_CODES.join(", ")
                ),
            });
        }
        self.language = language;
        Ok(self)
    }

    pub fn with_threads(mut self, n: i32) -> Self {
        if n > 0 {
            self.threads = n;
            if let Ok(mut guard) = self.decoder.lock() {
                guard.set_threads(n);
            }
        }
        self
    }

    pub fn intra_op_threads(&self) -> Option<i32> {
        Some(self.threads)
    }

    #[cfg(test)]
    fn with_decoder(decoder: Box<dyn Decoder>, threads: i32) -> Self {
        Self {
            decoder: Arc::new(Mutex::new(decoder)),
            language: STT_LANGUAGE.to_string(),
            threads,
        }
    }
}

impl Stt for FluidGgufStt {
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
        let text = self
            .decoder
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .decode(&audio, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let language = if self.language == LANGUAGE_AUTO {
            STT_LANGUAGE.to_string()
        } else {
            self.language.clone()
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err(provider("empty transcript"));
        }
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language,
        })
    }
}

trait Decoder: Send {
    fn decode(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String>;
    fn set_threads(&mut self, n: i32);
}

#[cfg(all(not(coverage), feature = "parakeet"))]
struct NativeDecoder {
    ctx: ParakeetContext,
}

#[cfg(all(not(coverage), feature = "parakeet"))]
impl Decoder for NativeDecoder {
    fn decode(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        self.ctx.transcribe(pcm).map_err(provider)
    }

    fn set_threads(&mut self, n: i32) {
        self.ctx.set_threads(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};
    use syllabix_native::types::AudioFrame;
    use syllabix_native::types::TurnId;

    struct ScriptedDecoder {
        replies: Vec<&'static str>,
        index: usize,
        shutdown_after: bool,
        threads: Arc<AtomicI32>,
    }

    impl ScriptedDecoder {
        fn new(replies: &[&'static str]) -> Self {
            Self {
                replies: replies.to_vec(),
                index: 0,
                shutdown_after: false,
                threads: Arc::new(AtomicI32::new(0)),
            }
        }
    }

    impl Decoder for ScriptedDecoder {
        fn decode(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if pcm.is_empty() {
                return Err(provider("no samples"));
            }
            let reply = self.replies.get(self.index).copied().unwrap_or("fallback");
            self.index += 1;
            if self.shutdown_after {
                return Err(Error::Cancelled);
            }
            Ok(reply.to_string())
        }

        fn set_threads(&mut self, n: i32) {
            self.threads.store(n, Ordering::Relaxed);
        }
    }

    fn utterance() -> Utterance {
        Utterance {
            turn: TurnId(1),
            frames: vec![AudioFrame {
                seq: 0,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![1000, -1000, 2000, -2000],
                capture_pcm: None,
            }],
        }
    }

    #[test]
    fn name_returns_provider_id() {
        let stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4);
        assert_eq!(stt.name(), PROVIDER_NAME);
    }

    #[test]
    fn clone_reuses_decoder_arc() {
        let stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["hello"])), 4);
        let mut cloned = stt.clone();
        let transcript = cloned
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.text, "hello");
    }

    #[test]
    fn cancel_before_decode_returns_cancelled() {
        let mut stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = stt.transcribe(&utterance(), &cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn empty_frames_are_a_provider_error() {
        let mut stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4);
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
    fn fluid_language_menu() {
        assert!(is_fluid_language("auto"));
        assert!(is_fluid_language("de"));
        assert!(!is_fluid_language("ja"));
        assert!(!is_fluid_language("or"));
    }

    #[test]
    fn with_language_rejects_unknown_codes() {
        let stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4);
        assert!(matches!(stt.with_language("ja"), Err(Error::Config { .. })));
    }

    #[test]
    fn with_language_accepts_auto() {
        let stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4)
            .with_language("auto")
            .expect("auto");
        assert_eq!(stt.language(), "auto");
    }

    #[test]
    fn transcribe_returns_scripted_text() {
        let mut stt =
            FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["hello world"])), 4);
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.text, "hello world");
        assert_eq!(transcript.language, STT_LANGUAGE);
    }

    #[test]
    fn configured_language_is_echoed() {
        let mut stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["bonjour"])), 4)
            .with_language("fr")
            .expect("fr");
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("scripted");
        assert_eq!(transcript.language, "fr");
    }

    #[test]
    fn empty_samples_are_a_provider_error() {
        let mut stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["x"])), 4);
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
        let mut stt = FluidGgufStt::with_decoder(Box::new(decoder), 4);
        let err = stt.transcribe(&utterance(), &Cancel::new()).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn empty_transcript_is_a_provider_error() {
        let mut stt = FluidGgufStt::with_decoder(Box::new(ScriptedDecoder::new(&["   "])), 4);
        let err = stt.transcribe(&utterance(), &Cancel::new()).unwrap_err();
        assert!(err.to_string().contains("empty transcript"), "{err}");
    }

    #[test]
    fn with_threads_updates_decoder() {
        let threads = Arc::new(AtomicI32::new(0));
        let decoder = ScriptedDecoder {
            replies: vec!["x"],
            index: 0,
            shutdown_after: false,
            threads: Arc::clone(&threads),
        };
        let stt = FluidGgufStt::with_decoder(Box::new(decoder), 4).with_threads(8);
        assert_eq!(stt.intra_op_threads(), Some(8));
        assert_eq!(threads.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn from_cache_rejects_non_fluid_models() {
        let cache = ModelCache::v0();
        let err = match FluidGgufStt::from_cache(
            &cache,
            &syllabix_native::models::BlockedFetcher::default(),
            &mut syllabix_native::models::NoProgress,
            &Cancel::new(),
            SttModel::Small,
        ) {
            Err(err) => err,
            Ok(_) => panic!("whisper model should not load as fluid gguf"),
        };
        assert!(err.to_string().contains(ASSET), "{err}");
    }

    #[test]
    fn from_cache_requires_asset() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-fluid-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(
            root,
            syllabix_native::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match FluidGgufStt::from_cache(
            &cache,
            &syllabix_native::models::BlockedFetcher::default(),
            &mut syllabix_native::models::NoProgress,
            &Cancel::new(),
            SttModel::FluidSpeechProGgufQ8,
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_path_does_not_load_weights() {
        let err = match FluidGgufStt::from_path("fake-fluid.gguf") {
            Err(err) => err,
            Ok(_) => panic!("coverage must not load Fluid GGUF"),
        };
        assert!(err.to_string().contains("coverage tests"), "{err}");
    }

    #[cfg(not(coverage))]
    #[test]
    fn missing_model_files_are_a_provider_error() {
        let err = match FluidGgufStt::from_path("/no/such/fluid.gguf") {
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
