//! In-process whisper.cpp speech-to-text for completed VAD utterances.

use std::os::raw::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

use syllabix_native::{DecodeError, WhisperContext};

use crate::defaults::{BuiltinDefaults, SttModel};
use crate::error::{Error, Result};
use crate::language::{is_supported as is_supported_language, LANGUAGE_AUTO};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Stt;
use crate::types::{Transcript, Utterance};
use crate::Cancel;

/// Default STT language. YAML may select any whisper-supported ISO code.
pub const STT_LANGUAGE: &str = "en";

/// In-process whisper.cpp adapter. Loads one menu GGML (default [`SttModel::Small`]).
pub struct WhisperStt {
    decoder: Arc<Mutex<Box<dyn Decoder>>>,
    model: SttModel,
    language: String,
}

impl Clone for WhisperStt {
    fn clone(&self) -> Self {
        Self {
            decoder: Arc::clone(&self.decoder),
            model: self.model,
            language: self.language.clone(),
        }
    }
}

impl WhisperStt {
    /// Load a ggml model from disk.
    pub fn from_model_path(path: impl AsRef<Path>, model: SttModel) -> Result<Self> {
        let decoder = WhisperDecoder::load(path.as_ref())?;
        Ok(Self {
            decoder: Arc::new(Mutex::new(Box::new(decoder))),
            model,
            language: STT_LANGUAGE.to_string(),
        })
    }

    /// Resolve the selected model's asset from the manifest cache, then load it.
    /// Only this id is fetched; the other menu sizes stay untouched on disk.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        model: SttModel,
    ) -> Result<Self> {
        let asset_id = model.asset_id();
        let asset = cache
            .manifest()
            .asset(asset_id)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {asset_id} asset"),
            })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_model_path(path, model)
    }

    /// Configured or last-detected STT language code.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Set the yaml language: any whisper-supported code or `auto`.
    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        if !is_supported_language(&language) {
            return Err(Error::Config {
                field: "pipeline.stt.language".into(),
                message: format!("unsupported value {language:?} (allowed: ISO code or \"auto\")"),
            });
        }
        self.language = language;
        Ok(self)
    }

    /// True when decode detects the language per utterance instead of pinning one.
    pub fn auto_detects(&self) -> bool {
        self.language == LANGUAGE_AUTO
    }

    /// Configured whisper.cpp model id.
    pub fn model(&self) -> SttModel {
        self.model
    }

    #[cfg(test)]
    fn with_decoder(decoder: Box<dyn Decoder>, model: SttModel) -> Self {
        Self {
            decoder: Arc::new(Mutex::new(decoder)),
            model,
            language: STT_LANGUAGE.to_string(),
        }
    }
}

impl Stt for WhisperStt {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().stt.as_str()
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no frames".into(),
            });
        }
        let pcm = utterance.pcm();
        if pcm.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no samples".into(),
            });
        }
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let (text, language) = self
            .decoder
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .decode(&audio, &self.language, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language,
        })
    }
}

trait Decoder: Send {
    /// Decode one utterance. Returns `(text, effective_language)`.
    fn decode(&mut self, pcm: &[f32], language: &str, cancel: &Cancel) -> Result<(String, String)>;
}

struct WhisperDecoder {
    ctx: WhisperContext,
}

impl WhisperDecoder {
    fn load(path: &Path) -> Result<Self> {
        let ctx = WhisperContext::load(path).map_err(|message| Error::Provider {
            provider: "whisper.cpp",
            message,
        })?;
        Ok(Self { ctx })
    }
}

impl Decoder for WhisperDecoder {
    fn decode(&mut self, pcm: &[f32], language: &str, cancel: &Cancel) -> Result<(String, String)> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort_user = cancel as *const Cancel as *mut c_void;
        match unsafe {
            self.ctx.decode(
                pcm,
                thread_count(),
                language,
                Some(abort_on_shutdown),
                abort_user,
            )
        } {
            Ok((text, detected)) => {
                if cancel.is_shutdown() {
                    Err(Error::Cancelled)
                } else {
                    Ok((text, detected))
                }
            }
            Err(DecodeError::Cancelled) => Err(Error::Cancelled),
            Err(DecodeError::Failed(_)) if cancel.is_shutdown() => Err(Error::Cancelled),
            Err(DecodeError::Failed(message)) => Err(Error::Provider {
                provider: "whisper.cpp",
                message,
            }),
        }
    }
}

unsafe extern "C" fn abort_on_shutdown(user_data: *mut c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // Safety: `user_data` is `&Cancel` for the duration of `whisper_full`.
    unsafe { (*(user_data as *const Cancel)).is_shutdown() }
}

fn thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4) as i32)
        .unwrap_or(1)
}

/// Return lowercase alphabetic words in their original order for transcript comparison.
pub fn transcript_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect()
}

/// Fraction of `expected` words that appear in order in `text`.
///
/// Extra hypothesis words are ignored. A missed reference word does not
/// consume later hypothesis tokens, so a numeral like "20" instead of
/// "twenty" costs one miss and still allows the rest of the quote to match.
pub fn word_match_ratio(text: &str, expected: &[&str]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let got = transcript_words(text);
    let mut index = 0usize;
    let mut matched = 0usize;
    for word in expected {
        if let Some(found) = got[index..].iter().position(|w| w == word) {
            index += found + 1;
            matched += 1;
        }
    }
    matched as f64 / expected.len() as f64
}

/// True when every `expected` word appears in order inside `text` (extra words allowed).
pub fn contains_words_in_order(text: &str, expected: &[&str]) -> bool {
    word_match_ratio(text, expected) >= 1.0 - f64::EPSILON
}

/// Require at least 80% of the official transcript words in order.
pub const LIBRISPEECH_MIN_WORD_MATCH: f64 = 0.8;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Stt;
    use crate::types::{
        AudioFrame, TurnId, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
    };
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    struct ScriptedDecoder {
        replies: Vec<String>,
        detected: String,
        delay: Duration,
        calls: Arc<Mutex<usize>>,
    }

    impl ScriptedDecoder {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: replies.iter().map(|s| (*s).to_string()).collect(),
                detected: STT_LANGUAGE.to_string(),
                delay: Duration::ZERO,
                calls: Arc::new(Mutex::new(0)),
            }
        }

        fn with_detected(mut self, detected: &str) -> Self {
            self.detected = detected.to_string();
            self
        }
    }

    impl Decoder for ScriptedDecoder {
        fn decode(
            &mut self,
            _pcm: &[f32],
            language: &str,
            cancel: &Cancel,
        ) -> Result<(String, String)> {
            if !self.delay.is_zero() {
                let start = std::time::Instant::now();
                while start.elapsed() < self.delay {
                    if cancel.is_shutdown() {
                        return Err(Error::Cancelled);
                    }
                    thread::sleep(Duration::from_millis(5));
                }
            }
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            *self.calls.lock().expect("call counter") += 1;
            if self.replies.is_empty() {
                return Err(Error::Provider {
                    provider: "whisper.cpp",
                    message: "scripted decoder exhausted".into(),
                });
            }
            let text = self.replies.remove(0);
            // Fixed-language runs echo the request; auto runs report detection.
            let detected = if language == LANGUAGE_AUTO {
                self.detected.clone()
            } else {
                language.to_string()
            };
            Ok((text, detected))
        }
    }

    fn speech_utterance(turn: u64, frames: usize) -> Utterance {
        let mut list = Vec::with_capacity(frames);
        for seq in 0..frames as u64 {
            list.push(
                AudioFrame::new(
                    seq,
                    DEFAULT_SAMPLE_RATE_HZ,
                    DEFAULT_CHANNELS,
                    vec![120; FRAME_SAMPLES],
                )
                .unwrap(),
            );
        }
        Utterance {
            turn: TurnId(turn),
            frames: list,
        }
    }

    #[test]
    fn name_and_model_match_v0_defaults() {
        let stt =
            WhisperStt::with_decoder(Box::new(ScriptedDecoder::new(&["hello"])), SttModel::Small);
        assert_eq!(stt.name(), "local");
        assert_eq!(stt.model().as_str(), "whisper-small");
        assert_eq!(SttModel::Small.asset_id(), "whisper-small");
        assert_eq!(stt.language(), STT_LANGUAGE);
        assert!(!stt.auto_detects());
        let err = match stt.with_language("klingon") {
            Err(err) => err,
            Ok(_) => panic!("unsupported language must fail"),
        };
        assert!(err.to_string().contains("pipeline.stt.language"));
        let fr = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder::new(&["bonjour"]).with_detected("fr")),
            SttModel::Small,
        )
        .with_language("fr")
        .expect("ISO code is accepted");
        assert_eq!(fr.language(), "fr");
        assert!(!fr.auto_detects());
        let auto = fr.with_language(LANGUAGE_AUTO).expect("auto is accepted");
        assert!(auto.auto_detects());
    }

    #[test]
    fn transcribe_uses_utterance_pcm_and_turn() {
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder::new(&["and so my fellow americans"])),
            SttModel::Small,
        );
        let transcript = stt
            .transcribe(&speech_utterance(4, 2), &Cancel::new())
            .unwrap();
        assert_eq!(transcript.turn, TurnId(4));
        assert_eq!(transcript.text, "and so my fellow americans");
        // Fixed language: the transcript carries the configured code.
        assert_eq!(transcript.language, STT_LANGUAGE);
    }

    #[test]
    fn auto_detect_surfaces_the_detected_language() {
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder::new(&["bonjour"]).with_detected("fr")),
            SttModel::Small,
        )
        .with_language(LANGUAGE_AUTO)
        .unwrap();
        let transcript = stt
            .transcribe(&speech_utterance(1, 1), &Cancel::new())
            .unwrap();
        assert_eq!(transcript.language, "fr");
        assert_eq!(transcript.text, "bonjour");
    }

    #[test]
    fn empty_utterance_is_rejected_before_decode() {
        let calls = Arc::new(Mutex::new(0));
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["should not run".into()],
                detected: STT_LANGUAGE.into(),
                delay: Duration::ZERO,
                calls: Arc::clone(&calls),
            }),
            SttModel::Small,
        );
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(0),
                    frames: vec![],
                },
                &Cancel::new(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Provider {
                provider: "local",
                ..
            }
        ));
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn shutdown_before_transcribe_is_cancelled() {
        let mut stt =
            WhisperStt::with_decoder(Box::new(ScriptedDecoder::new(&["nope"])), SttModel::Small);
        let cancel = Cancel::new();
        cancel.shutdown();
        assert!(matches!(
            stt.transcribe(&speech_utterance(0, 1), &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn cancel_during_decode_releases_the_adapter() {
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["late".into()],
                detected: STT_LANGUAGE.into(),
                delay: Duration::from_secs(2),
                calls: Arc::new(Mutex::new(0)),
            }),
            SttModel::Small,
        );
        let cancel = Cancel::new();
        let cancel_thread = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            cancel_thread.shutdown();
        });
        let err = stt.transcribe(&speech_utterance(0, 1), &cancel);
        assert!(matches!(err, Err(Error::Cancelled)));
        drop(stt);
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_model_file_is_a_provider_error() {
        let err = match WhisperStt::from_model_path("/no/such/ggml-small.bin", SttModel::Small) {
            Err(err) => err,
            Ok(_) => panic!("missing model path should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: "whisper.cpp",
                ..
            }
        ));
    }

    #[test]
    fn from_cache_requires_the_selected_asset() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-stt-missing-{}-{}",
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
        let err = match WhisperStt::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::Small,
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("whisper-small"));
    }

    /// Records every requested URL, then refuses the network.
    struct RecordingFetcher {
        urls: std::sync::Mutex<Vec<String>>,
    }

    impl Default for RecordingFetcher {
        fn default() -> Self {
            Self {
                urls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl crate::models::Fetcher for RecordingFetcher {
        fn fetch(
            &self,
            url: &str,
            _writer: &mut dyn std::io::Write,
            _on_chunk: &mut dyn FnMut(u64),
            _cancel: &Cancel,
        ) -> crate::error::Result<()> {
            self.urls
                .lock()
                .expect("recording fetcher")
                .push(url.to_string());
            Err(Error::ModelCache {
                message: "no network in test".into(),
            })
        }
    }

    #[test]
    fn from_cache_fetches_only_the_selected_model_id() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-stt-selective-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(root, crate::models::Manifest::v0());
        let fetcher = RecordingFetcher::default();
        // Native load fails on absent bytes, but the fetch log is the point:
        // exactly one URL, the selected id, even though the manifest lists five.
        let result = WhisperStt::from_cache(
            &cache,
            &fetcher,
            &mut crate::models::NoProgress,
            &Cancel::new(),
            SttModel::MediumQ5_0,
        );
        assert!(result.is_err(), "no real weights in this test");
        let urls = fetcher.urls.lock().expect("urls");
        assert_eq!(urls.len(), 1, "only the selected id may be fetched");
        assert!(urls[0].ends_with("ggml-medium-q5_0.bin"), "{urls:?}");
        for other in [
            "ggml-small.bin",
            "ggml-medium.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3-turbo-q5_0.bin",
        ] {
            assert!(!urls[0].contains(other), "{urls:?} must skip {other}");
        }
    }

    #[test]
    fn fixture_expectation_matches_jfk_quote_in_order() {
        let spoken =
            "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.";
        assert!(contains_words_in_order(
            spoken,
            &[
                "and",
                "so",
                "my",
                "fellow",
                "americans",
                "ask",
                "not",
                "what",
                "your",
                "country",
                "can",
                "do",
                "for",
                "you",
                "ask",
                "what",
                "you",
                "can",
                "do",
                "for",
                "your",
                "country",
            ]
        ));
        assert!(!contains_words_in_order(spoken, &["ask", "not", "canada"]));
    }

    #[test]
    fn word_match_ratio_is_one_when_expected_is_empty() {
        assert_eq!(word_match_ratio("anything", &[]), 1.0);
        assert_eq!(transcript_words("12 abc-DEF!").as_slice(), ["abc", "def"]);
    }

    #[test]
    fn empty_pcm_is_rejected_before_decode() {
        let calls = Arc::new(Mutex::new(0));
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["should not run".into()],
                detected: STT_LANGUAGE.into(),
                delay: Duration::ZERO,
                calls: Arc::clone(&calls),
            }),
            SttModel::Small,
        );
        let utterance = Utterance {
            turn: TurnId(0),
            frames: vec![AudioFrame {
                seq: 0,
                sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
                channels: DEFAULT_CHANNELS,
                samples: vec![],
                capture_pcm: None,
            }],
        };
        let err = stt.transcribe(&utterance, &Cancel::new()).unwrap_err();
        assert!(matches!(
            err,
            Error::Provider {
                provider: "local",
                ..
            }
        ));
        assert!(err.to_string().contains("no samples"));
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn clone_shares_the_scripted_decoder() {
        let calls = Arc::new(Mutex::new(0));
        let stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["one".into(), "two".into()],
                detected: STT_LANGUAGE.into(),
                delay: Duration::ZERO,
                calls: Arc::clone(&calls),
            }),
            SttModel::Small,
        );
        let mut a = stt.clone();
        let mut b = stt;
        assert_eq!(
            a.transcribe(&speech_utterance(0, 1), &Cancel::new())
                .unwrap()
                .text,
            "one"
        );
        assert_eq!(
            b.transcribe(&speech_utterance(1, 1), &Cancel::new())
                .unwrap()
                .text,
            "two"
        );
        assert_eq!(*calls.lock().unwrap(), 2);
    }

    #[test]
    fn librispeech_gate_allows_numeral_for_number_word() {
        let spoken = "She has been dead these 20 years.";
        let expected = ["she", "has", "been", "dead", "these", "twenty", "years"];
        let ratio = word_match_ratio(spoken, &expected);
        assert!(
            ratio >= LIBRISPEECH_MIN_WORD_MATCH,
            "ratio {ratio} should pass the 80% LibriSpeech gate"
        );
        assert!(ratio < 1.0, "numeral should not be a perfect word match");
        assert!(!contains_words_in_order(spoken, &expected));
    }
}
