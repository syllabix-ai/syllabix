//! In-process whisper.cpp speech-to-text for completed VAD utterances.

use std::path::Path;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

use crate::defaults::{BuiltinDefaults, SttModel};
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Stt;
use crate::types::{Transcript, Utterance};
use crate::Cancel;

/// v0 STT language. YAML language selection lands in a later PR.
pub const STT_LANGUAGE: &str = "en";

/// Manifest id for whisper.cpp `small`.
pub const WHISPER_SMALL_ASSET: &str = "whisper-small";

/// In-process whisper.cpp adapter. Loads the v0 `small` GGML weights.
pub struct WhisperStt {
    decoder: Box<dyn Decoder>,
    model: SttModel,
}

impl WhisperStt {
    /// Load a ggml model from disk. v0 always uses [`SttModel::Small`].
    pub fn from_model_path(path: impl AsRef<Path>, model: SttModel) -> Result<Self> {
        let decoder = WhisperDecoder::load(path.as_ref())?;
        Ok(Self {
            decoder: Box::new(decoder),
            model,
        })
    }

    /// Resolve `whisper-small` from the manifest cache, then load it.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let asset =
            cache
                .manifest()
                .asset(WHISPER_SMALL_ASSET)
                .ok_or_else(|| Error::ModelCache {
                    message: "manifest does not contain the whisper-small asset".into(),
                })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_model_path(path, SttModel::Small)
    }

    /// Configured whisper.cpp model id (`small`).
    pub fn model(&self) -> SttModel {
        self.model
    }

    #[cfg(test)]
    fn with_decoder(decoder: Box<dyn Decoder>, model: SttModel) -> Self {
        Self { decoder, model }
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
        let mut audio = vec![0.0f32; pcm.len()];
        whisper_rs::convert_integer_to_float_audio(&pcm, &mut audio).map_err(whisper_error)?;
        let text = self.decoder.decode(&audio, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(Transcript {
            turn: utterance.turn,
            text,
        })
    }
}

trait Decoder: Send {
    fn decode(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String>;
}

struct WhisperDecoder {
    ctx: WhisperContext,
}

impl WhisperDecoder {
    fn load(path: &Path) -> Result<Self> {
        hush_whisper_logs();
        let path = path.to_str().ok_or_else(|| Error::Provider {
            provider: "whisper.cpp",
            message: format!("model path is not valid UTF-8: {}", path.display()),
        })?;
        let mut params = WhisperContextParameters::default();
        params.use_gpu(false);
        let ctx = WhisperContext::new_with_params(path, params).map_err(whisper_error)?;
        Ok(Self { ctx })
    }
}

impl Decoder for WhisperDecoder {
    fn decode(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut state = self.ctx.create_state().map_err(whisper_error)?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(STT_LANGUAGE));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_nst(true);
        params.set_n_threads(thread_count());
        unsafe {
            params.set_abort_callback(Some(abort_on_shutdown));
            params.set_abort_callback_user_data(cancel as *const Cancel as *mut std::ffi::c_void);
        }
        match state.full(params, pcm) {
            Ok(_) => {}
            Err(_) if cancel.is_shutdown() => return Err(Error::Cancelled),
            Err(err) => return Err(whisper_error(err)),
        }
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut text = String::new();
        for index in 0..state.full_n_segments() {
            let Some(segment) = state.get_segment(index) else {
                continue;
            };
            let piece = segment.to_str_lossy().map_err(whisper_error)?;
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(piece);
        }
        drop(state);
        Ok(text)
    }
}

fn hush_whisper_logs() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(whisper_rs::install_logging_hooks);
}

unsafe extern "C" fn abort_on_shutdown(user_data: *mut std::ffi::c_void) -> bool {
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

fn whisper_error(error: WhisperError) -> Error {
    Error::Provider {
        provider: "whisper.cpp",
        message: error.to_string(),
    }
}

/// Lowercase alphabetic words in `text`, in order. Used by the fixture merge gate.
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

/// LibriSpeech fixture gate: at least 80% of official transcript words, in order.
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
        delay: Duration,
        calls: Arc<Mutex<usize>>,
    }

    impl Decoder for ScriptedDecoder {
        fn decode(&mut self, _pcm: &[f32], cancel: &Cancel) -> Result<String> {
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
            Ok(self.replies.remove(0))
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
        let stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["hello".into()],
                delay: Duration::ZERO,
                calls: Arc::new(Mutex::new(0)),
            }),
            SttModel::Small,
        );
        assert_eq!(stt.name(), "whisper.cpp");
        assert_eq!(stt.model().as_str(), "small");
        assert_eq!(STT_LANGUAGE, "en");
        assert_eq!(WHISPER_SMALL_ASSET, "whisper-small");
    }

    #[test]
    fn transcribe_uses_utterance_pcm_and_turn() {
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["and so my fellow americans".into()],
                delay: Duration::ZERO,
                calls: Arc::new(Mutex::new(0)),
            }),
            SttModel::Small,
        );
        let transcript = stt
            .transcribe(&speech_utterance(4, 2), &Cancel::new())
            .unwrap();
        assert_eq!(transcript.turn, TurnId(4));
        assert_eq!(transcript.text, "and so my fellow americans");
    }

    #[test]
    fn empty_utterance_is_rejected_before_decode() {
        let calls = Arc::new(Mutex::new(0));
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["should not run".into()],
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
                provider: "whisper.cpp",
                ..
            }
        ));
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn shutdown_before_transcribe_is_cancelled() {
        let mut stt = WhisperStt::with_decoder(
            Box::new(ScriptedDecoder {
                replies: vec!["nope".into()],
                delay: Duration::ZERO,
                calls: Arc::new(Mutex::new(0)),
            }),
            SttModel::Small,
        );
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
    fn from_cache_requires_whisper_small_asset() {
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
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("whisper-small"));
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
