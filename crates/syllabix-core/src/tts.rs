//! In-process Kokoro ONNX text-to-speech.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::session::Session;

use crate::audio::{f32_to_i16, PcmConverter, PcmFormat};
use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::g2p::{english_to_kokoro_ids, pad_input_ids, KOKORO_MAX_PHONEME_TOKENS};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Tts;
use crate::speech_text::{strip_markdown_for_speech, take_sentences};
use crate::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId, DEFAULT_SAMPLE_RATE_HZ};

/// Manifest id for the Kokoro ONNX graph.
pub const KOKORO_ASSET: &str = "kokoro";

/// Manifest id for the default `af_heart` style table.
pub const KOKORO_VOICE_ASSET: &str = "kokoro-voice";

/// Kokoro waveform rate before conversion to the v0 16 kHz contract.
pub const KOKORO_NATIVE_RATE_HZ: u32 = 24_000;

/// Text → TTS → Whisper round-trip: at least 80% of reference words, in order.
pub const TTS_ASR_MIN_WORD_MATCH: f64 = 0.8;

const VOICE_ROWS: usize = 510;
const STYLE_DIM: usize = 256;
const VOICE_BYTES: usize = VOICE_ROWS * STYLE_DIM * 4;

/// In-process Kokoro adapter. Buffers tokens until a sentence boundary.
pub struct KokoroTts {
    engine: Arc<Mutex<Box<dyn WaveformEngine>>>,
    buffer: String,
    turn: Option<TurnId>,
    generation: Option<GenerationId>,
    next_index: u32,
}

impl Clone for KokoroTts {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            buffer: String::new(),
            turn: None,
            generation: None,
            next_index: 0,
        }
    }
}

impl KokoroTts {
    /// Load ONNX weights and the `af_heart` style table from disk.
    pub fn from_paths(model: impl AsRef<Path>, voice: impl AsRef<Path>) -> Result<Self> {
        let engine = OrtKokoro::load(model.as_ref(), voice.as_ref())?;
        Ok(Self::from_engine(Box::new(engine)))
    }

    /// Resolve Kokoro assets from the manifest cache, then load them.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let model = cache
            .manifest()
            .asset(KOKORO_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the kokoro asset".into(),
            })?;
        let voice =
            cache
                .manifest()
                .asset(KOKORO_VOICE_ASSET)
                .ok_or_else(|| Error::ModelCache {
                    message: "manifest does not contain the kokoro-voice asset".into(),
                })?;
        let model_path = cache.resolve(model, fetcher, progress, cancel)?;
        let voice_path = cache.resolve(voice, fetcher, progress, cancel)?;
        Self::from_paths(model_path, voice_path)
    }

    fn from_engine(engine: Box<dyn WaveformEngine>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            buffer: String::new(),
            turn: None,
            generation: None,
            next_index: 0,
        }
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn WaveformEngine>) -> Self {
        Self::from_engine(engine)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.turn = None;
        self.generation = None;
        self.next_index = 0;
    }

    fn emit(&mut self, samples: Vec<i16>, token: &TokenChunk, is_last: bool) -> SynthesizedAudio {
        let index = self.next_index;
        self.next_index += 1;
        SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index,
            samples,
            is_last,
        }
    }
}

impl Tts for KokoroTts {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().tts.as_str()
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        if cancel.is_stale(token.generation) {
            self.reset();
            return Err(Error::Cancelled);
        }
        if self.generation != Some(token.generation) || self.turn != Some(token.turn) {
            self.reset();
            self.generation = Some(token.generation);
            self.turn = Some(token.turn);
        }

        self.buffer.push_str(&token.text);
        let sentences = take_sentences(&mut self.buffer, token.is_last);
        let mut out = Vec::new();
        let last_i = sentences.len().saturating_sub(1);
        for (i, sentence) in sentences.into_iter().enumerate() {
            if cancel.is_stale(token.generation) {
                self.reset();
                return Err(Error::Cancelled);
            }
            let spoken = strip_markdown_for_speech(&sentence);
            if spoken.is_empty() {
                continue;
            }
            let samples = self
                .engine
                .lock()
                .expect("kokoro engine")
                .synthesize(&spoken, cancel)?;
            let is_last = token.is_last && i == last_i;
            out.push(self.emit(samples, token, is_last));
        }
        if token.is_last && out.is_empty() {
            out.push(self.emit(vec![0; 16], token, true));
        } else if token.is_last {
            if let Some(last) = out.last_mut() {
                last.is_last = true;
            }
        }
        Ok(out)
    }
}

trait WaveformEngine: Send {
    fn synthesize(&mut self, sentence: &str, cancel: &Cancel) -> Result<Vec<i16>>;
}

struct OrtKokoro {
    session: Session,
    voice: Vec<Vec<f32>>,
    model_path: PathBuf,
}

impl OrtKokoro {
    fn load(model: &Path, voice: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_error)?
            .commit_from_file(model)
            .map_err(ort_error)?;
        let input_names: Vec<&str> = session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        if input_names != ["input_ids", "style", "speed"] {
            return Err(Error::Provider {
                provider: "kokoro",
                message: format!("unexpected ONNX inputs: {}", input_names.join(", ")),
            });
        }
        Ok(Self {
            session,
            voice: load_voice(voice)?,
            model_path: model.to_path_buf(),
        })
    }

    fn infer(&mut self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        let style_index = phoneme_ids.len().min(self.voice.len().saturating_sub(1));
        let style = self.voice[style_index].clone();
        let input_ids = pad_input_ids(phoneme_ids);
        let n = input_ids.len();
        let outputs = self
            .session
            .run(
                ort::inputs![
                    "input_ids" => ([1_usize, n], input_ids),
                    "style" => ([1_usize, STYLE_DIM], style),
                    "speed" => ([1_usize], vec![1.0_f32]),
                ]
                .map_err(ort_error)?,
            )
            .map_err(ort_error)?;
        let samples = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect::<Vec<f32>>();
        if samples.is_empty() {
            return Err(Error::Provider {
                provider: "kokoro",
                message: format!(
                    "ONNX output waveform was empty ({})",
                    self.model_path.display()
                ),
            });
        }
        Ok(samples)
    }
}

impl WaveformEngine for OrtKokoro {
    fn synthesize(&mut self, sentence: &str, cancel: &Cancel) -> Result<Vec<i16>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut pcm = Vec::new();
        for ids in phoneme_windows(sentence)? {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if ids.is_empty() {
                continue;
            }
            let native = self.infer(&ids)?;
            pcm.extend(resample_to_v0(&native));
        }
        if pcm.is_empty() {
            pcm.push(0);
        }
        Ok(pcm)
    }
}

fn phoneme_windows(sentence: &str) -> Result<Vec<Vec<i64>>> {
    match english_to_kokoro_ids(sentence) {
        Ok(ids) => Ok(vec![ids]),
        Err(_) => {
            let mut windows = Vec::new();
            let mut acc = String::new();
            for word in sentence.split_whitespace() {
                let trial = if acc.is_empty() {
                    word.to_string()
                } else {
                    format!("{acc} {word}")
                };
                match english_to_kokoro_ids(&trial) {
                    Ok(_) => acc = trial,
                    Err(_) => {
                        if !acc.is_empty() {
                            windows.push(english_to_kokoro_ids(&acc)?);
                        }
                        acc = word.to_string();
                        if english_to_kokoro_ids(&acc)
                            .map(|ids| ids.len())
                            .unwrap_or(usize::MAX)
                            > KOKORO_MAX_PHONEME_TOKENS
                        {
                            return Err(Error::Provider {
                                provider: "kokoro",
                                message: "word exceeds Kokoro phoneme context".into(),
                            });
                        }
                    }
                }
            }
            if !acc.is_empty() {
                windows.push(english_to_kokoro_ids(&acc)?);
            }
            Ok(windows)
        }
    }
}

fn resample_to_v0(native: &[f32]) -> Vec<i16> {
    let mut conv = PcmConverter::new(
        PcmFormat {
            sample_rate_hz: KOKORO_NATIVE_RATE_HZ,
            channels: 1,
        },
        PcmFormat {
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: 1,
        },
    )
    .expect("24 kHz to 16 kHz is a valid conversion");
    let mut f32_pcm = conv.push(native);
    f32_pcm.extend(conv.flush());
    f32_to_i16(&f32_pcm)
}

fn load_voice(path: &Path) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path)?;
    if bytes.len() != VOICE_BYTES {
        return Err(Error::Provider {
            provider: "kokoro",
            message: format!(
                "af_heart.bin must be {VOICE_BYTES} bytes; got {}",
                bytes.len()
            ),
        });
    }
    let mut rows = Vec::with_capacity(VOICE_ROWS);
    for chunk in bytes.chunks_exact(STYLE_DIM * 4) {
        let mut style = vec![0.0_f32; STYLE_DIM];
        for (slot, fbytes) in style.iter_mut().zip(chunk.chunks_exact(4)) {
            *slot = f32::from_le_bytes(fbytes.try_into().expect("4-byte float"));
        }
        rows.push(style);
    }
    Ok(rows)
}

fn ort_error(error: ort::Error) -> Error {
    Error::Provider {
        provider: "kokoro",
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DEFAULT_CHANNELS;
    use std::sync::{Arc, Mutex};

    struct ScriptedEngine {
        calls: Arc<Mutex<Vec<String>>>,
        hz: f32,
    }

    impl ScriptedEngine {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                hz: 440.0,
            }
        }
    }

    impl WaveformEngine for ScriptedEngine {
        fn synthesize(&mut self, sentence: &str, cancel: &Cancel) -> Result<Vec<i16>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            self.calls.lock().expect("calls").push(sentence.to_string());
            Ok(crate::audio::sine_i16(
                DEFAULT_SAMPLE_RATE_HZ,
                DEFAULT_CHANNELS,
                self.hz,
                std::time::Duration::from_millis(80),
                0.4,
            ))
        }
    }

    fn token(text: &str, index: u32, is_last: bool) -> TokenChunk {
        TokenChunk {
            turn: TurnId(0),
            generation: GenerationId(0),
            index,
            text: text.into(),
            is_last,
        }
    }

    #[test]
    fn first_sentence_emits_before_last_token() {
        let mut tts = KokoroTts::with_engine(Box::new(ScriptedEngine::new()));
        let first = tts
            .synthesize_chunk(&token("Hello world. ", 0, false), &Cancel::new())
            .unwrap();
        assert_eq!(first.len(), 1);
        assert!(!first[0].is_last);
        assert!(first[0].samples.iter().any(|s| *s != 0));
        let rest = tts
            .synthesize_chunk(&token("More later.", 1, true), &Cancel::new())
            .unwrap();
        assert_eq!(rest.len(), 1);
        assert!(rest[0].is_last);
        assert_eq!(rest[0].index, 1);
    }

    #[test]
    fn markdown_is_stripped_before_synthesis() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = KokoroTts::with_engine(Box::new(engine));
        tts.synthesize_chunk(&token("# Hello **world**.", 0, true), &Cancel::new())
            .unwrap();
        let calls = log.lock().expect("calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], "Hello world.");
        assert!(!calls[0].contains('#'));
        assert!(!calls[0].contains('*'));
    }

    #[test]
    fn cancel_before_sentence_is_cancelled() {
        let mut tts = KokoroTts::with_engine(Box::new(ScriptedEngine::new()));
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = tts
            .synthesize_chunk(&token("Hello.", 0, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn empty_last_token_still_closes_the_turn() {
        let mut tts = KokoroTts::with_engine(Box::new(ScriptedEngine::new()));
        let chunks = tts
            .synthesize_chunk(&token("", 0, true), &Cancel::new())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn stale_generation_resets_and_cancels() {
        let mut tts = KokoroTts::with_engine(Box::new(ScriptedEngine::new()));
        tts.synthesize_chunk(&token("Hello. ", 0, false), &Cancel::new())
            .unwrap();
        let cancel = Cancel::new();
        cancel.cancel_generation();
        let err = tts
            .synthesize_chunk(&token("World.", 1, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn name_matches_v0_and_clone_drops_buffer() {
        let mut tts = KokoroTts::with_engine(Box::new(ScriptedEngine::new()));
        assert_eq!(tts.name(), "kokoro");
        tts.synthesize_chunk(&token("Hello ", 0, false), &Cancel::new())
            .unwrap();
        let mut cloned = tts.clone();
        let chunks = cloned
            .synthesize_chunk(&token("world.", 0, true), &Cancel::new())
            .unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn missing_onnx_path_is_a_provider_error() {
        let err = match KokoroTts::from_paths("/no/such/kokoro.onnx", "/no/such/af_heart.bin") {
            Err(err) => err,
            Ok(_) => panic!("missing onnx path should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: "kokoro",
                ..
            }
        ));
    }

    #[test]
    fn load_voice_rejects_wrong_size() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-voice-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("af_heart.bin");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let err = load_voice(&path).unwrap_err();
        assert!(matches!(
            err,
            Error::Provider {
                provider: "kokoro",
                ..
            }
        ));
        assert!(err.to_string().contains("bytes"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_voice_reads_fixed_style_table() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-voice-ok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("af_heart.bin");
        std::fs::write(&path, vec![0u8; VOICE_BYTES]).unwrap();
        let rows = load_voice(&path).unwrap();
        assert_eq!(rows.len(), VOICE_ROWS);
        assert_eq!(rows[0].len(), STYLE_DIM);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn phoneme_windows_split_overlong_sentences() {
        let windows = phoneme_windows(&"hello ".repeat(400)).unwrap();
        assert!(
            windows.len() > 1,
            "expected word-level split, got {windows:?}"
        );
        assert!(windows.iter().all(|w| !w.is_empty()));
    }

    #[test]
    fn resample_24k_to_16k_keeps_energy() {
        let native: Vec<f32> = (0..240).map(|i| (i as f32 / 24.0).sin()).collect();
        let pcm = resample_to_v0(&native);
        assert!(!pcm.is_empty());
        assert!(pcm.iter().any(|s| *s != 0));
    }

    #[test]
    fn from_cache_requires_kokoro_assets() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-tts-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = crate::models::ModelCache::new(
            root,
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match KokoroTts::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("kokoro"));
    }
}
