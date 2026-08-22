//! In-process text-to-speech: Kokoro ONNX (default) and Qwen3-TTS (row 31).
//!
//! Both providers share one sentence-chunking core: tokens buffer until a
//! sentence boundary, the think filter runs first, and each completed
//! sentence becomes one [`SynthesizedAudio`] through a [`WaveformEngine`].

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::session::Session;
use syllabix_native::{QwenTtsContext, QwenTtsError};

use crate::audio::{f32_to_i16, PcmConverter, PcmFormat};
use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::g2p::{english_to_kokoro_ids, pad_input_ids, KOKORO_MAX_PHONEME_TOKENS};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Tts;
use crate::speech_text::{speak_text_for_tts, take_sentences, ThinkFilter};
use crate::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId, DEFAULT_SAMPLE_RATE_HZ};

/// Manifest id for the Kokoro ONNX graph.
pub const KOKORO_ASSET: &str = "kokoro";

/// Manifest id for the default `af_heart` style table.
pub const KOKORO_VOICE_ASSET: &str = "kokoro-voice";

/// Kokoro waveform rate before conversion to the v0 16 kHz contract.
pub const KOKORO_NATIVE_RATE_HZ: u32 = 24_000;

/// Manifest id for the Qwen3-TTS backbone GGUF.
pub const QWEN_TTS_ASSET: &str = "qwen3-tts";

/// Manifest id for the Qwen3-TTS speech-tokenizer projector.
pub const QWEN_TTS_MMPROJ_ASSET: &str = "qwen3-tts-mmproj";

/// The blessed Qwen3-TTS backbone weight (row 31).
pub const QWEN_TTS_MODEL_ID: &str = "qwen3-tts-1.7b-base";

/// Text → TTS → Whisper round-trip: at least 80% of reference words, in order.
pub const TTS_ASR_MIN_WORD_MATCH: f64 = 0.8;

const VOICE_ROWS: usize = 510;
const STYLE_DIM: usize = 256;
const VOICE_BYTES: usize = VOICE_ROWS * STYLE_DIM * 4;

/// Per-generation sentence buffering shared by both providers.
#[derive(Default)]
struct ChunkState {
    think: ThinkFilter,
    buffer: String,
    turn: Option<TurnId>,
    generation: Option<GenerationId>,
    next_index: u32,
}

impl ChunkState {
    fn reset(&mut self) {
        self.think = ThinkFilter::default();
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

/// The one sentence-chunking loop. Tokens buffer until a sentence boundary;
/// think-strip runs first, then markdown strip, then the engine. An empty
/// final turn still emits a tiny closing chunk so playback can finish.
fn synthesize_chunk_shared(
    engine: &Mutex<Box<dyn WaveformEngine>>,
    state: &mut ChunkState,
    token: &TokenChunk,
    cancel: &Cancel,
) -> Result<Vec<SynthesizedAudio>> {
    if cancel.is_stale(token.generation) {
        state.reset();
        return Err(Error::Cancelled);
    }
    if state.generation != Some(token.generation) || state.turn != Some(token.turn) {
        state.reset();
        state.generation = Some(token.generation);
        state.turn = Some(token.turn);
    }

    state
        .buffer
        .push_str(&state.think.push(&token.text, token.is_last));
    let sentences = take_sentences(&mut state.buffer, token.is_last);
    let mut out = Vec::new();
    let last_i = sentences.len().saturating_sub(1);
    for (i, sentence) in sentences.into_iter().enumerate() {
        if cancel.is_stale(token.generation) {
            state.reset();
            return Err(Error::Cancelled);
        }
        let spoken = speak_text_for_tts(&sentence);
        if spoken.is_empty() {
            continue;
        }
        let samples =
            engine
                .lock()
                .expect("tts engine")
                .synthesize(&spoken, cancel, token.generation)?;
        let is_last = token.is_last && i == last_i;
        out.push(state.emit(samples, token, is_last));
    }
    if token.is_last && out.is_empty() {
        out.push(state.emit(vec![0; 16], token, true));
    } else if token.is_last {
        if let Some(last) = out.last_mut() {
            last.is_last = true;
        }
    }
    Ok(out)
}

/// In-process Kokoro adapter. Buffers tokens until a sentence boundary.
pub struct KokoroTts {
    core: ChunkState,
    engine: Arc<Mutex<Box<dyn WaveformEngine>>>,
}

impl Clone for KokoroTts {
    fn clone(&self) -> Self {
        Self {
            core: ChunkState::default(),
            engine: Arc::clone(&self.engine),
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
            core: ChunkState::default(),
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn WaveformEngine>) -> Self {
        Self::from_engine(engine)
    }
}

impl Tts for KokoroTts {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().tts.as_str()
    }

    fn model_id(&self) -> Option<&str> {
        Some(KOKORO_ASSET)
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        synthesize_chunk_shared(&self.engine, &mut self.core, token, cancel)
    }
}

/// Row 31 adapter: Qwen3-TTS through the shared llama.cpp/ggml path.
pub struct QwenTts {
    core: ChunkState,
    engine: Arc<Mutex<Box<dyn WaveformEngine>>>,
}

impl Clone for QwenTts {
    fn clone(&self) -> Self {
        Self {
            core: ChunkState::default(),
            engine: Arc::clone(&self.engine),
        }
    }
}

impl QwenTts {
    /// Load the backbone GGUF + mmproj from disk and speak `language`.
    pub fn from_paths(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        language: &str,
    ) -> Result<Self> {
        // Runtime sampling is randomized (llama.cpp default seed).
        Self::from_paths_with_seed(model, mmproj, language, u32::MAX)
    }

    /// [`QwenTts::from_paths`] with a pinned sampler seed. The native
    /// round-trip test uses this so fixtures reproduce; runtime callers use
    /// the random-seed path above.
    pub fn from_paths_with_seed(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        language: &str,
        seed: u32,
    ) -> Result<Self> {
        let engine = NativeQwen::load_with_seed(model.as_ref(), mmproj.as_ref(), language, seed)?;
        Ok(Self::from_engine(Box::new(engine)))
    }

    /// Resolve the two Qwen3-TTS assets from the manifest cache, then load.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        language: &str,
    ) -> Result<Self> {
        let model = cache
            .manifest()
            .asset(QWEN_TTS_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the qwen3-tts asset".into(),
            })?;
        let mmproj = cache
            .manifest()
            .asset(QWEN_TTS_MMPROJ_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the qwen3-tts-mmproj asset".into(),
            })?;
        let model_path = cache.resolve(model, fetcher, progress, cancel)?;
        let mmproj_path = cache.resolve(mmproj, fetcher, progress, cancel)?;
        Self::from_paths(model_path, mmproj_path, language)
    }

    fn from_engine(engine: Box<dyn WaveformEngine>) -> Self {
        Self {
            core: ChunkState::default(),
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn WaveformEngine>) -> Self {
        Self::from_engine(engine)
    }
}

impl Tts for QwenTts {
    fn name(&self) -> &'static str {
        "qwen"
    }

    fn model_id(&self) -> Option<&str> {
        Some(QWEN_TTS_MODEL_ID)
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        synthesize_chunk_shared(&self.engine, &mut self.core, token, cancel)
    }
}

trait WaveformEngine: Send {
    fn synthesize(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
    ) -> Result<Vec<i16>>;
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
    fn synthesize(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
    ) -> Result<Vec<i16>> {
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        let mut pcm = Vec::new();
        for ids in phoneme_windows(sentence)? {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            if ids.is_empty() {
                continue;
            }
            let native = self.infer(&ids)?;
            if cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            pcm.extend(resample_to_v0(&native));
        }
        if pcm.is_empty() {
            pcm.push(0);
        }
        Ok(pcm)
    }
}

/// Qwen3-TTS waveform engine over the shared ggml native path. One sentence
/// per call; the underlying context keeps no cross-sentence state.
struct NativeQwen {
    ctx: QwenTtsContext,
    lang: String,
}

impl NativeQwen {
    fn load_with_seed(model: &Path, mmproj: &Path, language: &str, seed: u32) -> Result<Self> {
        // Same cap as the LLM/STT engines: portable-CPU friendly.
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4) as i32)
            .unwrap_or(1);
        let ctx = QwenTtsContext::load(model, mmproj, n_threads, seed).map_err(|message| {
            Error::Provider {
                provider: "qwen",
                message,
            }
        })?;
        Ok(Self {
            ctx,
            lang: language.to_string(),
        })
    }
}

impl WaveformEngine for NativeQwen {
    fn synthesize(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
    ) -> Result<Vec<i16>> {
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        let abort_user = cancel as *const Cancel as *mut std::ffi::c_void;
        match unsafe {
            self.ctx
                .synthesize(sentence, &self.lang, Some(abort_on_shutdown), abort_user)
        } {
            Ok((rate, samples)) => {
                if cancel.is_stale(generation) || cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                Ok(resample_i16_to_v0(&samples, rate))
            }
            Err(QwenTtsError::Cancelled) => Err(Error::Cancelled),
            Err(QwenTtsError::Failed(_)) if cancel.is_shutdown() => Err(Error::Cancelled),
            Err(QwenTtsError::Failed(message)) => Err(Error::Provider {
                provider: "qwen",
                message,
            }),
        }
    }
}

unsafe extern "C" fn abort_on_shutdown(user_data: *mut std::ffi::c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // Safety: `user_data` is `&Cancel` for the duration of the synthesis.
    unsafe { (*(user_data as *const Cancel)).is_shutdown() }
}

/// Convert an engine's native-rate mono PCM to the v0 16 kHz contract.
fn resample_i16_to_v0(samples: &[i16], rate_hz: i32) -> Vec<i16> {
    let f32_pcm: Vec<f32> = samples.iter().map(|s| f32::from(*s) / 32_767.0).collect();
    let mut conv = PcmConverter::new(
        PcmFormat {
            sample_rate_hz: rate_hz.max(1) as u32,
            channels: 1,
        },
        PcmFormat {
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: 1,
        },
    )
    .expect("engine rates are valid conversions");
    let mut out_f32 = conv.push(&f32_pcm);
    out_f32.extend(conv.flush());
    let mut pcm = f32_to_i16(&out_f32);
    if pcm.is_empty() {
        pcm.push(0);
    }
    pcm
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
        fn synthesize(
            &mut self,
            sentence: &str,
            cancel: &Cancel,
            generation: GenerationId,
        ) -> Result<Vec<i16>> {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
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
    fn think_tags_never_reach_synthesis() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = KokoroTts::with_engine(Box::new(engine));
        tts.synthesize_chunk(
            &token("<think>do not say this.</think> Hello **world**.", 0, true),
            &Cancel::new(),
        )
        .unwrap();
        let calls = log.lock().expect("calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], "Hello world.");
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
    fn qwen_name_and_buffering_match_contract() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = QwenTts::with_engine(Box::new(engine));
        assert_eq!(tts.name(), "qwen");
        let first = tts
            .synthesize_chunk(&token("Hello world. ", 0, false), &Cancel::new())
            .unwrap();
        assert_eq!(first.len(), 1);
        assert!(!first[0].is_last);
        let rest = tts
            .synthesize_chunk(&token("More later.", 1, true), &Cancel::new())
            .unwrap();
        assert_eq!(rest.len(), 1);
        assert!(rest[0].is_last);
        // Think-strip and markdown-strip run before the qwen engine too.
        assert_eq!(
            log.lock().expect("calls").as_slice(),
            ["Hello world.", "More later."]
        );
    }

    #[test]
    fn qwen_strips_think_and_markdown_before_synthesis() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = QwenTts::with_engine(Box::new(engine));
        tts.synthesize_chunk(
            &token("<think>chain</think> It costs **$5**.", 0, true),
            &Cancel::new(),
        )
        .unwrap();
        let calls = log.lock().expect("calls");
        assert_eq!(calls.len(), 1);
        assert!(!calls[0].contains('<'));
        assert!(!calls[0].contains('*'));
    }

    #[test]
    fn qwen_cancel_paths_match_kokoro() {
        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()));
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = tts
            .synthesize_chunk(&token("Hello.", 0, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()));
        let cancel = Cancel::new();
        cancel.cancel_generation();
        let err = tts
            .synthesize_chunk(&token("World.", 0, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        // Empty final turn still closes.
        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()));
        let chunks = tts
            .synthesize_chunk(&token("", 0, true), &Cancel::new())
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn qwen_from_cache_requires_both_assets() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-qwen-missing-{}-{}",
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
        let err = match QwenTts::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            "en",
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("qwen3-tts"));
    }

    #[test]
    fn resample_i16_native_rates_reach_16k() {
        let rate = 24_000_i32;
        let native: Vec<i16> = (0..480)
            .map(|i| ((i as f32 / 24.0).sin() * 8192.0) as i16)
            .collect();
        let pcm = resample_i16_to_v0(&native, rate);
        assert!(!pcm.is_empty());
        assert!(pcm.iter().any(|s| *s != 0));
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
