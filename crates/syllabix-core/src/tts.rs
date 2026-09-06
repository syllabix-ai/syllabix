//! In-process text-to-speech using Pocket TTS, Kokoro ONNX, or Qwen3-TTS.
//!
//! Both providers share one sentence-chunking core: tokens buffer until a
//! sentence boundary, the think filter runs first, and each completed
//! sentence becomes one [`SynthesizedAudio`] through a [`WaveformEngine`].

#[cfg(any(not(coverage), test))]
use std::fs;
use std::path::Path;
#[cfg(not(coverage))]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[cfg(not(coverage))]
use ort::session::Session;
#[cfg(not(coverage))]
use syllabix_native::{QwenTtsContext, QwenTtsError};

#[cfg(any(not(coverage), test))]
use crate::audio::{f32_to_i16, PcmConverter, PcmFormat};
use crate::cancel::Cancel;
use crate::defaults::{BuiltinDefaults, TtsModel};
use crate::error::{Error, Result};
#[cfg(not(coverage))]
use crate::g2p::pad_input_ids;
#[cfg(any(not(coverage), test))]
use crate::g2p::{english_to_kokoro_ids, KOKORO_MAX_PHONEME_TOKENS};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Tts;
use crate::speech_text::{speak_text_for_tts, take_sentences, ThinkFilter};
#[cfg(any(not(coverage), test))]
use crate::types::DEFAULT_SAMPLE_RATE_HZ;
use crate::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId};

/// Manifest id for the Kokoro ONNX graph.
pub const KOKORO_ASSET: &str = "kokoro";

/// Manifest id for the default `af_heart` style table.
pub const KOKORO_VOICE_ASSET: &str = "kokoro-voice";

/// Kokoro waveform rate before conversion to the 16 kHz pipeline rate.
pub const KOKORO_NATIVE_RATE_HZ: u32 = 24_000;

/// Manifest id for the Qwen3-TTS backbone GGUF.
pub const QWEN_TTS_ASSET: &str = "qwen3-tts";

/// Manifest id for the 0.6B Qwen3-TTS backbone GGUF. Pairs with
/// [`QWEN_TTS_06B_MMPROJ_ASSET`].
pub const QWEN_TTS_06B_ASSET: &str = "qwen3-tts-06b";

/// Manifest id for the 0.6B speech-tokenizer projector. The tokenizer
/// *encoder* is byte-identical across both backbone sizes, but the mmproj
/// also carries the projector into the LM embedding space (2048-d vs
/// 1024-d), so each backbone needs its own mmproj file.
pub const QWEN_TTS_06B_MMPROJ_ASSET: &str = "qwen3-tts-06b-mmproj";

/// Manifest id for the Qwen3-TTS speech-tokenizer projector (1.7B pairing).
pub const QWEN_TTS_MMPROJ_ASSET: &str = "qwen3-tts-mmproj";

/// Text → TTS → Whisper round-trip: at least 80% of reference words, in order.
pub const TTS_ASR_MIN_WORD_MATCH: f64 = 0.8;

#[cfg(any(not(coverage), test))]
const VOICE_ROWS: usize = 510;
#[cfg(any(not(coverage), test))]
const STYLE_DIM: usize = 256;
#[cfg(any(not(coverage), test))]
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

/// Qwen3-TTS adapter using the shared
/// llama.cpp/ggml path. The native engine pins a self-generated voice anchor
/// at load, so one speaker holds across every sentence of every run.
pub struct QwenTts {
    core: ChunkState,
    engine: Arc<Mutex<Box<dyn WaveformEngine>>>,
    model: TtsModel,
}

impl Clone for QwenTts {
    fn clone(&self) -> Self {
        Self {
            core: ChunkState::default(),
            engine: Arc::clone(&self.engine),
            model: self.model,
        }
    }
}

impl QwenTts {
    /// Load the backbone GGUF + mmproj from disk and speak `language`.
    /// `selected` picks the loaded weight for logs and the diagnostics sidecar.
    pub fn from_paths(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        language: &str,
        selected: TtsModel,
    ) -> Result<Self> {
        // Runtime sampling is randomized (llama.cpp default seed); the voice
        // identity comes from the native engine's pinned self-voice anchor.
        Self::from_paths_with_seed(model, mmproj, language, selected, u32::MAX)
    }

    /// [`QwenTts::from_paths`] with a pinned sampler seed. The native
    /// round-trip test uses this so fixtures reproduce; runtime callers use
    /// the random-seed path above.
    pub fn from_paths_with_seed(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        language: &str,
        selected: TtsModel,
        seed: u32,
    ) -> Result<Self> {
        let engine = NativeQwen::load_with_seed(model.as_ref(), mmproj.as_ref(), language, seed)?;
        Ok(Self::from_engine(Box::new(engine), selected))
    }

    /// Resolve the selected Qwen3-TTS backbone plus its matching mmproj from
    /// the manifest cache, then load. Only the selected pair is fetched.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        language: &str,
        selected: TtsModel,
    ) -> Result<Self> {
        let backbone =
            cache
                .manifest()
                .asset(selected.asset_id())
                .ok_or_else(|| Error::ModelCache {
                    message: format!(
                        "manifest does not contain the {} asset",
                        selected.asset_id()
                    ),
                })?;
        let mmproj_id = selected.mmproj_asset_id().unwrap_or(QWEN_TTS_MMPROJ_ASSET);
        let mmproj = cache
            .manifest()
            .asset(mmproj_id)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {mmproj_id} asset"),
            })?;
        let model_path = cache.resolve(backbone, fetcher, progress, cancel)?;
        let mmproj_path = cache.resolve(mmproj, fetcher, progress, cancel)?;
        Self::from_paths(model_path, mmproj_path, language, selected)
    }

    fn from_engine(engine: Box<dyn WaveformEngine>, model: TtsModel) -> Self {
        Self {
            core: ChunkState::default(),
            engine: Arc::new(Mutex::new(engine)),
            model,
        }
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn WaveformEngine>, model: TtsModel) -> Self {
        Self::from_engine(engine, model)
    }

    /// Whether the native self-voice anchor engaged at load. `false` means
    /// synthesis fell back to unconditioned sampling (voice may drift).
    pub fn voice_anchor_engaged(&self) -> bool {
        self.engine.lock().expect("tts engine").has_voice_anchor()
    }
}

impl Tts for QwenTts {
    fn name(&self) -> &'static str {
        // The sidecar uses the same local/online provider vocabulary as the LLM.
        "local"
    }

    fn model_id(&self) -> Option<&str> {
        Some(self.model.model_id())
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        let mut out = Vec::new();
        self.synthesize_chunk_into(token, cancel, &mut |audio| {
            out.push(audio);
            Ok(())
        })?;
        Ok(out)
    }

    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_stale(token.generation) {
            self.core.reset();
            return Err(Error::Cancelled);
        }
        if self.core.generation != Some(token.generation) || self.core.turn != Some(token.turn) {
            self.core.reset();
            self.core.generation = Some(token.generation);
            self.core.turn = Some(token.turn);
        }
        self.core
            .buffer
            .push_str(&self.core.think.push(&token.text, token.is_last));
        // Qwen owns prosody across punctuation. It receives the whole reply,
        // not sentence or clause fragments, then streams its vocoder PCM.
        if !token.is_last {
            return Ok(());
        }
        let spoken = speak_text_for_tts(&self.core.buffer);
        self.core.buffer.clear();
        if spoken.is_empty() {
            on_audio(self.core.emit(vec![0; 16], token, true))?;
            return Ok(());
        }
        let mut emitted = false;
        self.engine
            .lock()
            .expect("tts engine")
            .synthesize_streaming(
                &spoken,
                cancel,
                token.generation,
                &mut |samples, is_last| {
                    if samples.is_empty() && !is_last {
                        return Ok(());
                    }
                    emitted = true;
                    on_audio(self.core.emit(samples, token, is_last))
                },
            )?;
        if !emitted {
            on_audio(self.core.emit(vec![0; 16], token, true))?;
        }
        Ok(())
    }
}

trait WaveformEngine: Send {
    fn synthesize(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
    ) -> Result<Vec<i16>>;

    fn synthesize_streaming(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
        on_audio: &mut dyn FnMut(Vec<i16>, bool) -> Result<()>,
    ) -> Result<()> {
        on_audio(self.synthesize(sentence, cancel, generation)?, true)
    }

    /// Whether a pinned speaker identity is active (Qwen3-TTS self-voice
    /// anchor; Kokoro's fixed ONNX voice counts trivially as `false` here
    /// because identity is baked into the weights, not sampled).
    fn has_voice_anchor(&self) -> bool {
        false
    }
}

#[cfg(not(coverage))]
struct OrtKokoro {
    session: Session,
    voice: Vec<Vec<f32>>,
    model_path: PathBuf,
}

#[cfg(not(coverage))]
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

#[cfg(not(coverage))]
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
#[cfg(not(coverage))]
struct NativeQwen {
    ctx: QwenTtsContext,
    lang: String,
    voice_active: bool,
}

#[cfg(not(coverage))]
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
            voice_active: ctx.has_voice(),
            ctx,
            lang: language.to_string(),
        })
    }
}

#[cfg(not(coverage))]
impl WaveformEngine for NativeQwen {
    fn has_voice_anchor(&self) -> bool {
        self.voice_active
    }

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

    fn synthesize_streaming(
        &mut self,
        sentence: &str,
        cancel: &Cancel,
        generation: GenerationId,
        on_audio: &mut dyn FnMut(Vec<i16>, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        let abort_user = cancel as *const Cancel as *mut std::ffi::c_void;
        let mut converter: Option<PcmConverter> = None;
        let mut callback_error: Option<Error> = None;
        let result = unsafe {
            self.ctx.synthesize_streaming(
                sentence,
                &self.lang,
                Some(abort_on_shutdown),
                abort_user,
                &mut |rate, pcm, is_last| {
                    if cancel.is_shutdown() || cancel.is_stale(generation) {
                        return Err(QwenTtsError::Cancelled);
                    }
                    let conv = converter.get_or_insert_with(|| {
                        PcmConverter::new(
                            PcmFormat {
                                sample_rate_hz: rate.max(1) as u32,
                                channels: 1,
                            },
                            PcmFormat {
                                sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
                                channels: 1,
                            },
                        )
                        .expect("native Qwen rate is valid")
                    });
                    let mut out = f32_to_i16(&conv.push(pcm));
                    if is_last {
                        out.extend(f32_to_i16(&conv.flush()));
                    }
                    if !out.is_empty() || is_last {
                        if let Err(err) = on_audio(out, is_last) {
                            callback_error = Some(err);
                            return Err(QwenTtsError::Cancelled);
                        }
                    }
                    Ok(())
                },
            )
        };
        if let Some(err) = callback_error {
            return Err(err);
        }
        match result {
            Ok(()) => Ok(()),
            Err(QwenTtsError::Cancelled) => Err(Error::Cancelled),
            Err(QwenTtsError::Failed(_message))
                if cancel.is_shutdown() || cancel.is_stale(generation) =>
            {
                Err(Error::Cancelled)
            }
            Err(QwenTtsError::Failed(message)) => Err(Error::Provider {
                provider: "qwen",
                message,
            }),
        }
    }
}

#[cfg(coverage)]
struct OrtKokoro;

#[cfg(coverage)]
impl OrtKokoro {
    fn load(_model: &Path, _voice: &Path) -> Result<Self> {
        Err(Error::Provider {
            provider: "kokoro",
            message: "Kokoro inference is not loaded in coverage tests.".into(),
        })
    }
}

#[cfg(coverage)]
impl WaveformEngine for OrtKokoro {
    fn synthesize(
        &mut self,
        _sentence: &str,
        _cancel: &Cancel,
        _generation: GenerationId,
    ) -> Result<Vec<i16>> {
        Err(Error::Provider {
            provider: "kokoro",
            message: "Kokoro inference is not loaded in coverage tests.".into(),
        })
    }
}

#[cfg(coverage)]
struct NativeQwen;

#[cfg(coverage)]
impl NativeQwen {
    fn load_with_seed(_model: &Path, _mmproj: &Path, _language: &str, _seed: u32) -> Result<Self> {
        Err(Error::Provider {
            provider: "qwen",
            message: "Qwen inference is not loaded in coverage tests.".into(),
        })
    }
}

#[cfg(coverage)]
impl WaveformEngine for NativeQwen {
    fn has_voice_anchor(&self) -> bool {
        false
    }

    fn synthesize(
        &mut self,
        _sentence: &str,
        _cancel: &Cancel,
        _generation: GenerationId,
    ) -> Result<Vec<i16>> {
        Err(Error::Provider {
            provider: "qwen",
            message: "Qwen inference is not loaded in coverage tests.".into(),
        })
    }
}

#[cfg(not(coverage))]
unsafe extern "C" fn abort_on_shutdown(user_data: *mut std::ffi::c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // Safety: `user_data` is `&Cancel` for the duration of the synthesis.
    unsafe { (*(user_data as *const Cancel)).is_shutdown() }
}

/// Convert an engine's native-rate mono PCM to the 16 kHz pipeline format.
#[cfg(any(not(coverage), test))]
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

#[cfg(any(not(coverage), test))]
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

#[cfg(any(not(coverage), test))]
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

#[cfg(any(not(coverage), test))]
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

#[cfg(not(coverage))]
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
        // `name()` reports where inference runs; the engine identity
        // stays in `model_id()`.
        assert_eq!(tts.name(), "local");
        assert_eq!(tts.model_id(), Some("kokoro"));
        tts.synthesize_chunk(&token("Hello ", 0, false), &Cancel::new())
            .unwrap();
        let mut cloned = tts.clone();
        let chunks = cloned
            .synthesize_chunk(&token("world.", 0, true), &Cancel::new())
            .unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn qwen_buffers_full_reply_and_preserves_punctuation_prosody() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = QwenTts::with_engine(Box::new(engine), TtsModel::Qwen06);
        // The sidecar uses the same local/online provider vocabulary as the LLM.
        assert_eq!(tts.name(), "local");
        assert_eq!(tts.model_id(), Some("qwen3-tts-0.6b-base"));
        assert!(!tts.voice_anchor_engaged(), "scripted engine has no anchor");
        let first = tts
            .synthesize_chunk(&token("Hello world. ", 0, false), &Cancel::new())
            .unwrap();
        assert!(first.is_empty(), "Qwen must not split at a period");
        let rest = tts
            .synthesize_chunk(&token("More later.", 1, true), &Cancel::new())
            .unwrap();
        assert_eq!(rest.len(), 1);
        assert!(rest[0].is_last);
        // Think-strip and markdown-strip run before the full Qwen utterance.
        assert_eq!(
            log.lock().expect("calls").as_slice(),
            ["Hello world. More later."]
        );
    }

    #[test]
    fn qwen_model_ids_follow_the_menu() {
        let engine = ScriptedEngine::new();
        let tts = QwenTts::with_engine(Box::new(engine), TtsModel::Qwen17);
        assert_eq!(tts.model_id(), Some("qwen3-tts-1.7b-base"));
    }

    #[test]
    fn qwen_strips_think_and_markdown_before_synthesis() {
        let engine = ScriptedEngine::new();
        let log = Arc::clone(&engine.calls);
        let mut tts = QwenTts::with_engine(Box::new(engine), TtsModel::Qwen06);
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
        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()), TtsModel::Qwen06);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = tts
            .synthesize_chunk(&token("Hello.", 0, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()), TtsModel::Qwen06);
        let cancel = Cancel::new();
        cancel.cancel_generation();
        let err = tts
            .synthesize_chunk(&token("World.", 0, true), &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));

        // Empty final turn still closes.
        let mut tts = QwenTts::with_engine(Box::new(ScriptedEngine::new()), TtsModel::Qwen06);
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
            TtsModel::Qwen06,
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("qwen3-tts-06b"));
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

    #[cfg(coverage)]
    #[test]
    fn coverage_native_tts_seams_are_callable() {
        let cancel = Cancel::new();
        let mut kokoro = OrtKokoro;
        assert!(kokoro
            .synthesize("hello", &cancel, GenerationId(0))
            .is_err());
        let mut qwen = NativeQwen;
        assert!(!qwen.has_voice_anchor());
        assert!(qwen.synthesize("hello", &cancel, GenerationId(0)).is_err());
    }
}
