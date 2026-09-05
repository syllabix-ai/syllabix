//! Moonshine streaming STT via ONNX Runtime (small and medium).
//!
//! Batch finalize only: one completed VAD [`Utterance`] in, one [`Transcript`]
//! out — the same contract as [`WhisperStt`](crate::stt::WhisperStt). The VAD
//! turn policy is untouched (`threshold` + `min_speech_ms` open the turn,
//! shared `end_silence_ms` closes it). Live partials arrive in the follow-up
//! step; this module proves weights, decode loop, and the SentencePiece-style
//! tokenizer first.
//!
//! Weights are the community INT8 exports of
//! `moonshine-ai/moonshine-streaming-small` and
//! `moonshine-ai/moonshine-streaming-medium` (MIT): one encoder plus a first-step
//! decoder and a KV-cache decoder, with `tokenizer.json`. English only.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use ort::{session::Session, AsPointer};
use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Stt;
use crate::types::{AudioFrame, Transcript, TurnId, Utterance};
use crate::Cancel;

/// Config / log name for this engine.
pub const PROVIDER_NAME: &str = "moonshine";
/// Default STT language. Any other yaml value is rejected at load.
pub const LANGUAGE: &str = "en";

pub const ENCODER_ASSET: &str = "moonshine-encoder";
pub const DECODER_ASSET: &str = "moonshine-decoder";
pub const DECODER_PAST_ASSET: &str = "moonshine-decoder-past";
pub const TOKENIZER_ASSET: &str = "moonshine-tokenizer";

pub const MEDIUM_ENCODER_ASSET: &str = "moonshine-medium-encoder";
pub const MEDIUM_DECODER_ASSET: &str = "moonshine-medium-decoder";
pub const MEDIUM_DECODER_PAST_ASSET: &str = "moonshine-medium-decoder-past";
pub const MEDIUM_TOKENIZER_ASSET: &str = "moonshine-medium-tokenizer";

const BOS: i64 = 1;
const EOS: i64 = 2;
/// Audio is padded up to a multiple of 80 samples for the encoder.
const ENCODER_PAD: usize = 80;
/// Upper bound on output tokens per second of audio (model-card rule is 6.5).
const MAX_TOKENS_PER_SECOND: f64 = 6.5;
/// Absolute ceiling so a runaway loop cannot spin forever.
const MAX_TOKENS_ABSOLUTE: usize = 1024;
/// Decode often enough to feel live without monopolising the STT worker. At
/// 16 kHz / 512-sample capture frames this is roughly half a second.
const PARTIAL_EVERY_FRAMES: usize = 16;

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

fn ort_error(error: ort::Error) -> Error {
    provider(error.to_string())
}

/// BPE decoder for the Moonshine 32k `tokenizer.json`.
///
/// Use the model's tokenizer implementation rather than reimplementing byte
/// fallback and special-token rules beside the ONNX graph.
#[derive(Debug, Clone)]
pub struct MoonshineTokenizer {
    inner: Tokenizer,
}

impl MoonshineTokenizer {
    /// Load `tokenizer.json` from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        Tokenizer::from_file(path)
            .map(|inner| Self { inner })
            .map_err(|err| {
                provider(format!(
                    "could not load tokenizer {}: {err}",
                    path.display()
                ))
            })
    }

    /// Parse `tokenizer.json` text. Kept separate so unit tests can use a
    /// synthetic vocabulary without shipping weights.
    pub fn from_json(text: &str) -> Result<Self> {
        Tokenizer::from_bytes(text.as_bytes())
            .map(|inner| Self { inner })
            .map_err(|err| provider(format!("bad tokenizer.json: {err}")))
    }

    /// Decode one greedy id sequence into spoken text.
    pub fn decode(&self, ids: &[i64]) -> String {
        let ids: Vec<u32> = ids
            .iter()
            .take_while(|id| **id != EOS)
            .filter_map(|id| u32::try_from(*id).ok())
            .collect();
        self.inner.decode(&ids, true).unwrap_or_default()
    }
}

/// Minimal C-API runner for dynamic-shape tensors (KV cache, variable-length
/// audio). Mirrors the `pocket_tts` escape hatch: the high-level `ort::inputs!`
/// macro cannot name 42 KV inputs that change shape every decode step.
struct RawTensor {
    shape: Vec<i64>,
    data: RawData,
}

enum RawData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl RawTensor {
    fn f32(shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            shape,
            data: RawData::F32(data),
        }
    }

    fn i64(shape: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            shape,
            data: RawData::I64(data),
        }
    }
}

struct RawValue(*mut ort_sys::OrtValue);
impl Drop for RawValue {
    fn drop(&mut self) {
        unsafe { ort::ortsys!(ReleaseValue)(self.0) };
    }
}

struct RawRunner<'a> {
    session: &'a mut Session,
    allocator: *mut ort_sys::OrtAllocator,
}

impl<'a> RawRunner<'a> {
    fn new(session: &'a mut Session) -> Result<Self> {
        let mut allocator = ptr::null_mut();
        check(unsafe { ort::ortsys!(GetAllocatorWithDefaultOptions)(&mut allocator) })?;
        Ok(Self { session, allocator })
    }

    fn run(&mut self, inputs: &[(&str, &RawTensor)], outputs: &[&str]) -> Result<Vec<RawTensor>> {
        let names = inputs
            .iter()
            .map(|(n, _)| CString::new(*n).map_err(|_| provider("invalid input name")))
            .collect::<Result<Vec<_>>>()?;
        let out_names = outputs
            .iter()
            .map(|n| CString::new(*n).map_err(|_| provider("invalid output name")))
            .collect::<Result<Vec<_>>>()?;
        let values = inputs
            .iter()
            .map(|(_, t)| self.value(t))
            .collect::<Result<Vec<_>>>()?;
        let in_names = names
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<*const c_char>>();
        let in_values = values.iter().map(|v| v.0.cast_const()).collect::<Vec<_>>();
        let out_ptrs = out_names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        let mut raw_outputs = vec![ptr::null_mut(); outputs.len()];
        check(unsafe {
            ort::ortsys!(Run)(
                self.session.ptr_mut(),
                ptr::null(),
                in_names.as_ptr(),
                in_values.as_ptr(),
                in_values.len(),
                out_ptrs.as_ptr(),
                out_ptrs.len(),
                raw_outputs.as_mut_ptr(),
            )
        })?;
        raw_outputs
            .into_iter()
            .map(|p| self.read(RawValue(p)))
            .collect()
    }

    fn value(&self, tensor: &RawTensor) -> Result<RawValue> {
        let (element_type, byte_len, source): (
            ort_sys::ONNXTensorElementDataType,
            usize,
            *const c_void,
        ) = match &tensor.data {
            RawData::F32(data) => (
                ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                data.len() * size_of::<f32>(),
                data.as_ptr().cast(),
            ),
            RawData::I64(data) => (
                ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                data.len() * size_of::<i64>(),
                data.as_ptr().cast(),
            ),
        };
        let mut value = ptr::null_mut();
        check(unsafe {
            ort::ortsys!(CreateTensorAsOrtValue)(
                self.allocator,
                tensor.shape.as_ptr(),
                tensor.shape.len(),
                element_type,
                &mut value,
            )
        })?;
        let value = RawValue(value);
        if byte_len != 0 {
            let mut data: *mut c_void = ptr::null_mut();
            check(unsafe { ort::ortsys!(GetTensorMutableData)(value.0, &mut data) })?;
            unsafe {
                ptr::copy_nonoverlapping(source.cast::<u8>(), data.cast::<u8>(), byte_len);
            }
        }
        Ok(value)
    }

    fn read(&self, value: RawValue) -> Result<RawTensor> {
        let mut info = ptr::null_mut();
        check(unsafe { ort::ortsys!(GetTensorTypeAndShape)(value.0, &mut info) })?;
        let mut rank = 0;
        check(unsafe { ort::ortsys!(GetDimensionsCount)(info, &mut rank) })?;
        let mut element_type =
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        check(unsafe { ort::ortsys!(GetTensorElementType)(info, &mut element_type) })?;
        let mut shape = vec![0_i64; rank];
        check(unsafe { ort::ortsys!(GetDimensions)(info, shape.as_mut_ptr(), rank) })?;
        unsafe { ort::ortsys!(ReleaseTensorTypeAndShapeInfo)(info) };
        let len = shape
            .iter()
            .try_fold(1_usize, |a, d| {
                usize::try_from(*d).ok().and_then(|d| a.checked_mul(d))
            })
            .ok_or_else(|| provider("invalid output shape"))?;
        let mut data: *mut c_void = ptr::null_mut();
        if len != 0 {
            check(unsafe { ort::ortsys!(GetTensorMutableData)(value.0, &mut data) })?;
        }
        match element_type {
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                let mut out = vec![0_f32; len];
                if len != 0 {
                    unsafe {
                        ptr::copy_nonoverlapping(data.cast::<f32>(), out.as_mut_ptr(), len);
                    }
                }
                Ok(RawTensor::f32(shape, out))
            }
            _ => Err(provider("unexpected non-f32 decoder output")),
        }
    }
}

fn check(status: ort_sys::OrtStatusPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let message = unsafe {
        CStr::from_ptr(ort::ortsys!(GetErrorMessage)(status))
            .to_string_lossy()
            .into_owned()
    };
    unsafe { ort::ortsys!(ReleaseStatus)(status) };
    Err(provider(format!("ONNX Runtime C API: {message}")))
}

fn load_session(path: &Path, label: &str) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(|err| provider(format!("could not load {label} {}: {err}", path.display())))
}

fn session_names(session: &Session) -> Vec<String> {
    session.inputs.iter().map(|i| i.name.clone()).collect()
}

fn session_outputs(session: &Session) -> Vec<String> {
    session.outputs.iter().map(|o| o.name.clone()).collect()
}

/// In-process Moonshine streaming adapter (batch finalize; small or medium).
pub struct MoonshineStt {
    encoder: Session,
    decoder: Session,
    decoder_past: Session,
    /// Maps each decoder output KV tensor to its `decoder_with_past` input.
    kv_map: Vec<(usize, usize)>,
    past_inputs: Vec<String>,
    past_outputs: Vec<String>,
    tokenizer: MoonshineTokenizer,
    language: String,
    active_turn: Option<TurnId>,
    active_frames: Vec<AudioFrame>,
    frames_since_partial: usize,
    last_partial: String,
}

impl MoonshineStt {
    /// Resolve the four Moonshine assets for `model` and load every graph.
    /// Only the selected size's ids are fetched; Whisper and the other
    /// Moonshine size stay untouched on disk.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        model: crate::defaults::SttModel,
    ) -> Result<Self> {
        let ids = match model {
            crate::defaults::SttModel::MoonshineStreamingSmall => [
                ENCODER_ASSET,
                DECODER_ASSET,
                DECODER_PAST_ASSET,
                TOKENIZER_ASSET,
            ],
            crate::defaults::SttModel::MoonshineStreamingMedium => [
                MEDIUM_ENCODER_ASSET,
                MEDIUM_DECODER_ASSET,
                MEDIUM_DECODER_PAST_ASSET,
                MEDIUM_TOKENIZER_ASSET,
            ],
            other => {
                return Err(provider(format!(
                    "MoonshineStt::from_cache requires a Moonshine model, got {}",
                    other.as_str()
                )));
            }
        };
        let mut paths = Vec::with_capacity(4);
        for id in ids {
            let asset = cache
                .manifest()
                .asset(id)
                .ok_or_else(|| Error::ModelCache {
                    message: format!("manifest does not contain the {id} asset"),
                })?;
            paths.push(cache.resolve(asset, fetcher, progress, cancel)?);
        }
        Self::from_paths(&paths)
    }

    fn from_paths(paths: &[PathBuf]) -> Result<Self> {
        let [encoder_path, decoder_path, past_path, tokenizer_path] = paths else {
            return Err(provider("expected the four pinned moonshine assets"));
        };
        let encoder = load_session(encoder_path, "moonshine encoder")?;
        let decoder = load_session(decoder_path, "moonshine decoder")?;
        let decoder_past = load_session(past_path, "moonshine KV decoder")?;
        Self::validate_contract(&encoder, &decoder, &decoder_past)?;
        let kv_map = build_kv_map(&decoder, &decoder_past)?;
        let past_inputs = session_names(&decoder_past);
        let past_outputs = session_outputs(&decoder_past);
        let tokenizer = MoonshineTokenizer::from_file(tokenizer_path)?;
        Ok(Self {
            encoder,
            decoder,
            decoder_past,
            kv_map,
            past_inputs,
            past_outputs,
            tokenizer,
            language: LANGUAGE.to_string(),
            active_turn: None,
            active_frames: Vec::new(),
            frames_since_partial: 0,
            last_partial: String::new(),
        })
    }

    fn validate_contract(encoder: &Session, decoder: &Session, past: &Session) -> Result<()> {
        if session_names(encoder) != ["input_values", "attention_mask"] {
            return Err(provider(format!(
                "unexpected encoder inputs: {}",
                session_names(encoder).join(", ")
            )));
        }
        if session_outputs(encoder) != ["encoder_hidden_states"] {
            return Err(provider("unexpected encoder outputs"));
        }
        if session_names(decoder) != ["decoder_input_ids", "encoder_hidden_states"] {
            return Err(provider(format!(
                "unexpected decoder inputs: {}",
                session_names(decoder).join(", ")
            )));
        }
        if !session_outputs(decoder).starts_with(&["logits".to_string()]) {
            return Err(provider("decoder must emit logits first"));
        }
        if !session_outputs(past).starts_with(&["logits".to_string()]) {
            return Err(provider("KV decoder must emit logits first"));
        }
        Ok(())
    }

    /// Configured STT language. Only `en` is accepted; anything else fails.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The Moonshine export is English-only: accept `en`, reject everything
    /// else (including `auto`) so the `{language}` prompt pin stays truthful.
    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        validate_language(&language)?;
        self.language = language;
        Ok(self)
    }

    fn transcribe_pcm(&mut self, pcm_f32: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut audio = pcm_f32.to_vec();
        let remainder = audio.len() % ENCODER_PAD;
        if remainder != 0 {
            audio.resize(audio.len() + ENCODER_PAD - remainder, 0.0);
        }
        let mask = vec![1_i64; audio.len()];
        let audio_tensor = RawTensor::f32(vec![1, audio.len() as i64], audio);
        let mask_tensor = RawTensor::i64(vec![1, mask.len() as i64], mask);
        let hidden = {
            let mut runner = RawRunner::new(&mut self.encoder)?;
            let out = runner.run(
                &[
                    ("input_values", &audio_tensor),
                    ("attention_mask", &mask_tensor),
                ],
                &["encoder_hidden_states"],
            )?;
            out.into_iter()
                .next()
                .ok_or_else(|| provider("encoder returned no output"))?
        };
        let hidden_shape = hidden.shape.clone();
        let hidden_data = match hidden.data {
            RawData::F32(data) => data,
            RawData::I64(_) => return Err(provider("encoder emitted non-f32 states")),
        };
        let hidden_tensor = RawTensor::f32(hidden_shape, hidden_data);

        // Clone the session contracts up front: the runners below borrow the
        // sessions mutably while the name tables are still needed.
        let kv_output_names: Vec<String> =
            session_outputs(&self.decoder).into_iter().skip(1).collect();
        let past_inputs = self.past_inputs.clone();
        let past_outputs = self.past_outputs.clone();
        let kv_map = self.kv_map.clone();

        let max_tokens = ((pcm_f32.len() as f64 * MAX_TOKENS_PER_SECOND / 16_000.0) as usize + 8)
            .min(MAX_TOKENS_ABSOLUTE);
        let id_tensor = RawTensor::i64(vec![1, 1], vec![BOS]);
        let mut names: Vec<&str> = vec!["logits"];
        names.extend(kv_output_names.iter().map(|s| s.as_str()));
        let mut runner = RawRunner::new(&mut self.decoder)?;
        let mut tensors = runner
            .run(
                &[
                    ("decoder_input_ids", &id_tensor),
                    ("encoder_hidden_states", &hidden_tensor),
                ],
                &names,
            )?
            .into_iter();
        let mut logits = tensors
            .next()
            .ok_or_else(|| provider("decoder returned no logits"))?;
        let mut kv: Vec<RawTensor> = tensors.collect();
        if kv.len() != kv_map.len() {
            return Err(provider(
                "decoder KV count does not match the past-input map",
            ));
        }
        let mut ids: Vec<i64> = Vec::new();
        for _ in 0..max_tokens {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let next = argmax(&logits)?;
            if next == EOS {
                break;
            }
            ids.push(next);
            let id_tensor = RawTensor::i64(vec![1, 1], vec![next]);
            let mut inputs: Vec<(&str, &RawTensor)> = Vec::with_capacity(2 + kv.len());
            inputs.push(("decoder_input_ids", &id_tensor));
            inputs.push(("encoder_hidden_states", &hidden_tensor));
            for (slot, tensor) in kv.iter().enumerate() {
                inputs.push((past_inputs[kv_map[slot].1].as_str(), tensor));
            }
            let mut runner = RawRunner::new(&mut self.decoder_past)?;
            let out_names: Vec<&str> = past_outputs.iter().map(|s| s.as_str()).collect();
            let mut tensors = runner.run(&inputs, &out_names)?.into_iter();
            logits = tensors
                .next()
                .ok_or_else(|| provider("KV decoder returned no logits"))?;
            kv = tensors.collect();
        }
        Ok(self.tokenizer.decode(&ids))
    }
}

/// English-only gate shared by the adapter and `AgentConfig` validation.
fn validate_language(language: &str) -> Result<()> {
    if language != LANGUAGE {
        return Err(Error::Config {
            field: "pipeline.stt.language".into(),
            message: format!("unsupported value {language:?} (allowed: \"en\" with this model)"),
        });
    }
    Ok(())
}

/// Index of the largest logit. Logits arrive as `[1, 1, vocab]`.
fn argmax(logits: &RawTensor) -> Result<i64> {
    match &logits.data {
        RawData::F32(data) => {
            let (mut best_index, mut best_value) = (0_i64, f32::NEG_INFINITY);
            for (index, value) in data.iter().enumerate() {
                if *value > best_value {
                    best_value = *value;
                    best_index = index as i64;
                }
            }
            Ok(best_index)
        }
        RawData::I64(_) => Err(provider("logits are not f32")),
    }
}

impl Stt for MoonshineStt {
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
        let text = self.transcribe_pcm(&audio, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        self.clear_active(utterance.turn);
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: self.language.clone(),
        })
    }

    fn supports_partials(&self) -> bool {
        true
    }

    fn start_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        self.active_turn = Some(turn);
        self.active_frames.clear();
        self.frames_since_partial = 0;
        self.last_partial.clear();
        Ok(())
    }

    fn push_frame(&mut self, frame: &AudioFrame, cancel: &Cancel) -> Result<Option<String>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if self.active_turn.is_none() {
            return Ok(None);
        }
        self.active_frames.push(frame.clone());
        self.frames_since_partial += 1;
        if self.frames_since_partial < PARTIAL_EVERY_FRAMES {
            return Ok(None);
        }
        self.frames_since_partial = 0;
        let utterance = Utterance {
            turn: self.active_turn.expect("active turn checked"),
            frames: self.active_frames.clone(),
        };
        let pcm = utterance.pcm();
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let text = self.transcribe_pcm(&audio, cancel)?;
        // ASR hypotheses can revise. The terminal contract is monotonic, so
        // retain the longest confirmed prefix until VAD asks for the final
        // transcript, which is allowed to replace this provisional text.
        if !text.is_empty() && text.starts_with(&self.last_partial) && text != self.last_partial {
            self.last_partial = text.clone();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    fn cancel_turn(&mut self, turn: TurnId) {
        self.clear_active(turn);
    }
}

impl MoonshineStt {
    fn clear_active(&mut self, turn: TurnId) {
        if self.active_turn == Some(turn) {
            self.active_turn = None;
            self.active_frames.clear();
            self.frames_since_partial = 0;
            self.last_partial.clear();
        }
    }
}

/// Map each decoder KV output to its `decoder_with_past` input: `present_X`
/// feeds `past_X`, and cross-attention tensors feed their `*_orig` inputs.
fn build_kv_map(decoder: &Session, past: &Session) -> Result<Vec<(usize, usize)>> {
    let decoder_outputs = session_outputs(decoder);
    let past_inputs = session_names(past);
    let mut map = Vec::new();
    for (out_index, name) in decoder_outputs.iter().enumerate().skip(1) {
        let rest = name.strip_prefix("present_").ok_or_else(|| {
            provider(format!(
                "unexpected decoder output without present_ prefix: {name}"
            ))
        })?;
        let candidate = format!("past_{rest}");
        if let Some(in_index) = past_inputs.iter().position(|n| *n == candidate) {
            map.push((out_index, in_index));
            continue;
        }
        let candidate = format!("{name}_orig");
        if let Some(in_index) = past_inputs.iter().position(|n| *n == candidate) {
            map.push((out_index, in_index));
            continue;
        }
        return Err(provider(format!(
            "decoder output {name} has no past input (tried {candidate})"
        )));
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_TOKENIZER: &str = r#"{
        "model": {"type": "BPE", "vocab": {
            "<unk>": 0, "<s>": 1, "</s>": 2,
            "▁": 10, "▁hello": 11, "▁world": 12, "ing": 13,
            "<0x20>": 14, "<0x41>": 15, "!": 16
        }, "merges": []},
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        ]}
    }"#;

    fn synthetic() -> MoonshineTokenizer {
        MoonshineTokenizer::from_json(SYNTHETIC_TOKENIZER).expect("synthetic tokenizer parses")
    }

    #[test]
    fn decode_replaces_markers_and_strips_leading_space() {
        // "▁hello ▁world" + "ing" fuses to "hello worlding".
        assert_eq!(synthetic().decode(&[11, 12, 13]), "hello worlding");
    }

    #[test]
    fn decode_stops_at_eos_and_skips_specials() {
        assert_eq!(synthetic().decode(&[1, 11, 0, 12, 2, 11]), "hello world");
    }

    #[test]
    fn decode_resolves_byte_fallback_tokens() {
        // <0x20> is a space, <0x41> is "A".
        assert_eq!(synthetic().decode(&[11, 14, 15, 16]), "hello A!");
    }

    #[test]
    fn decode_skips_unknown_ids() {
        assert_eq!(synthetic().decode(&[11, 999_999, 12]), "hello world");
    }

    #[test]
    fn decode_empty_is_empty() {
        assert_eq!(synthetic().decode(&[]), "");
        assert_eq!(synthetic().decode(&[2]), "");
    }

    #[test]
    fn english_only_gate_rejects_everything_but_en() {
        validate_language("en").expect("en is accepted");
        for bad in ["fr", "auto", "", "EN"] {
            let err = validate_language(bad).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.stt.language"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn malformed_tokenizer_json_is_a_provider_error() {
        let err = MoonshineTokenizer::from_json("{nope").unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
        let err = MoonshineTokenizer::from_json(r#"{"model": {}}"#).unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[test]
    fn missing_weights_fail_before_any_decode() {
        let err = match MoonshineStt::from_paths(&[
            PathBuf::from("/no/such/encoder.onnx"),
            PathBuf::from("/no/such/decoder.onnx"),
            PathBuf::from("/no/such/past.onnx"),
            PathBuf::from("/no/such/tokenizer.json"),
        ]) {
            Err(err) => err,
            Ok(_) => panic!("missing weights should fail"),
        };
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[test]
    fn wrong_path_count_is_a_provider_error() {
        let err = match MoonshineStt::from_paths(&[PathBuf::from("only-one")]) {
            Err(err) => err,
            Ok(_) => panic!("one path should fail"),
        };
        assert!(err.to_string().contains("four pinned moonshine assets"));
    }

    /// Full weights + fixture check. Ignored by default (needs ~360 MB of
    /// ONNX + tokenizer): point `SYLLABIX_MOONSHINE_DIR` at a directory with
    /// `encoder_model_int8.onnx`, `decoder_model_int8.onnx`,
    /// `decoder_with_past_model_int8.onnx`, `tokenizer.json`, then:
    /// `cargo test -p syllabix-core --lib moonshine::tests::native -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn native_recorded_fixtures_meet_word_match_gate() {
        let dir = std::env::var("SYLLABIX_MOONSHINE_DIR")
            .expect("SYLLABIX_MOONSHINE_DIR must point at the moonshine ONNX directory");
        let root = Path::new(&dir);
        let mut stt = MoonshineStt::from_paths(&[
            root.join("encoder_model_int8.onnx"),
            root.join("decoder_model_int8.onnx"),
            root.join("decoder_with_past_model_int8.onnx"),
            root.join("tokenizer.json"),
        ])
        .expect("moonshine weights load");
        assert_eq!(stt.name(), PROVIDER_NAME);
        assert_eq!(stt.language(), LANGUAGE);
        let fixtures: &[(&str, &[&str], f64)] = &[
            (
                "jfk.wav",
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
                ],
                1.0,
            ),
            (
                "librispeech-1089-134686-0000.wav",
                &[
                    "he", "hoped", "there", "would", "be", "stew", "for", "dinner", "turnips",
                    "and", "carrots", "and", "bruised", "potatoes", "and", "fat", "mutton",
                    "pieces", "to", "be", "ladled", "out", "in", "thick", "peppered", "flour",
                    "fattened", "sauce",
                ],
                0.8,
            ),
            (
                "librispeech-121-127105-0009.wav",
                &["she", "has", "been", "dead", "these", "twenty", "years"],
                0.8,
            ),
            (
                "librispeech-1995-1837-0005.wav",
                &[
                    "she", "was", "so", "strange", "and", "human", "a", "creature",
                ],
                0.8,
            ),
        ];
        for (index, (file, expected, minimum)) in fixtures.iter().enumerate() {
            let wav = std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/stt")
                    .join(file),
            )
            .unwrap_or_else(|err| panic!("{file} fixture: {err}"));
            let pcm = crate::audio::read_wav(std::io::Cursor::new(wav))
                .unwrap_or_else(|err| panic!("{file} parses: {err}"));
            let frames = crate::audio::record_fixture_to_frames(&pcm)
                .unwrap_or_else(|err| panic!("{file} frames: {err}"));
            let transcript = stt
                .transcribe(
                    &Utterance {
                        turn: crate::types::TurnId(index as u64),
                        frames,
                    },
                    &Cancel::new(),
                )
                .unwrap_or_else(|err| panic!("{file} transcribes: {err}"));
            let ratio = crate::stt::word_match_ratio(&transcript.text, expected);
            eprintln!(
                "Moonshine {file}: {:.1}% word match ({:?})",
                ratio * 100.0,
                transcript.text
            );
            assert!(
                ratio >= *minimum,
                "{file} transcript {:?} matched {:.1}% of {:?} (need {:.0}%)",
                transcript.text,
                ratio * 100.0,
                expected,
                minimum * 100.0
            );
            assert_eq!(transcript.language, LANGUAGE);
        }
    }

    #[test]
    fn provider_name_is_moonshine() {
        assert_eq!(PROVIDER_NAME, "moonshine");
        assert_eq!(LANGUAGE, "en");
        assert_eq!(crate::defaults::SttModel::Small.as_str(), "whisper-small");
    }
}
