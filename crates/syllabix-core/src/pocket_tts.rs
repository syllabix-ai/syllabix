//! P1 native feasibility contract for Kyutai Pocket TTS.
//!
//! This module is intentionally not a [`crate::providers::Tts`] implementation:
//! P2 owns configuration, playback streaming, cancellation, and barge-in. P1
//! proves that the pinned multi-graph ONNX package loads in-process through the
//! already-shipped ONNX Runtime, and that its text-conditioning graph produces
//! deterministic native output. The fixed `alba` state is only inspected here;
//! no user audio is accepted and no voice state is registered.

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    path::{Path, PathBuf},
    ptr,
};

use serde::Deserialize;

use ort::{session::Session, AsPointer};

use crate::models::{Fetcher, ModelCache, Progress};
use crate::{Cancel, Error, Result};

pub const POCKET_TTS_BUNDLE_ASSET: &str = "pocket-tts-bundle";
pub const POCKET_TTS_BOS_ASSET: &str = "pocket-tts-bos";
pub const POCKET_TTS_TOKENIZER_ASSET: &str = "pocket-tts-tokenizer";
pub const POCKET_TTS_TEXT_CONDITIONER_ASSET: &str = "pocket-tts-text-conditioner";
pub const POCKET_TTS_FLOW_MAIN_ASSET: &str = "pocket-tts-flow-main";
pub const POCKET_TTS_FLOW_ASSET: &str = "pocket-tts-flow";
pub const POCKET_TTS_MIMI_DECODER_ASSET: &str = "pocket-tts-mimi-decoder";
pub const POCKET_TTS_VOICE_ASSET: &str = "pocket-tts-voice-alba";

const ASSETS: [&str; 8] = [
    POCKET_TTS_BUNDLE_ASSET,
    POCKET_TTS_BOS_ASSET,
    POCKET_TTS_TOKENIZER_ASSET,
    POCKET_TTS_TEXT_CONDITIONER_ASSET,
    POCKET_TTS_FLOW_MAIN_ASSET,
    POCKET_TTS_FLOW_ASSET,
    POCKET_TTS_MIMI_DECODER_ASSET,
    POCKET_TTS_VOICE_ASSET,
];

/// P1 loader for the pinned English ONNX graph set.
pub struct PocketTts {
    text_conditioner: Session,
    flow_main: Session,
    flow: Session,
    decoder: Session,
    flow_state: Vec<StateSpec>,
    mimi_state: Vec<StateSpec>,
    voice: PathBuf,
}

impl PocketTts {
    /// Resolve exactly the P1 assets, then load every inference graph with the
    /// existing ONNX Runtime. Nothing here changes the zero-config stack.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let paths = ASSETS
            .iter()
            .map(|id| resolve(cache, id, fetcher, progress, cancel))
            .collect::<Result<Vec<_>>>()?;
        Self::from_paths(&paths)
    }

    fn from_paths(paths: &[PathBuf]) -> Result<Self> {
        let [bundle, bos, tokenizer, text, flow_main, flow, decoder, voice] = paths else {
            return Err(provider("expected the eight pinned P1 assets"));
        };
        let (flow_state, mimi_state) = inspect_package(bundle, bos, tokenizer, voice)?;
        let text_conditioner = load_graph(text, "text conditioner")?;
        let flow_main = load_graph(flow_main, "flow LM main")?;
        let flow = load_graph(flow, "flow LM flow")?;
        let decoder = load_graph(decoder, "Mimi decoder")?;
        validate_text_contract(&text_conditioner)?;
        Ok(Self {
            text_conditioner,
            flow_main,
            flow,
            decoder,
            flow_state,
            mimi_state,
            voice: voice.clone(),
        })
    }

    /// Deterministic native fixture for all three target families. The token
    /// ids come from the upstream export's own fixture, avoiding a tokenizer
    /// implementation until P2 while still exercising the real graph.
    pub fn text_fixture(&mut self) -> Result<Vec<f32>> {
        let tokens = vec![10_i64, 20, 30, 40, 50];
        let outputs = self
            .text_conditioner
            .run(
                ort::inputs!["token_ids" => ([1_usize, tokens.len()], tokens)]
                    .map_err(ort_error)?,
            )
            .map_err(ort_error)?;
        let embeddings = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if embeddings.len() != 5 * 1024 || embeddings.iter().any(|v| !v.is_finite()) {
            return Err(provider("text fixture produced invalid embeddings"));
        }
        Ok(embeddings)
    }

    /// Produce a deterministic decoder frame through ONNX Runtime's C API.
    /// Its recurrent state is fed directly as OrtValues; this is the required
    /// escape hatch for Pocket TTS's dynamic state tensors.
    pub fn c_api_fixture(&mut self) -> Result<Vec<f32>> {
        let mut runner = RawRunner::new(&mut self.decoder)?;
        let latent = RawTensor::f32(vec![1, 1, 32], vec![0.0; 32]);
        let states = self
            .mimi_state
            .iter()
            .map(StateSpec::initial)
            .collect::<Vec<_>>();
        let mut inputs = vec![("latent", &latent)];
        inputs.extend(
            self.mimi_state
                .iter()
                .zip(&states)
                .map(|(spec, state)| (spec.input_name.as_str(), state)),
        );
        let outputs = runner.run(&inputs, &["audio_frame"])?;
        Ok(outputs[0].f32.clone())
    }

    /// A deterministic, text-conditioned native fixture. It intentionally
    /// generates one latent frame: P1 needs a portable feasibility gate, while
    /// P2 owns user-facing streaming and duration policy.
    pub fn synthesize_fixture(&mut self) -> Result<Vec<f32>> {
        let embeddings = RawTensor::f32(vec![1, 5, 1024], self.text_fixture()?);
        let empty_sequence = RawTensor::f32(vec![1, 0, 32], vec![]);
        let mut state = voice_state(&self.flow_state, &self.voice)?;
        let mut main = RawRunner::new(&mut self.flow_main)?;
        let mut inputs = vec![
            ("sequence", &empty_sequence),
            ("text_embeddings", &embeddings),
        ];
        inputs.extend(
            self.flow_state
                .iter()
                .zip(&state)
                .map(|(spec, tensor)| (spec.input_name.as_str(), tensor)),
        );
        let mut output_names = vec!["conditioning", "eos_logit"];
        output_names.extend(self.flow_state.iter().map(|spec| spec.output_name.as_str()));
        let outputs = main.run(&inputs, &output_names)?;
        state = outputs.into_iter().skip(2).collect();

        let current = RawTensor::f32(vec![1, 1, 32], vec![f32::NAN; 32]);
        let empty_text = RawTensor::f32(vec![1, 0, 1024], vec![]);
        let mut inputs = vec![("sequence", &current), ("text_embeddings", &empty_text)];
        inputs.extend(
            self.flow_state
                .iter()
                .zip(&state)
                .map(|(spec, tensor)| (spec.input_name.as_str(), tensor)),
        );
        let outputs = main.run(&inputs, &output_names)?;
        let conditioning = outputs[0].f32_data()?.to_vec();
        let conditioning = RawTensor::f32(outputs[0].shape.clone(), conditioning);
        let x = RawTensor::f32(vec![1, 32], vec![0.0; 32]);
        let s = RawTensor::f32(vec![1, 1], vec![0.0]);
        let t = RawTensor::f32(vec![1, 1], vec![1.0]);
        let mut flow = RawRunner::new(&mut self.flow)?;
        let latent = flow.run(
            &[("c", &conditioning), ("s", &s), ("t", &t), ("x", &x)],
            &["flow_dir"],
        )?;
        let latent = RawTensor::f32(vec![1, 1, 32], latent[0].f32_data()?.to_vec());
        let decoder_state = self
            .mimi_state
            .iter()
            .map(StateSpec::initial)
            .collect::<Vec<_>>();
        let mut decoder = RawRunner::new(&mut self.decoder)?;
        let mut inputs = vec![("latent", &latent)];
        inputs.extend(
            self.mimi_state
                .iter()
                .zip(&decoder_state)
                .map(|(spec, tensor)| (spec.input_name.as_str(), tensor)),
        );
        let audio = decoder.run(&inputs, &["audio_frame"])?;
        Ok(audio[0].f32_data()?.to_vec())
    }
}

#[derive(Clone, Debug)]
enum RawData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<u8>),
}
#[derive(Clone, Debug)]
struct RawTensor {
    shape: Vec<i64>,
    data: RawData,
    f32: Vec<f32>,
}
impl RawTensor {
    fn f32(shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            shape,
            f32: data.clone(),
            data: RawData::F32(data),
        }
    }
    fn i64(shape: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            shape,
            f32: vec![],
            data: RawData::I64(data),
        }
    }
    fn bool(shape: Vec<i64>, data: Vec<u8>) -> Self {
        Self {
            shape,
            f32: vec![],
            data: RawData::Bool(data),
        }
    }
    fn f32_data(&self) -> Result<&[f32]> {
        match &self.data {
            RawData::F32(data) => Ok(data),
            _ => Err(provider("expected a float32 ONNX output")),
        }
    }
}

#[derive(Clone, Deserialize)]
struct StateSpec {
    input_name: String,
    output_name: String,
    module: String,
    key: String,
    dtype: String,
    fill: String,
    shape: Vec<i64>,
}
impl StateSpec {
    fn initial(&self) -> RawTensor {
        let len = self.shape.iter().product::<i64>().max(0) as usize;
        match self.dtype.as_str() {
            "int64" => RawTensor::i64(self.shape.clone(), vec![0; len]),
            "bool" => RawTensor::bool(self.shape.clone(), vec![u8::from(self.fill == "ones"); len]),
            _ => RawTensor::f32(
                self.shape.clone(),
                vec![if self.fill == "nan" { f32::NAN } else { 0.0 }; len],
            ),
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
            RawData::Bool(data) => (
                ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                data.len(),
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
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                let mut out = vec![0_i64; len];
                if len != 0 {
                    unsafe {
                        ptr::copy_nonoverlapping(data.cast::<i64>(), out.as_mut_ptr(), len);
                    }
                }
                Ok(RawTensor::i64(shape, out))
            }
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
                let mut out = vec![0_u8; len];
                if len != 0 {
                    unsafe {
                        ptr::copy_nonoverlapping(data.cast::<u8>(), out.as_mut_ptr(), len);
                    }
                }
                Ok(RawTensor::bool(shape, out))
            }
            _ => Err(provider("unsupported ONNX output type")),
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
    Err(provider(&format!("ONNX Runtime C API: {message}")))
}

fn resolve(
    cache: &ModelCache,
    id: &str,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<PathBuf> {
    let asset = cache
        .manifest()
        .asset(id)
        .ok_or_else(|| provider(&format!("manifest does not contain {id}")))?;
    cache.resolve(asset, fetcher, progress, cancel)
}

fn load_graph(path: &Path, label: &str) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(|err| provider(&format!("could not load {label} {}: {err}", path.display())))
}

fn validate_text_contract(session: &Session) -> Result<()> {
    let inputs = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    let outputs = session
        .outputs
        .iter()
        .map(|output| output.name.as_str())
        .collect::<Vec<_>>();
    if inputs != ["token_ids"] || outputs != ["embeddings"] {
        return Err(provider(&format!(
            "unexpected text conditioner contract: inputs={} outputs={}",
            inputs.join(", "),
            outputs.join(", ")
        )));
    }
    Ok(())
}

fn inspect_package(
    bundle: &Path,
    bos: &Path,
    tokenizer: &Path,
    voice: &Path,
) -> Result<(Vec<StateSpec>, Vec<StateSpec>)> {
    let bundle_bytes = std::fs::read(bundle)?;
    let value: serde_json::Value = serde_json::from_slice(&bundle_bytes)
        .map_err(|err| provider(&format!("invalid bundle.json: {err}")))?;
    if !value.is_object() {
        return Err(provider("bundle.json must be an object"));
    }
    if !std::fs::read(bos)?.starts_with(b"\x93NUMPY") {
        return Err(provider("bos_before_voice.npy is not a NumPy asset"));
    }
    if std::fs::metadata(tokenizer)?.len() == 0 {
        return Err(provider("tokenizer.model is empty"));
    }
    let voice_bytes = std::fs::read(voice)?;
    let Some(header_len) = voice_bytes
        .get(..8)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
    else {
        return Err(provider("fixed voice is not a safetensors file"));
    };
    let header_end = 8_usize.saturating_add(header_len as usize);
    if header_len == 0
        || header_end >= voice_bytes.len()
        || serde_json::from_slice::<serde_json::Value>(&voice_bytes[8..header_end]).is_err()
    {
        return Err(provider("fixed voice is not a safetensors file"));
    }
    let parse = |key| {
        serde_json::from_value(
            value
                .get(key)
                .cloned()
                .ok_or_else(|| provider(&format!("bundle lacks {key}")))?,
        )
        .map_err(|err| provider(&format!("invalid {key}: {err}")))
    };
    Ok((
        parse("flow_lm_state_manifest")?,
        parse("mimi_state_manifest")?,
    ))
}

#[derive(Deserialize)]
struct SafeTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn voice_state(specs: &[StateSpec], path: &Path) -> Result<Vec<RawTensor>> {
    let bytes = std::fs::read(path)?;
    let header_len = bytes
        .get(..8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| provider("invalid fixed voice"))? as usize;
    let start = 8 + header_len;
    let header: std::collections::BTreeMap<String, SafeTensorHeader> =
        serde_json::from_slice(&bytes[8..start])
            .map_err(|error| provider(&format!("invalid fixed voice header: {error}")))?;
    specs
        .iter()
        .map(|spec| {
            let mut target = spec.initial();
            let name = format!("{}/{}", spec.module, spec.key);
            let source = header.get(&name).or_else(|| {
                (spec.key == "step")
                    .then(|| header.get(&format!("{}/offset", spec.module)))
                    .flatten()
            });
            let Some(source) = source else {
                return Ok(target);
            };
            let range = start + source.data_offsets[0]..start + source.data_offsets[1];
            let data = bytes
                .get(range)
                .ok_or_else(|| provider("fixed voice tensor exceeds file"))?;
            match (&mut target.data, source.dtype.as_str()) {
                (RawData::F32(out), "F32") => {
                    let values = data
                        .chunks_exact(4)
                        .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes")))
                        .collect::<Vec<_>>();
                    copy_voice_f32(out, &spec.shape, &values, &source.shape);
                    target.f32 = out.clone();
                }
                (RawData::I64(out), "I64") => {
                    if let Some(value) = data.get(..8) {
                        out[0] = i64::from_le_bytes(value.try_into().expect("eight bytes"));
                    }
                }
                _ => {}
            }
            Ok(target)
        })
        .collect()
}

fn copy_voice_f32(
    target: &mut [f32],
    target_shape: &[i64],
    source: &[f32],
    source_shape: &[usize],
) {
    if target_shape.len() == 5 && source_shape.len() == 5 {
        let inner = source_shape[3] * source_shape[4];
        let target_stride = target_shape[2] as usize * inner;
        let source_stride = source_shape[2] * inner;
        for batch in 0..target_shape[0] as usize {
            let count = source_stride.min(target_stride);
            target[batch * target_stride..batch * target_stride + count]
                .copy_from_slice(&source[batch * source_stride..batch * source_stride + count]);
        }
    } else {
        let len = target.len().min(source.len());
        target[..len].copy_from_slice(&source[..len]);
    }
}

fn provider(message: &str) -> Error {
    Error::Provider {
        provider: "pocket-tts",
        message: message.into(),
    }
}

fn ort_error(error: ort::Error) -> Error {
    provider(&format!("ONNX Runtime: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p1_asset_set_is_complete_and_internal() {
        assert_eq!(ASSETS.len(), 8);
        assert!(ASSETS.iter().all(|id| id.starts_with("pocket-tts-")));
    }
}
