//! Audio8 0.1B ONNX native feasibility runtime (A1).
//!
//! This deliberately is not wired into `TtsModel` or the live pipeline. It
//! proves the published CPU ONNX contract in-process: tokenizer + fixed
//! upstream prompt codes → bounded SlowAR/FastAR recurrence → codec PCM.
//! Voice registration and its encoder are intentionally not included.

use std::fs;
use std::path::Path;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};

pub const AUDIO8_SAMPLE_RATE_HZ: u32 = 44_100;
pub const AUDIO8_MAX_FRAMES: usize = 256;
const AUDIO8_CODEBOOKS: usize = 10;
const AUDIO8_CODEBOOK_SIZE: usize = 4_096;
const AUDIO8_SEMANTIC_BEGIN: i64 = 65_537;
const AUDIO8_EOS_LOGIT: usize = AUDIO8_CODEBOOK_SIZE;
const AUDIO8_REFERENCE_TEXT: &str = "至今为止，元气火箭总共发行了两张专辑。";

const ASSETS: [&str; 9] = [
    "audio8-slow-ar",
    "audio8-slow-ar-data",
    "audio8-fast-ar",
    "audio8-fast-ar-data",
    "audio8-codec-decoder",
    "audio8-codec-decoder-data",
    "audio8-tokenizer",
    "audio8-runtime-manifest",
    "audio8-reference-codes",
];

#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    sample_rate: u32,
    max_seq_len: usize,
    num_layers: usize,
    n_local_heads: usize,
    head_dim: usize,
    num_fast_layers: usize,
    fast_n_local_heads: usize,
    fast_head_dim: usize,
    num_codebooks: usize,
    codebook_size: usize,
    semantic_begin_id: i64,
    im_end_id: i64,
}

/// Isolated CPU-only Audio8 ONNX engine used by A1's native test.
pub struct Audio8Native {
    slow: Session,
    fast: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    reference_codes: Vec<i64>,
    manifest: RuntimeManifest,
}

impl Audio8Native {
    /// Fetch and verify exactly the runtime assets required for synthesis.
    /// The model-cache paths retain upstream graph/data basenames because ONNX
    /// external-data references are relative to their graph files.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let mut path = |id: &str| -> Result<std::path::PathBuf> {
            let asset = cache
                .manifest()
                .asset(id)
                .ok_or_else(|| Error::ModelCache {
                    message: format!("manifest does not contain {id}"),
                })?;
            cache.resolve(asset, fetcher, progress, cancel)
        };
        let mut resolved = Vec::with_capacity(ASSETS.len());
        for id in ASSETS {
            resolved.push((id, path(id)?));
        }
        let get = |id: &str| -> &Path {
            resolved
                .iter()
                .find(|(candidate, _)| *candidate == id)
                .expect("Audio8 asset resolved")
                .1
                .as_path()
        };
        Self::from_paths(
            get("audio8-slow-ar"),
            get("audio8-fast-ar"),
            get("audio8-codec-decoder"),
            get("audio8-tokenizer"),
            get("audio8-runtime-manifest"),
            get("audio8-reference-codes"),
        )
    }

    /// Load a fully verified local Audio8 model directory. The decoder is
    /// intentionally CPU-only: it reuses the application's existing ORT C
    /// API and never configures a second execution provider.
    pub fn from_paths(
        slow: &Path,
        fast: &Path,
        decoder: &Path,
        tokenizer: &Path,
        runtime_manifest: &Path,
        reference_codes: &Path,
    ) -> Result<Self> {
        let manifest: RuntimeManifest = serde_json::from_slice(&fs::read(runtime_manifest)?)
            .map_err(|error| provider(format!("invalid runtime manifest: {error}")))?;
        validate_manifest(&manifest)?;
        let slow_session = session(slow)?;
        let fast_session = session(fast)?;
        let decoder_session = session(decoder)?;
        validate_session_inputs(
            &slow_session,
            "slow",
            &[
                "codes",
                "position",
                "cache_keys",
                "cache_values",
                "conv_states",
                "ssm_states",
            ],
        )?;
        validate_session_inputs(
            &fast_session,
            "fast",
            &[
                "slow_hidden",
                "token_id",
                "use_slow_hidden",
                "input_pos",
                "cache_key_0",
                "cache_value_0",
                "cache_key_1",
                "cache_value_1",
                "cache_key_2",
                "cache_value_2",
                "cache_key_3",
                "cache_value_3",
            ],
        )?;
        validate_session_inputs(&decoder_session, "decoder", &["codes"])?;
        let tokenizer = Tokenizer::from_file(tokenizer)
            .map_err(|error| provider(format!("invalid tokenizer: {error}")))?;
        let reference_codes = read_i64_npy(reference_codes, AUDIO8_CODEBOOKS)?;
        Ok(Self {
            slow: slow_session,
            fast: fast_session,
            decoder: decoder_session,
            tokenizer,
            reference_codes,
            manifest,
        })
    }

    /// Greedy, bounded synthesis for the deterministic A1 fixture. A2 will
    /// add streaming into the live playback queue; A1 returns native 44.1 kHz
    /// PCM directly so the graph contract can be verified independently.
    pub fn synthesize_greedy(
        &self,
        text: &str,
        max_frames: usize,
        cancel: &Cancel,
    ) -> Result<Vec<f32>> {
        let max_frames = max_frames.clamp(1, AUDIO8_MAX_FRAMES);
        let prompt = self.build_prompt(text)?;
        let prompt_len = prompt.len() / (AUDIO8_CODEBOOKS + 1);
        if prompt_len >= self.manifest.max_seq_len {
            return Err(provider(format!(
                "Audio8 prompt has {prompt_len} positions; max is {}",
                self.manifest.max_seq_len
            )));
        }
        let mut slow_state = SlowState::new(&self.manifest);
        let (mut logits, mut hidden) = self.prefill(&prompt, prompt_len, &mut slow_state)?;
        let mut frames = Vec::<i64>::new();
        for frame_index in 0..max_frames.min(self.manifest.max_seq_len - prompt_len) {
            cancelled(cancel)?;
            let semantic_index = argmax(&logits);
            if semantic_index == AUDIO8_EOS_LOGIT {
                break;
            }
            if semantic_index >= self.manifest.codebook_size {
                return Err(provider(format!(
                    "slow AR produced invalid semantic index {semantic_index}"
                )));
            }
            let semantic = self.manifest.semantic_begin_id + semantic_index as i64;
            let frame = self.fast_frame(&hidden, semantic, cancel)?;
            frames.extend_from_slice(&frame);
            cancelled(cancel)?;
            let codes = std::iter::once(semantic).chain(frame).collect::<Vec<_>>();
            let position = [i64::try_from(prompt_len + frame_index).expect("Audio8 position")];
            (logits, hidden) = self.slow_step(&codes, &position, &mut slow_state)?;
        }
        if frames.is_empty() {
            return Err(provider("Audio8 stopped before producing codec frames"));
        }
        self.decode_pcm(&frames, cancel)
    }

    fn build_prompt(&self, text: &str) -> Result<Vec<i64>> {
        let prefix = format!(
            "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n<|speaker:0|>{AUDIO8_REFERENCE_TEXT}\n\nSpeech:\n"
        );
        let suffix = format!(
            "<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n<|voice|>",
            clean_text(text)
        );
        let mut row0 = self.encode(&prefix)?;
        row0.extend(
            self.reference_codes
                .iter()
                .step_by(AUDIO8_CODEBOOKS)
                .map(|id| id + AUDIO8_SEMANTIC_BEGIN),
        );
        row0.extend(self.encode(&suffix)?);
        let reference_start = self.encode(&prefix)?.len();
        let mut all = vec![0_i64; (AUDIO8_CODEBOOKS + 1) * row0.len()];
        all[..row0.len()].copy_from_slice(&row0);
        let reference_frames = self.reference_codes.len() / AUDIO8_CODEBOOKS;
        for codebook in 0..AUDIO8_CODEBOOKS {
            let dest = (codebook + 1) * row0.len() + reference_start;
            let source = codebook * reference_frames;
            all[dest..dest + reference_frames]
                .copy_from_slice(&self.reference_codes[source..source + reference_frames]);
        }
        Ok(all)
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().iter().map(|id| i64::from(*id)).collect())
            .map_err(|error| provider(format!("Audio8 tokenization failed: {error}")))
    }

    fn slow_step(
        &self,
        codes: &[i64],
        positions: &[i64],
        state: &mut SlowState,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if codes.len() != AUDIO8_CODEBOOKS + 1 || positions.len() != 1 {
            return Err(provider(
                "Audio8 SlowAR requires exactly one packed position per step",
            ));
        }
        let mut inputs = vec![
            named_tensor("codes", vec![1, AUDIO8_CODEBOOKS + 1, 1], codes.to_vec())?,
            named_tensor("position", vec![1], positions.to_vec())?,
        ];
        inputs.extend(state.inputs());
        let outputs = self.slow.run(inputs).map_err(ort_error)?;
        if outputs.len() != 6 {
            return Err(provider(format!(
                "slow AR returned {} outputs; expected {}",
                outputs.len(),
                6
            )));
        }
        let logits = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect();
        let hidden = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect();
        state.update(&outputs, positions[0])?;
        Ok((logits, hidden))
    }

    fn prefill(
        &self,
        prompt: &[i64],
        prompt_len: usize,
        state: &mut SlowState,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut last = None;
        for position in 0..prompt_len {
            let column = (0..=AUDIO8_CODEBOOKS)
                .map(|row| prompt[row * prompt_len + position])
                .collect::<Vec<_>>();
            last = Some(self.slow_step(&column, &[position as i64], state)?);
        }
        last.ok_or_else(|| provider("Audio8 prompt was empty"))
    }

    fn fast_frame(&self, hidden: &[f32], semantic: i64, cancel: &Cancel) -> Result<Vec<i64>> {
        let mut cache = CacheSet::fast(&self.manifest);
        let mut token = 0_i64;
        let mut frame = Vec::with_capacity(AUDIO8_CODEBOOKS);
        for position in 0..AUDIO8_CODEBOOKS {
            cancelled(cancel)?;
            let use_hidden = position == 0;
            let mut inputs = vec![
                named_tensor("slow_hidden", vec![1, 1, hidden.len()], hidden.to_vec())?,
                named_tensor("token_id", vec![1, 1], vec![token])?,
                named_tensor("use_slow_hidden", vec![1], vec![use_hidden])?,
                named_tensor("input_pos", vec![1], vec![position as i64])?,
            ];
            inputs.extend(cache.inputs());
            let outputs = self.fast.run(inputs).map_err(ort_error)?;
            let logits = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
            cache.update(&outputs, 1, &[position as i64])?;
            token = if position == 0 {
                (semantic - self.manifest.semantic_begin_id)
                    .clamp(0, (AUDIO8_CODEBOOK_SIZE - 1) as i64)
            } else {
                i64::try_from(argmax(&logits.iter().copied().collect::<Vec<_>>()))
                    .expect("codec id")
            };
            if token < 0 || token as usize >= self.manifest.codebook_size {
                return Err(provider(format!(
                    "fast AR produced invalid codec id {token}"
                )));
            }
            frame.push(token);
        }
        Ok(frame)
    }

    fn decode_pcm(&self, codes: &[i64], cancel: &Cancel) -> Result<Vec<f32>> {
        cancelled(cancel)?;
        let frames = codes.len() / AUDIO8_CODEBOOKS;
        let output = self
            .decoder
            .run(vec![named_tensor(
                "codes",
                vec![1, AUDIO8_CODEBOOKS, frames],
                codes.to_vec(),
            )?])
            .map_err(ort_error)?;
        let pcm: Vec<f32> = output[0]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect();
        if pcm.is_empty() || pcm.iter().any(|sample| !sample.is_finite()) {
            return Err(provider("Audio8 codec returned empty or non-finite PCM"));
        }
        Ok(pcm)
    }
}

/// The published 0.1B SlowAR export has four packed recurrent state tensors:
/// 24-layer attention keys/values plus Falcon-H1 convolution and SSM state.
/// Keep them bounded at the manifest dimensions; there is no growing cache.
struct SlowState {
    cache_keys: Vec<f32>,
    cache_values: Vec<f32>,
    conv_states: Vec<f32>,
    ssm_states: Vec<f32>,
    cache_shape: Vec<usize>,
    conv_shape: Vec<usize>,
    ssm_shape: Vec<usize>,
}

impl SlowState {
    fn new(manifest: &RuntimeManifest) -> Self {
        let cache_shape = vec![
            manifest.num_layers,
            1,
            manifest.n_local_heads,
            manifest.max_seq_len,
            manifest.head_dim,
        ];
        let conv_shape = vec![manifest.num_layers, 1, 896, 4];
        let ssm_shape = vec![manifest.num_layers, 1, 24, 32, 64];
        Self {
            cache_keys: vec![0.0; cache_shape.iter().product()],
            cache_values: vec![0.0; cache_shape.iter().product()],
            conv_states: vec![0.0; conv_shape.iter().product()],
            ssm_states: vec![0.0; ssm_shape.iter().product()],
            cache_shape,
            conv_shape,
            ssm_shape,
        }
    }

    fn inputs(&self) -> Vec<(String, SessionInputValue<'static>)> {
        vec![
            named_tensor(
                "cache_keys",
                self.cache_shape.clone(),
                self.cache_keys.clone(),
            ),
            named_tensor(
                "cache_values",
                self.cache_shape.clone(),
                self.cache_values.clone(),
            ),
            named_tensor(
                "conv_states",
                self.conv_shape.clone(),
                self.conv_states.clone(),
            ),
            named_tensor(
                "ssm_states",
                self.ssm_shape.clone(),
                self.ssm_states.clone(),
            ),
        ]
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .expect("validated Audio8 recurrent state")
    }

    fn update(
        &mut self,
        outputs: &ort::session::SessionOutputs<'_, '_>,
        position: i64,
    ) -> Result<()> {
        copy_slow_cache_delta(
            &mut self.cache_keys,
            &self.cache_shape,
            &outputs[2],
            position,
            "cache_keys",
        )?;
        copy_slow_cache_delta(
            &mut self.cache_values,
            &self.cache_shape,
            &outputs[3],
            position,
            "cache_values",
        )?;
        copy_state(
            &mut self.conv_states,
            &self.conv_shape,
            &outputs[4],
            "conv_states",
        )?;
        copy_state(
            &mut self.ssm_states,
            &self.ssm_shape,
            &outputs[5],
            "ssm_states",
        )
    }
}

fn copy_slow_cache_delta(
    destination: &mut [f32],
    full_shape: &[usize],
    value: &ort::value::DynValue,
    position: i64,
    label: &str,
) -> Result<()> {
    let source = value.try_extract_tensor::<f32>().map_err(ort_error)?;
    let expected = [full_shape[0], 1, full_shape[2], full_shape[4]];
    if source.shape() != expected {
        return Err(provider(format!(
            "unexpected {label} delta shape {:?}; expected {:?}",
            source.shape(),
            expected
        )));
    }
    let position = usize::try_from(position).map_err(|_| provider("negative Audio8 position"))?;
    if position >= full_shape[3] {
        return Err(provider("Audio8 cache position exceeds bound"));
    }
    let source = source.as_slice().expect("contiguous ONNX cache delta");
    let heads = full_shape[2];
    let width = full_shape[4];
    for layer in 0..full_shape[0] {
        for head in 0..heads {
            let src = (layer * heads + head) * width;
            let dst = ((layer * heads + head) * full_shape[3] + position) * width;
            destination[dst..dst + width].copy_from_slice(&source[src..src + width]);
        }
    }
    Ok(())
}

fn copy_state(
    destination: &mut [f32],
    expected_shape: &[usize],
    value: &ort::value::DynValue,
    label: &str,
) -> Result<()> {
    let source = value.try_extract_tensor::<f32>().map_err(ort_error)?;
    if source.shape() != expected_shape {
        return Err(provider(format!(
            "unexpected {label} state shape {:?}; expected {:?}",
            source.shape(),
            expected_shape
        )));
    }
    destination.copy_from_slice(source.as_slice().expect("contiguous ONNX state"));
    Ok(())
}

struct CacheSet {
    values: Vec<Vec<f32>>,
    shapes: Vec<Vec<usize>>,
    names: Vec<String>,
}

impl CacheSet {
    fn fast(manifest: &RuntimeManifest) -> Self {
        Self::new(
            manifest.num_fast_layers,
            vec![
                1,
                manifest.fast_n_local_heads,
                AUDIO8_CODEBOOKS,
                manifest.fast_head_dim,
            ],
        )
    }

    fn new(layers: usize, shape: Vec<usize>) -> Self {
        let mut values = Vec::with_capacity(layers * 2);
        let mut shapes = Vec::with_capacity(layers * 2);
        let mut names = Vec::with_capacity(layers * 2);
        let len = shape.iter().product();
        for layer in 0..layers {
            for kind in ["key", "value"] {
                values.push(vec![0.0; len]);
                shapes.push(shape.clone());
                names.push(format!("cache_{kind}_{layer}"));
            }
        }
        Self {
            values,
            shapes,
            names,
        }
    }

    fn inputs(&self) -> Vec<(String, SessionInputValue<'static>)> {
        self.names
            .iter()
            .zip(&self.shapes)
            .zip(&self.values)
            .map(|((name, shape), value)| named_tensor(name, shape.clone(), value.clone()))
            .collect::<Result<Vec<_>>>()
            .expect("validated cache tensors")
    }

    fn update(
        &mut self,
        outputs: &ort::session::SessionOutputs<'_, '_>,
        offset: usize,
        positions: &[i64],
    ) -> Result<()> {
        for (index, cache) in self.values.iter_mut().enumerate() {
            let delta = outputs[offset + index]
                .try_extract_tensor::<f32>()
                .map_err(ort_error)?;
            let shape = delta.shape();
            if shape.len() != 4
                || shape[0] != 1
                || shape[1] != self.shapes[index][1]
                || shape[3] != self.shapes[index][3]
                || shape[2] != positions.len()
            {
                return Err(provider(format!(
                    "unexpected cache delta shape {:?}",
                    shape
                )));
            }
            for (delta_position, &position) in positions.iter().enumerate() {
                let position =
                    usize::try_from(position).map_err(|_| provider("negative cache position"))?;
                if position >= self.shapes[index][2] {
                    return Err(provider("cache position exceeds bound"));
                }
                for head in 0..self.shapes[index][1] {
                    let width = self.shapes[index][3];
                    let dst = (head * self.shapes[index][2] + position) * width;
                    let src = (head * positions.len() + delta_position) * width;
                    cache[dst..dst + width].copy_from_slice(
                        &delta.as_slice().expect("contiguous ONNX output")[src..src + width],
                    );
                }
            }
        }
        Ok(())
    }
}

fn session(path: &Path) -> Result<Session> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(1);
    Session::builder()
        .map_err(ort_error)?
        .with_intra_threads(threads)
        .map_err(ort_error)?
        .with_inter_threads(1)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

fn validate_manifest(manifest: &RuntimeManifest) -> Result<()> {
    if manifest.sample_rate != AUDIO8_SAMPLE_RATE_HZ
        || manifest.num_codebooks != AUDIO8_CODEBOOKS
        || manifest.codebook_size != AUDIO8_CODEBOOK_SIZE
        || manifest.semantic_begin_id != AUDIO8_SEMANTIC_BEGIN
        || manifest.im_end_id != 4096
        || manifest.max_seq_len != 2048
    {
        return Err(provider(
            "Audio8 runtime manifest does not match the pinned 0.1B contract",
        ));
    }
    Ok(())
}

fn validate_session_inputs(session: &Session, label: &str, expected: &[&str]) -> Result<()> {
    let actual = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(provider(format!(
            "Audio8 {label} graph inputs {actual:?}; expected {expected:?}",
        )));
    }
    Ok(())
}

fn named_tensor<T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug + Clone + 'static>(
    name: &str,
    shape: Vec<usize>,
    values: Vec<T>,
) -> Result<(String, SessionInputValue<'static>)> {
    let tensor = Tensor::from_array((shape, values)).map_err(ort_error)?;
    Ok((name.to_string(), tensor.into()))
}

fn read_i64_npy(path: &Path, expected_rows: usize) -> Result<Vec<i64>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 128 || &bytes[..6] != b"\x93NUMPY" || bytes[6] != 1 || bytes[7] != 0 {
        return Err(provider("unsupported Audio8 reference_codes.npy header"));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])
        .map_err(|_| provider("non-UTF8 numpy header"))?;
    if !header.contains("'descr': '<i8'")
        || !header.contains("'fortran_order': False")
        || !header.contains(&format!("({},", expected_rows))
    {
        return Err(provider("unexpected Audio8 reference-code numpy layout"));
    }
    let payload = &bytes[10 + header_len..];
    if payload.len() % 8 != 0 || payload.is_empty() {
        return Err(provider("invalid Audio8 reference-code payload"));
    }
    Ok(payload
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect())
}

fn clean_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
fn cancelled(cancel: &Cancel) -> Result<()> {
    if cancel.is_shutdown() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "audio8",
        message: message.into(),
    }
}
fn ort_error(error: ort::Error) -> Error {
    provider(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_codes_are_little_endian_i64_matrix() {
        let path = std::env::temp_dir().join("audio8-test.npy");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let header =
            b"{'descr': '<i8', 'fortran_order': False, 'shape': (10, 1), }                  \n";
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header);
        for value in 0..10_i64 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            read_i64_npy(&path, 10).unwrap(),
            (0..10).collect::<Vec<_>>()
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounds_are_kept_small_and_fixed() {
        assert_eq!(AUDIO8_SAMPLE_RATE_HZ, 44_100);
        assert_eq!(AUDIO8_MAX_FRAMES, 256);
        assert_eq!(AUDIO8_CODEBOOKS, 10);
        assert_eq!(AUDIO8_CODEBOOK_SIZE, 4096);
    }
}
