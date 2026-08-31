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
use sentencepiece_rs::SentencePieceProcessor;

use crate::audio::{f32_to_i16, PcmConverter, PcmFormat};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Tts;
use crate::speech_text::{speak_text_for_tts, take_sentences, ThinkFilter};
use crate::{Cancel, Error, GenerationId, Result, SynthesizedAudio, TokenChunk, TurnId};

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

// The pinned English 2026-04 Pocket export retains the upstream flow model's
// one-step decoder and end-of-speech calibration. These are model semantics,
// not Syllabix tuning knobs: a zero seed collapses flow-matching diversity and
// a higher EOS threshold lets the autoregressive loop run into repeated tails.
const FLOW_STEPS: usize = 1;
const EOS_THRESHOLD: f32 = -4.0;
const FLOW_TEMPERATURE: f32 = 0.7;
const DECODER_BATCH_FRAMES: usize = 12;
// FlowLM's EOS score marks the beginning of the ending, not the exact PCM
// boundary. Keep generating through the export's small post-EOS window so
// Mimi's recurrent convolution can finish sentence-final phonemes. The
// window includes the EOS frame itself: normal prompts get that frame plus
// two continuations; very short prompts get four continuations.
const NORMAL_EOS_WINDOW_FRAMES: usize = 3;
const SHORT_EOS_WINDOW_FRAMES: usize = 5;
const SHORT_PROMPT_WORDS: usize = 4;

/// P1 loader for the pinned English ONNX graph set.
pub struct PocketTts {
    text_conditioner: Session,
    flow_main: Session,
    flow: Session,
    decoder: Session,
    flow_state: Vec<StateSpec>,
    mimi_state: Vec<StateSpec>,
    voice: PathBuf,
    tokenizer: SentencePieceProcessor,
    think: ThinkFilter,
    buffer: String,
    turn: Option<TurnId>,
    generation: Option<GenerationId>,
    next_index: u32,
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
            tokenizer: SentencePieceProcessor::open(tokenizer)
                .map_err(|err| provider(&format!("could not load tokenizer.model: {err}")))?,
            think: ThinkFilter::default(),
            buffer: String::new(),
            turn: None,
            generation: None,
            next_index: 0,
        })
    }

    fn reset_turn(&mut self) {
        self.think = ThinkFilter::default();
        self.buffer.clear();
        self.turn = None;
        self.generation = None;
        self.next_index = 0;
    }

    fn start_turn(&mut self, token: &TokenChunk, cancel: &Cancel) -> Result<()> {
        if cancel.is_stale(token.generation) {
            self.reset_turn();
            return Err(Error::Cancelled);
        }
        if self.turn != Some(token.turn) || self.generation != Some(token.generation) {
            self.reset_turn();
            self.turn = Some(token.turn);
            self.generation = Some(token.generation);
        }
        Ok(())
    }

    /// Generate Pocket TTS frames for one cleaned sentence. The exported Flow
    /// LM is recurrent: text conditioning is supplied once, then every latent
    /// frame is fed back through its returned state. Mimi is likewise kept
    /// stateful so individual frames can reach playback as they are decoded.
    fn synthesize_sentence_into(
        &mut self,
        text: &str,
        token: &TokenChunk,
        is_last_sentence: bool,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        let ids = self
            .tokenizer
            .encode_to_ids(text)
            .map_err(|err| provider(&format!("could not tokenize Pocket TTS text: {err}")))?
            .into_iter()
            .map(|id| id as i64)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Ok(());
        }
        let token_count = ids.len();
        let outputs = self
            .text_conditioner
            .run(ort::inputs!["token_ids" => ([1_usize, ids.len()], ids)].map_err(ort_error)?)
            .map_err(ort_error)?;
        let embeddings = RawTensor::f32(
            vec![1, token_count as i64, 1024],
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(ort_error)?
                .iter()
                .copied()
                .collect(),
        );
        let mut flow_state = voice_state(&self.flow_state, &self.voice)?;
        // Feed the complete text into FlowLM before it starts producing audio.
        // The generation loop below then has only a new audio frame and the
        // state that this call returned, matching the exported model's order.
        let empty_sequence = RawTensor::f32(vec![1, 0, 32], vec![]);
        let mut main = RawRunner::new(&mut self.flow_main)?;
        let mut inputs = vec![
            ("sequence", &empty_sequence),
            ("text_embeddings", &embeddings),
        ];
        inputs.extend(
            self.flow_state
                .iter()
                .zip(&flow_state)
                .map(|(s, v)| (s.input_name.as_str(), v)),
        );
        let state_names = self
            .flow_state
            .iter()
            .map(|s| s.output_name.as_str())
            .collect::<Vec<_>>();
        flow_state = main.run(&inputs, &state_names)?;
        let mut mimi_state = self
            .mimi_state
            .iter()
            .map(StateSpec::initial)
            .collect::<Vec<_>>();
        // The FlowLM uses a NaN sentinel for the first autoregressive input;
        // its learned audio-BOS embedding replaces that sentinel internally.
        // Later iterations feed back the generated latent below.
        let mut previous = RawTensor::f32(vec![1, 1, 32], vec![f32::NAN; 32]);
        let mut converter = PcmConverter::new(
            PcmFormat {
                sample_rate_hz: 24_000,
                channels: 1,
            },
            PcmFormat {
                sample_rate_hz: 16_000,
                channels: 1,
            },
        )?;
        // The upstream export limits text chunks to 50 tokens. One generated
        // frame is 80 ms, and this conservative cap bounds CPU work while the
        // EOS head remains the normal stopping condition.
        let max_frames = (text.chars().count().saturating_mul(2)).clamp(8, 250);
        let eos_window_frames = eos_window_frames(text);
        let mut eos_frame = None;
        let mut pending_latents = Vec::with_capacity(DECODER_BATCH_FRAMES * 32);
        for frame in 0..max_frames {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            // EOS is an acoustic look-ahead marker. Do not feed zero latents
            // to Mimi: continue the same FlowLM/Mimi state for the bounded
            // model-defined window, then stop before another latent is made.
            if eos_window_exhausted_before_frame(frame, eos_frame, eos_window_frames) {
                break;
            }
            let empty_text = RawTensor::f32(vec![1, 0, 1024], vec![]);
            let mut main = RawRunner::new(&mut self.flow_main)?;
            let mut inputs = vec![("sequence", &previous), ("text_embeddings", &empty_text)];
            inputs.extend(
                self.flow_state
                    .iter()
                    .zip(&flow_state)
                    .map(|(s, v)| (s.input_name.as_str(), v)),
            );
            let mut names = vec!["conditioning", "eos_logit"];
            names.extend(self.flow_state.iter().map(|s| s.output_name.as_str()));
            let result = main.run(&inputs, &names)?;
            let conditioning = result[0].clone();
            let eos = result[1]
                .f32_data()?
                .first()
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            flow_state = result.into_iter().skip(2).collect();
            if frame > 0 && eos >= EOS_THRESHOLD && eos_frame.is_none() {
                eos_frame = Some(frame);
            }
            let is_final_frame = frame + 1 == max_frames
                || eos_window_completes_on_frame(frame, eos_frame, eos_window_frames);
            // Flow matching starts each audio frame from fresh Gaussian noise,
            // then integrates the learned velocity field. Reusing an all-zero
            // seed made distinct frames converge on the same fragment.
            let mut latent = standard_normal_tensor(32, FLOW_TEMPERATURE.sqrt())?;
            for step in 0..FLOW_STEPS {
                let s = RawTensor::f32(vec![1, 1], vec![step as f32 / FLOW_STEPS as f32]);
                let t = RawTensor::f32(vec![1, 1], vec![(step + 1) as f32 / FLOW_STEPS as f32]);
                let mut flow = RawRunner::new(&mut self.flow)?;
                let direction = flow
                    .run(
                        &[("c", &conditioning), ("s", &s), ("t", &t), ("x", &latent)],
                        &["flow_dir"],
                    )?
                    .into_iter()
                    .next()
                    .ok_or_else(|| provider("flow graph returned no latent"))?;
                let scale = 1.0 / FLOW_STEPS as f32;
                let values = latent
                    .f32_data()?
                    .iter()
                    .zip(direction.f32_data()?)
                    .map(|(current, velocity)| current + velocity * scale)
                    .collect();
                latent = RawTensor::f32(vec![1, 32], values);
            }
            previous = RawTensor::f32(vec![1, 1, 32], latent.f32_data()?.to_vec());
            pending_latents.extend_from_slice(previous.f32_data()?);

            // Mimi needs a small amount of following audio to make a stable
            // waveform. Decode its native 12-frame blocks, rather than a
            // separate 80 ms call for every generated frame.
            if pending_latents.len() / 32 < DECODER_BATCH_FRAMES && !is_final_frame {
                continue;
            }
            let batch_frames = pending_latents.len() / 32;
            let latent = RawTensor::f32(
                vec![1, batch_frames as i64, 32],
                std::mem::take(&mut pending_latents),
            );
            let mut decoder = RawRunner::new(&mut self.decoder)?;
            let mut decoder_inputs = vec![("latent", &latent)];
            decoder_inputs.extend(
                self.mimi_state
                    .iter()
                    .zip(&mimi_state)
                    .map(|(s, v)| (s.input_name.as_str(), v)),
            );
            let mut decoder_names = vec!["audio_frame"];
            decoder_names.extend(self.mimi_state.iter().map(|s| s.output_name.as_str()));
            let decoded = decoder.run(&decoder_inputs, &decoder_names)?;
            let mut pcm = f32_to_i16(&converter.push(decoded[0].f32_data()?));
            mimi_state = decoded.into_iter().skip(1).collect();
            if is_final_frame {
                pcm.extend(f32_to_i16(&converter.flush()));
            }
            if !pcm.is_empty() {
                let index = self.next_index;
                self.next_index += 1;
                on_audio(SynthesizedAudio {
                    turn: token.turn,
                    generation: token.generation,
                    index,
                    samples: pcm,
                    is_last: is_last_sentence && is_final_frame,
                })?;
            }
            if is_final_frame {
                break;
            }
        }
        Ok(())
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

fn eos_window_frames(text: &str) -> usize {
    if text.split_whitespace().count() <= SHORT_PROMPT_WORDS {
        SHORT_EOS_WINDOW_FRAMES
    } else {
        NORMAL_EOS_WINDOW_FRAMES
    }
}

fn eos_window_exhausted_before_frame(
    frame: usize,
    eos_frame: Option<usize>,
    window_frames: usize,
) -> bool {
    eos_frame.is_some_and(|eos| frame >= eos + window_frames)
}

fn eos_window_completes_on_frame(
    frame: usize,
    eos_frame: Option<usize>,
    window_frames: usize,
) -> bool {
    eos_frame.is_some_and(|eos| frame + 1 >= eos + window_frames)
}

impl Tts for PocketTts {
    fn name(&self) -> &'static str {
        "local"
    }

    fn model_id(&self) -> Option<&str> {
        Some("pocket-tts")
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
        self.start_turn(token, cancel)?;
        self.buffer
            .push_str(&self.think.push(&token.text, token.is_last));
        let sentences = take_sentences(&mut self.buffer, token.is_last);
        let last = sentences.len().saturating_sub(1);
        let mut emitted = false;
        for (i, sentence) in sentences.into_iter().enumerate() {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                self.reset_turn();
                return Err(Error::Cancelled);
            }
            let spoken = speak_text_for_tts(&sentence);
            if spoken.is_empty() {
                continue;
            }
            self.synthesize_sentence_into(
                &spoken,
                token,
                token.is_last && i == last,
                cancel,
                on_audio,
            )?;
            emitted = true;
        }
        if token.is_last && !emitted {
            let index = self.next_index;
            self.next_index += 1;
            on_audio(SynthesizedAudio {
                turn: token.turn,
                generation: token.generation,
                index,
                samples: vec![0; 16],
                is_last: true,
            })?;
        }
        Ok(())
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

/// Sample one standard-normal latent using the OS CSPRNG. Pocket TTS is a
/// flow-matching model, so each generated audio frame needs a new noise seed;
/// a fixed zero vector is not a valid inference input.
fn standard_normal_tensor(width: usize, scale: f32) -> Result<RawTensor> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(provider(
            "Pocket TTS noise scale must be positive and finite",
        ));
    }
    let mut bytes = vec![0_u8; width.saturating_add(width % 2) * 4];
    getrandom::getrandom(&mut bytes)
        .map_err(|err| provider(&format!("could not sample Pocket TTS noise: {err}")))?;
    let uniforms = bytes
        .chunks_exact(4)
        .map(|chunk| {
            // Keep away from zero so Box-Muller remains finite.
            (u32::from_le_bytes(chunk.try_into().expect("four bytes")) as f64 + 1.0)
                / (u32::MAX as f64 + 2.0)
        })
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(width);
    for pair in uniforms.chunks_exact(2) {
        let radius = (-2.0 * pair[0].ln()).sqrt();
        let angle = std::f64::consts::TAU * pair[1];
        values.push((radius * angle.cos()) as f32 * scale);
        if values.len() < width {
            values.push((radius * angle.sin()) as f32 * scale);
        }
    }
    debug_assert_eq!(values.len(), width);
    Ok(RawTensor::f32(vec![1, width as i64], values))
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

    #[test]
    fn flow_noise_is_finite_and_has_the_requested_shape() {
        let noise = standard_normal_tensor(31, FLOW_TEMPERATURE.sqrt()).expect("OS randomness");
        assert_eq!(noise.shape, vec![1, 31]);
        assert!(noise.f32.iter().all(|value| value.is_finite()));
        assert!(noise.f32.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn flow_noise_rejects_invalid_scale() {
        assert!(standard_normal_tensor(1, 0.0).is_err());
        assert!(standard_normal_tensor(1, f32::NAN).is_err());
    }

    #[test]
    fn eos_window_keeps_the_export_required_post_eos_frames() {
        assert_eq!(eos_window_frames("one two three four"), 5);
        assert_eq!(eos_window_frames("one two three four five"), 3);

        let eos = 7;
        let normal = eos_window_frames("a normal length prompt has five words");
        assert!(
            !eos_window_exhausted_before_frame(9, Some(eos), normal),
            "decode through the final continuation"
        );
        assert!(eos_window_completes_on_frame(9, Some(eos), normal));
        assert!(
            eos_window_exhausted_before_frame(10, Some(eos), normal),
            "do not generate a repeated tail frame"
        );
    }
}
