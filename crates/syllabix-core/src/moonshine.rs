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

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::onnx::{OnnxData, OnnxSession, OnnxTensor, OrtSession};
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

fn load_session(path: &Path, label: &str) -> Result<Box<dyn OnnxSession>> {
    OrtSession::load(path, PROVIDER_NAME, label).map(|session| Box::new(session) as _)
}

/// In-process Moonshine streaming adapter (batch finalize; small or medium).
///
/// Sessions are injected as [`OnnxSession`] so coverage tests can script
/// inference without downloading weights; production builds [`OrtSession`].
pub struct MoonshineStt {
    encoder: Box<dyn OnnxSession>,
    decoder: Box<dyn OnnxSession>,
    decoder_past: Box<dyn OnnxSession>,
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
        let tokenizer = MoonshineTokenizer::from_file(tokenizer_path)?;
        Self::from_parts(encoder, decoder, decoder_past, tokenizer)
    }

    /// Assemble an adapter from injected sessions. Production passes
    /// [`OrtSession`]; coverage tests pass scripted mocks.
    pub(crate) fn from_parts(
        encoder: Box<dyn OnnxSession>,
        decoder: Box<dyn OnnxSession>,
        decoder_past: Box<dyn OnnxSession>,
        tokenizer: MoonshineTokenizer,
    ) -> Result<Self> {
        Self::validate_contract(encoder.as_ref(), decoder.as_ref(), decoder_past.as_ref())?;
        let kv_map = build_kv_map(&decoder.output_names(), &decoder_past.input_names())?;
        let past_inputs = decoder_past.input_names();
        let past_outputs = decoder_past.output_names();
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

    fn validate_contract(
        encoder: &dyn OnnxSession,
        decoder: &dyn OnnxSession,
        past: &dyn OnnxSession,
    ) -> Result<()> {
        if encoder.input_names() != ["input_values", "attention_mask"] {
            return Err(provider(format!(
                "unexpected encoder inputs: {}",
                encoder.input_names().join(", ")
            )));
        }
        if encoder.output_names() != ["encoder_hidden_states"] {
            return Err(provider("unexpected encoder outputs"));
        }
        if decoder.input_names() != ["decoder_input_ids", "encoder_hidden_states"] {
            return Err(provider(format!(
                "unexpected decoder inputs: {}",
                decoder.input_names().join(", ")
            )));
        }
        if !decoder.output_names().starts_with(&["logits".to_string()]) {
            return Err(provider("decoder must emit logits first"));
        }
        if !past.output_names().starts_with(&["logits".to_string()]) {
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
        let audio_tensor = OnnxTensor::f32(vec![1, audio.len() as i64], audio);
        let mask_tensor = OnnxTensor::i64(vec![1, mask.len() as i64], mask);
        let hidden = {
            let out = self.encoder.run(
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
            OnnxData::F32(data) => data,
            _ => return Err(provider("encoder emitted non-f32 states")),
        };
        let hidden_tensor = OnnxTensor::f32(hidden_shape, hidden_data);

        // Clone the session contracts up front: the runners below borrow the
        // sessions mutably while the name tables are still needed.
        let kv_output_names: Vec<String> =
            self.decoder.output_names().into_iter().skip(1).collect();
        let past_inputs = self.past_inputs.clone();
        let past_outputs = self.past_outputs.clone();
        let kv_map = self.kv_map.clone();

        let max_tokens = ((pcm_f32.len() as f64 * MAX_TOKENS_PER_SECOND / 16_000.0) as usize + 8)
            .min(MAX_TOKENS_ABSOLUTE);
        let id_tensor = OnnxTensor::i64(vec![1, 1], vec![BOS]);
        let mut names: Vec<&str> = vec!["logits"];
        names.extend(kv_output_names.iter().map(|s| s.as_str()));
        let mut tensors = self
            .decoder
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
        let mut kv: Vec<OnnxTensor> = tensors.collect();
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
            let id_tensor = OnnxTensor::i64(vec![1, 1], vec![next]);
            let mut inputs: Vec<(&str, &OnnxTensor)> = Vec::with_capacity(2 + kv.len());
            inputs.push(("decoder_input_ids", &id_tensor));
            inputs.push(("encoder_hidden_states", &hidden_tensor));
            for (slot, tensor) in kv.iter().enumerate() {
                inputs.push((past_inputs[kv_map[slot].1].as_str(), tensor));
            }
            let out_names: Vec<&str> = past_outputs.iter().map(|s| s.as_str()).collect();
            let mut tensors = self.decoder_past.run(&inputs, &out_names)?.into_iter();
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
fn argmax(logits: &OnnxTensor) -> Result<i64> {
    match &logits.data {
        OnnxData::F32(data) => {
            let (mut best_index, mut best_value) = (0_i64, f32::NEG_INFINITY);
            for (index, value) in data.iter().enumerate() {
                if *value > best_value {
                    best_value = *value;
                    best_index = index as i64;
                }
            }
            Ok(best_index)
        }
        _ => Err(provider("logits are not f32")),
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
fn build_kv_map(decoder_outputs: &[String], past_inputs: &[String]) -> Result<Vec<(usize, usize)>> {
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

    // Full-weights listening gate lives in `moonshine_tests.rs`
    // (`native_recorded_fixtures_meet_word_match_gate`, ignored by default)
    // so the measured file carries no never-executed native body.
    #[test]
    fn provider_name_is_moonshine() {
        assert_eq!(PROVIDER_NAME, "moonshine");
        assert_eq!(LANGUAGE, "en");
        assert_eq!(crate::defaults::SttModel::Small.as_str(), "whisper-small");
    }
}

#[cfg(test)]
#[path = "moonshine_tests.rs"]
mod moonshine_tests;
