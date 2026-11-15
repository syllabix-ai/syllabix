//! Fluid Speech Pro int8 ONNX: NeMo mel frontend and greedy TDT decode.
//!
//! Pinned export at
//! `FluidVoice-Labs/fluid-speech-pro-onnx` revision `fa588e9f87f1b9ad0ca2e2d8f7eaf84dc3803581`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use std::sync::OnceLock;

use regex::Regex;

use crate::fluid_onnx_mel::{nemo_log_mel, slaney_mel_filterbank, N_FREQ, N_MELS};
use crate::Stt;
use syllabix_native::error::{Error, Result};
use syllabix_native::models::{Fetcher, ModelCache, Progress};
use syllabix_native::onnx::{OnnxData, OnnxSession, OnnxTensor, OrtSession};
use syllabix_native::types::{Transcript, Utterance};
use syllabix_native::Cancel;

/// Config / log name for this engine.
pub const PROVIDER_NAME: &str = "fluid-onnx";

pub const ENCODER_ASSET: &str = "fluid-speech-pro-onnx-encoder";
pub const ENCODER_DATA_ASSET: &str = "fluid-speech-pro-onnx-encoder-data";
pub const DECODER_JOINT_ASSET: &str = "fluid-speech-pro-onnx-decoder-joint";
pub const DECODER_JOINT_DATA_ASSET: &str = "fluid-speech-pro-onnx-decoder-joint-data";
pub const VOCAB_ASSET: &str = "fluid-speech-pro-onnx-vocab";

pub const SAMPLE_RATE_HZ: u32 = 16_000;
const BLANK: i32 = 8192;
const TOKEN_LOGITS: usize = 8193;
const TOTAL_LOGITS: usize = 8198;
const STATE_LAYERS: i64 = 2;
const STATE_HIDDEN: i64 = 640;
const ENCODER_DIM: i64 = 1024;
const MAX_EMIT_PER_FRAME: usize = 10;

static INTRA_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Bench and `run` can pin ORT intra-op threads before the first `from_cache`.
pub fn set_intra_threads(n: usize) {
    INTRA_THREADS.store(n, Ordering::SeqCst);
}

fn effective_intra_threads() -> usize {
    let n = INTRA_THREADS.load(Ordering::SeqCst);
    if n > 0 {
        return n;
    }
    std::thread::available_parallelism()
        .map(|p| p.get().min(4))
        .unwrap_or(1)
}

/// ISO codes accepted by yaml (`auto` is handled in config, not stored on the engine).
pub const ALLOWED_LANGUAGE_CODES: &[&str] = &[
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt",
    "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
];

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

fn load_session(path: &Path, label: &str, intra_threads: usize) -> Result<Box<dyn OnnxSession>> {
    OrtSession::load_with_threads(path, PROVIDER_NAME, label, intra_threads)
        .map(|session| Box::new(session) as _)
}

fn decode_space_pattern() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    CELL.get_or_init(|| Regex::new(r"\A\s|\s\B|(\s)\b").expect("DECODE_SPACE_PATTERN compiles"))
}

/// Spacing rule from onnx-asr `DECODE_SPACE_PATTERN`.
pub fn apply_onnx_asr_spacing(text: &str) -> String {
    decode_space_pattern()
        .replace_all(
            text,
            |caps: &regex::Captures<'_>| {
                if caps.get(1).is_some() {
                    " "
                } else {
                    ""
                }
            },
        )
        .trim()
        .to_string()
}

/// Parse `vocab.txt` lines `token id`.
pub fn parse_vocab(text: &str) -> Result<Vec<String>> {
    let mut pieces = vec![String::new(); TOKEN_LOGITS];
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((piece, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let id: usize = id
            .parse()
            .map_err(|_| provider(format!("bad vocab id in line {line:?}")))?;
        if id >= TOKEN_LOGITS {
            continue;
        }
        pieces[id] = piece.replace('\u{2581}', " ");
    }
    if pieces[BLANK as usize].is_empty() {
        pieces[BLANK as usize] = "<blk>".into();
    }
    Ok(pieces)
}

fn tokens_to_text(tokens: &[i32], vocab: &[String]) -> String {
    let raw: String = tokens
        .iter()
        .filter_map(|&id| {
            if id >= 0 && (id as usize) < vocab.len() {
                Some(vocab[id as usize].as_str())
            } else {
                None
            }
        })
        .collect();
    apply_onnx_asr_spacing(&raw)
}

pub fn validate_stt_language(language: &str) -> Result<()> {
    if language == crate::language::LANGUAGE_AUTO {
        return Ok(());
    }
    if ALLOWED_LANGUAGE_CODES.contains(&language) {
        return Ok(());
    }
    Err(Error::Config {
        field: "pipeline.stt.language".into(),
        message: format!(
            "unsupported value {language:?} (allowed: auto, {})",
            ALLOWED_LANGUAGE_CODES.join(", ")
        ),
    })
}

/// In-process Fluid Speech Pro TDT adapter (finalize only).
pub struct FluidOnnxStt {
    encoder: Box<dyn OnnxSession>,
    decoder_joint: Box<dyn OnnxSession>,
    vocab: Vec<String>,
    mel_filters: Vec<f32>,
    language: String,
    intra_threads: usize,
}

impl FluidOnnxStt {
    pub fn from_paths(
        encoder: PathBuf,
        decoder_joint: PathBuf,
        vocab: PathBuf,
        intra_threads: usize,
    ) -> Result<Self> {
        let vocab_text = std::fs::read_to_string(&vocab)
            .map_err(|err| provider(format!("could not read {}: {err}", vocab.display())))?;
        let mel_filters = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE_HZ, 0.0, 8000.0);
        Ok(Self {
            encoder: load_session(&encoder, "encoder", intra_threads)?,
            decoder_joint: load_session(&decoder_joint, "decoder_joint", intra_threads)?,
            vocab: parse_vocab(&vocab_text)?,
            mel_filters,
            language: crate::language::LANGUAGE_AUTO.into(),
            intra_threads,
        })
    }

    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let intra_threads = effective_intra_threads();
        let ids = [
            ENCODER_ASSET,
            ENCODER_DATA_ASSET,
            DECODER_JOINT_ASSET,
            DECODER_JOINT_DATA_ASSET,
            VOCAB_ASSET,
        ];
        let mut paths = Vec::with_capacity(ids.len());
        for id in ids {
            let asset = cache
                .manifest()
                .asset(id)
                .ok_or_else(|| Error::ModelCache {
                    message: format!("manifest does not contain the {id} asset"),
                })?;
            paths.push(cache.resolve(asset, fetcher, progress, cancel)?);
        }
        Self::from_paths(
            paths[0].clone(),
            paths[2].clone(),
            paths[4].clone(),
            intra_threads,
        )
    }

    pub fn from_parts(
        encoder: Box<dyn OnnxSession>,
        decoder_joint: Box<dyn OnnxSession>,
        vocab: Vec<String>,
        intra_threads: usize,
    ) -> Self {
        let mel_filters = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE_HZ, 0.0, 8000.0);
        Self {
            encoder,
            decoder_joint,
            vocab,
            mel_filters,
            language: crate::language::LANGUAGE_AUTO.into(),
            intra_threads,
        }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        validate_stt_language(&language)?;
        self.language = language;
        Ok(self)
    }

    pub fn intra_op_threads(&self) -> i32 {
        i32::try_from(self.intra_threads).unwrap_or(i32::MAX)
    }

    /// BCT `[C, T_pad]` layout from the encoder, decoded out to `T_decode` rows.
    fn transpose_encoded_bct(
        data: &[f32],
        c_len: usize,
        t_padded: usize,
        t_decode: usize,
    ) -> Result<Vec<f32>> {
        if data.len() < t_padded * c_len {
            return Err(provider(
                "encoder output is shorter than its time dimension",
            ));
        }
        if t_decode == 0 {
            return Ok(Vec::new());
        }
        let mut transposed = vec![0.0_f32; t_decode * c_len];
        for t in 0..t_decode {
            for c in 0..c_len {
                transposed[t * c_len + c] = data[c * t_padded + t];
            }
        }
        Ok(transposed)
    }

    fn run_encoder(
        &mut self,
        mel: &[f32],
        n_frames: usize,
        valid_len: usize,
    ) -> Result<(Vec<f32>, i64)> {
        let audio = OnnxTensor::f32(vec![1, N_MELS as i64, n_frames as i64], mel.to_vec());
        let length = OnnxTensor::i64(vec![1], vec![valid_len as i64]);
        let outs = self.encoder.run(
            &[("audio_signal", &audio), ("length", &length)],
            &["outputs", "encoded_lengths"],
        )?;
        let hidden = outs
            .first()
            .ok_or_else(|| provider("encoder returned no outputs"))?;
        let encoded_len = outs
            .get(1)
            .and_then(|t| match &t.data {
                OnnxData::I64(d) => d.first().copied(),
                OnnxData::I32(d) => d.first().map(|&v| i64::from(v)),
                _ => None,
            })
            .ok_or_else(|| provider("encoder returned no encoded_lengths"))?;
        let data = match &hidden.data {
            OnnxData::F32(d) => d.clone(),
            _ => return Err(provider("encoder outputs are not f32")),
        };
        let c_len = ENCODER_DIM as usize;
        let t_padded = hidden
            .shape
            .get(2)
            .copied()
            .filter(|&d| d > 0)
            .map(|d| d as usize)
            .unwrap_or_else(|| data.len().saturating_div(c_len));
        let t_decode = (encoded_len.max(0) as usize).min(t_padded);
        let transposed = Self::transpose_encoded_bct(&data, c_len, t_padded, t_decode)?;
        Ok((transposed, t_decode as i64))
    }

    fn greedy_tdt(
        &mut self,
        encoded: &[f32],
        encoded_len: i64,
        cancel: &Cancel,
    ) -> Result<Vec<i32>> {
        let t_len = encoded_len.max(0) as usize;
        if t_len == 0 {
            return Ok(Vec::new());
        }
        let mut state1 = vec![0.0f32; (STATE_LAYERS * STATE_HIDDEN) as usize];
        let mut state2 = vec![0.0f32; (STATE_LAYERS * STATE_HIDDEN) as usize];
        let mut tokens = Vec::new();
        let mut t = 0usize;
        let mut emitted = 0usize;
        while (t as i64) < encoded_len {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let base = t * ENCODER_DIM as usize;
            let frame = encoded[base..base + ENCODER_DIM as usize].to_vec();
            let frame_tensor = OnnxTensor::f32(vec![1, ENCODER_DIM, 1], frame);
            let last = tokens.last().copied().unwrap_or(BLANK);
            let targets = OnnxTensor::i32(vec![1, 1], vec![last]);
            let target_length = OnnxTensor::i32(vec![1], vec![1]);
            let s1 = OnnxTensor::f32(vec![STATE_LAYERS, 1, STATE_HIDDEN], state1.clone());
            let s2 = OnnxTensor::f32(vec![STATE_LAYERS, 1, STATE_HIDDEN], state2.clone());
            let outs = self.decoder_joint.run(
                &[
                    ("encoder_outputs", &frame_tensor),
                    ("targets", &targets),
                    ("target_length", &target_length),
                    ("input_states_1", &s1),
                    ("input_states_2", &s2),
                ],
                &[
                    "outputs",
                    "prednet_lengths",
                    "output_states_1",
                    "output_states_2",
                ],
            )?;
            let logits = outs
                .first()
                .ok_or_else(|| provider("decoder_joint returned no outputs"))?;
            let logits_f = match &logits.data {
                OnnxData::F32(d) => d.as_slice(),
                _ => return Err(provider("decoder logits are not f32")),
            };
            if logits_f.len() < TOTAL_LOGITS {
                return Err(provider("decoder logits too short"));
            }
            let tok = argmax(&logits_f[..TOKEN_LOGITS])? as i32;
            let dur = argmax(&logits_f[TOKEN_LOGITS..TOTAL_LOGITS])? as i32;
            if tok != BLANK {
                if let Some(out1) = outs.get(2) {
                    if let OnnxData::F32(d) = &out1.data {
                        state1.clone_from(d);
                    }
                }
                if let Some(out2) = outs.get(3) {
                    if let OnnxData::F32(d) = &out2.data {
                        state2.clone_from(d);
                    }
                }
                tokens.push(tok);
                emitted += 1;
            }
            if dur > 0 {
                t += dur as usize;
                emitted = 0;
            } else if tok == BLANK || emitted == MAX_EMIT_PER_FRAME {
                t += 1;
                emitted = 0;
            }
        }
        Ok(tokens)
    }

    fn transcribe_pcm(&mut self, pcm_f32: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let (mel, n_frames, valid_len) = nemo_log_mel(pcm_f32, &self.mel_filters)?;
        let (encoded, encoded_len) = self.run_encoder(&mel, n_frames, valid_len)?;
        let tokens = self.greedy_tdt(&encoded, encoded_len, cancel)?;
        Ok(tokens_to_text(&tokens, &self.vocab))
    }
}

fn argmax(slice: &[f32]) -> Result<usize> {
    let (mut best, mut value) = (0_usize, f32::NEG_INFINITY);
    for (i, &v) in slice.iter().enumerate() {
        if v > value {
            value = v;
            best = i;
        }
    }
    Ok(best)
}

impl Stt for FluidOnnxStt {
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
        let pcm = utterance.pcm_without_splitter_flush_pad();
        if pcm.is_empty() {
            return Err(provider("utterance has no samples"));
        }
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let text = self.transcribe_pcm(&audio, cancel)?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: self.language.clone(),
        })
    }

    fn supports_partials(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syllabix_native::onnx::{MockSession, OnnxTensor};
    use syllabix_native::types::TurnId;
    use syllabix_native::{Cancel, Error};

    fn names(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| (*w).to_string()).collect()
    }

    fn tiny_vocab() -> Vec<String> {
        let mut v = vec![String::new(); TOKEN_LOGITS];
        v[0] = " hi".into();
        v[1] = " there".into();
        v[BLANK as usize] = "<blk>".into();
        v
    }

    fn logits(tok_peak: usize, dur_peak: usize) -> OnnxTensor {
        let mut values = vec![-20.0f32; TOTAL_LOGITS];
        values[tok_peak] = 10.0;
        values[TOKEN_LOGITS + dur_peak] = 10.0;
        OnnxTensor::f32(vec![1, 1, 1, TOTAL_LOGITS as i64], values)
    }

    fn states() -> OnnxTensor {
        OnnxTensor::f32(
            vec![STATE_LAYERS, 1, STATE_HIDDEN],
            vec![0.0; (STATE_LAYERS * STATE_HIDDEN) as usize],
        )
    }

    #[test]
    fn spacing_rule_matches_onnx_asr() {
        assert_eq!(apply_onnx_asr_spacing("hello world"), "hello world");
        assert_eq!(apply_onnx_asr_spacing(" x"), "x");
    }

    #[test]
    fn parse_vocab_reads_blank() {
        let text = "▁a 0\n<blk> 8192\n";
        let v = parse_vocab(text).expect("vocab");
        assert_eq!(v[0], " a");
        assert_eq!(v[BLANK as usize], "<blk>");
    }

    #[test]
    fn validate_language_accepts_european_set() {
        validate_stt_language("en").expect("en");
        validate_stt_language("auto").expect("auto");
        assert!(validate_stt_language("xx").is_err());
    }

    #[test]
    fn with_language_stores_code() {
        let stt = FluidOnnxStt::from_parts(
            Box::new(MockSession::script(
                names(&["audio_signal", "length"]),
                names(&["outputs", "encoded_lengths"]),
                vec![],
            )),
            Box::new(MockSession::script(
                names(&["encoder_outputs"]),
                names(&["outputs"]),
                vec![],
            )),
            tiny_vocab(),
            2,
        )
        .with_language("de")
        .expect("de");
        assert_eq!(stt.language(), "de");
    }

    #[test]
    fn mock_tdt_emits_token_then_advances_duration() {
        let steps = vec![
            Ok(vec![
                logits(0, 1),
                OnnxTensor::i32(vec![1], vec![1]),
                states(),
                states(),
            ]),
            Ok(vec![
                logits(BLANK as usize, 0),
                OnnxTensor::i32(vec![1], vec![1]),
                states(),
                states(),
            ]),
        ];
        let encoder = MockSession::script(
            names(&["audio_signal", "length"]),
            names(&["outputs", "encoded_lengths"]),
            vec![Ok(vec![
                OnnxTensor::f32(vec![1, ENCODER_DIM, 2], vec![0.0; 2 * ENCODER_DIM as usize]),
                OnnxTensor::i64(vec![1], vec![2]),
            ])],
        );
        let decoder = MockSession::script(
            names(&[
                "encoder_outputs",
                "targets",
                "target_length",
                "input_states_1",
                "input_states_2",
            ]),
            names(&[
                "outputs",
                "prednet_lengths",
                "output_states_1",
                "output_states_2",
            ]),
            steps,
        );
        let mut stt =
            FluidOnnxStt::from_parts(Box::new(encoder), Box::new(decoder), tiny_vocab(), 1);
        let text = stt
            .transcribe_pcm(&vec![0.0; 3200], &Cancel::new())
            .expect("decode");
        assert_eq!(text, "hi");
    }

    #[test]
    fn transcribe_checks_cancel() {
        let stt = FluidOnnxStt::from_parts(
            Box::new(MockSession::failing(
                names(&["audio_signal", "length"]),
                names(&["outputs"]),
                "nope",
            )),
            Box::new(MockSession::script(
                names(&["encoder_outputs"]),
                names(&["outputs"]),
                vec![],
            )),
            tiny_vocab(),
            1,
        );
        let cancel = Cancel::new();
        cancel.shutdown();
        let utterance = Utterance {
            turn: TurnId(0),
            frames: vec![syllabix_native::types::AudioFrame {
                seq: 0,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0; 160],
                capture_pcm: None,
                flush_pad_samples: 0,
            }],
        };
        let mut stt2 = stt;
        assert!(matches!(
            stt2.transcribe_pcm(&[0.0; 320], &cancel),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            stt2.transcribe(&utterance, &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn transpose_encoded_bct_uses_padded_stride_not_decode_len() {
        let c_len = 4usize;
        let t_padded = 5usize;
        let t_decode = 4usize;
        let mut bct = vec![0.0_f32; c_len * t_padded];
        for c in 0..c_len {
            for t in 0..t_padded {
                bct[c * t_padded + t] = (c * 10 + t) as f32;
            }
        }
        let got = FluidOnnxStt::transpose_encoded_bct(&bct, c_len, t_padded, t_decode).expect("ok");
        assert_eq!(got.len(), t_decode * c_len);
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 10.0);
        assert_eq!(got[c_len], 1.0);
    }

    #[test]
    fn intra_op_threads_reports_effective_count() {
        set_intra_threads(4);
        let stt = FluidOnnxStt::from_parts(
            Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
            Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
            tiny_vocab(),
            4,
        );
        assert_eq!(stt.intra_op_threads(), 4);
        set_intra_threads(0);
    }
}

#[cfg(test)]
#[path = "fluid_onnx_tests.rs"]
mod fluid_onnx_tests;
