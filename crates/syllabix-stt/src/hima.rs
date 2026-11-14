//! Hima-ASR ONNX engine: log-mel frontend plus greedy KV-cache decode.
//!
//! Pinned to Prahaladha/hima-asr-ONNX revision
//! `18fa854f8ecaf2f48fe7416035643f86da3834d8` (int8 encoder/decoder). Config
//! yaml, `SttModel`, and manifest wiring are the next PR. Default CI uses
//! [`syllabix_native::onnx::MockSession`]; production loads [`OrtSession`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

use crate::npy;
use crate::stft::power_spectrogram;
use crate::Stt;
use syllabix_native::error::{Error, Result};
use syllabix_native::onnx::{OnnxData, OnnxSession, OnnxTensor, OrtSession};
use syllabix_native::types::{Transcript, Utterance};
use syllabix_native::Cancel;

/// Config / log name for this engine (not selectable yet).
pub const PROVIDER_NAME: &str = "hima";
/// 16 kHz mono, same as `hima_onnx.RATE`.
pub const SAMPLE_RATE_HZ: u32 = 16_000;
/// STFT size in `hima_onnx._log_mel`.
pub const N_FFT: usize = 400;
/// Hop size in samples.
pub const HOP: usize = 160;
/// rFFT bins for `N_FFT`.
pub const N_FREQ: usize = N_FFT / 2 + 1;
/// Mel bins in `mel_filters.npy`.
pub const N_MELS: usize = 128;
/// Encoder chunk size along the time axis (`hima_onnx.CHUNK`).
pub const CHUNK_FRAMES: usize = 100;
/// Clips shorter than 0.5 s are zero-padded to this many samples inside log-mel.
pub const MIN_SAMPLES: usize = 8_000;
/// Decoder embedding width (`embed.npy` columns, `sound` last dim).
pub const EMBED_DIM: usize = 640;
/// KV-cache layers in the int8 decoder.
pub const KV_LAYERS: usize = 18;
/// KV head dim in the int8 decoder.
pub const KV_DIM: usize = 256;
/// `hima_onnx` BOS id.
pub const BOS: i64 = 2;
/// `hima_onnx` EOS id.
pub const EOS: i64 = 1;
/// Default `max_pieces` in `hima_onnx._write`.
pub const MAX_PIECES: usize = 512;
/// Hima-ASR's stated range is 30 s. Longer audio is split into chunks of at
/// most this many seconds (quietest energy near the boundary) rather than
/// rejected. A 37 s clip decoded as one piece on CPU in the reference; we
/// still chunk so a runaway utterance cannot pin the encoder.
pub const MAX_CHUNK_SECONDS: u32 = 30;
/// [`MAX_CHUNK_SECONDS`] at [`SAMPLE_RATE_HZ`].
pub const MAX_CHUNK_SAMPLES: usize = MAX_CHUNK_SECONDS as usize * SAMPLE_RATE_HZ as usize;
const ENERGY_FRAME: usize = 320;
const SPLIT_WINDOW: usize = 2 * SAMPLE_RATE_HZ as usize;
const ENCODER_FILE: &str = "encoder_model_int8.onnx";
const DECODER_FILE: &str = "decoder_model_int8.onnx";
const JOINERS: [char; 2] = ['\u{200c}', '\u{200d}'];
const VIRAMA: char = '\u{0d4d}';
const ZWJ: char = '\u{200d}';
const MALAYALAM_START: char = '\u{0d00}';
const MALAYALAM_END: char = '\u{0d7f}';

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

/// Log-mel features: `[n_mels, n_frames]` C-order, time padded to a multiple of 100.
#[derive(Debug, Clone, PartialEq)]
pub struct LogMel {
    /// Always [`N_MELS`].
    pub n_mels: usize,
    /// Padded frame count (multiple of [`CHUNK_FRAMES`]).
    pub n_frames: usize,
    /// Frames before time padding (`n_valid` in the Python reference).
    pub n_valid: usize,
    /// Length `n_mels * n_frames`.
    pub values: Vec<f32>,
}

/// Three encoder downsample steps from `hima_onnx._post`.
pub fn post_frame_count(mut n: usize) -> usize {
    for _ in 0..3 {
        n = if n > 0 { (n - 1) / 2 + 1 } else { 0 };
    }
    n
}

/// Load `mel_filters.npy` with shape `(201, 128)` little-endian f32.
pub fn load_mel_filters(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    load_mel_filters_bytes(&bytes)
}

fn load_mel_filters_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let arr = npy::read_f32(bytes)?;
    if arr.shape != [N_FREQ, N_MELS] {
        return Err(provider(format!(
            "mel_filters.npy shape {:?} is not ({N_FREQ}, {N_MELS})",
            arr.shape
        )));
    }
    Ok(arr.data)
}

/// `_log_mel`: 16 kHz mono f32 in, padded log-mel and `n_valid` out.
pub fn log_mel(samples: &[f32], filters: &[f32]) -> Result<LogMel> {
    if filters.len() != N_FREQ * N_MELS {
        return Err(provider(format!(
            "mel filterbank is {} bins, expected {}",
            filters.len(),
            N_FREQ * N_MELS
        )));
    }
    let mut x = samples.to_vec();
    if x.len() < MIN_SAMPLES {
        x.resize(MIN_SAMPLES, 0.0);
    }
    let spec = power_spectrogram(&x, N_FFT, N_FFT, HOP)?;
    if spec.n_freq != N_FREQ {
        return Err(provider(format!(
            "rFFT produced {} bins, expected {N_FREQ}",
            spec.n_freq
        )));
    }
    let n_valid = spec.n_frames.saturating_sub(1);
    let mut log = vec![0.0f32; N_MELS * n_valid];
    for t in 0..n_valid {
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for k in 0..N_FREQ {
                let mag = spec.values[t * N_FREQ + k];
                let w = filters[k * N_MELS + m];
                acc += w * mag;
            }
            log[m * n_valid + t] = acc.max(1e-10).log10();
        }
    }
    if let Some(max) = log.iter().copied().reduce(f32::max) {
        let floor = max - 8.0;
        for v in &mut log {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0) / 4.0;
        }
    }
    let pad = (CHUNK_FRAMES - (n_valid % CHUNK_FRAMES)) % CHUNK_FRAMES;
    let n_frames = n_valid + pad;
    let mut values = vec![0.0f32; N_MELS * n_frames];
    if n_valid > 0 {
        for m in 0..N_MELS {
            let src = m * n_valid;
            let dst = m * n_frames;
            values[dst..dst + n_valid].copy_from_slice(&log[src..src + n_valid]);
        }
    }
    Ok(LogMel {
        n_mels: N_MELS,
        n_frames,
        n_valid,
        values,
    })
}

/// Sum of `_post(min(CHUNK, remaining))` over padded 100-frame chunks.
pub fn encoder_output_frames(n_valid: usize, n_frames: usize) -> usize {
    let n_chunks = n_frames / CHUNK_FRAMES;
    (0..n_chunks)
        .map(|i| {
            let remaining = n_valid.saturating_sub(i * CHUNK_FRAMES);
            post_frame_count(remaining.min(CHUNK_FRAMES))
        })
        .sum()
}

/// ISO codes Hima's prompt table knows, matching `hima_onnx.LANGUAGE_NAMES`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HimaLanguage {
    /// Assamese (`as`).
    As,
    /// Bengali (`bn`).
    Bn,
    /// English (`en`).
    En,
    /// Gujarati (`gu`).
    Gu,
    /// Hindi (`hi`).
    Hi,
    /// Kannada (`kn`).
    Kn,
    /// Malayalam (`ml`).
    Ml,
    /// Marathi (`mr`).
    Mr,
    /// Odia (`or`).
    Or,
    /// Punjabi (`pa`).
    Pa,
    /// Tamil (`ta`).
    Ta,
    /// Telugu (`te`).
    Te,
}

impl HimaLanguage {
    /// Every supported code, stable order.
    pub const ALL: [Self; 12] = [
        Self::As,
        Self::Bn,
        Self::En,
        Self::Gu,
        Self::Hi,
        Self::Kn,
        Self::Ml,
        Self::Mr,
        Self::Or,
        Self::Pa,
        Self::Ta,
        Self::Te,
    ];

    /// ISO code (`hi`).
    pub fn code(self) -> &'static str {
        match self {
            Self::As => "as",
            Self::Bn => "bn",
            Self::En => "en",
            Self::Gu => "gu",
            Self::Hi => "hi",
            Self::Kn => "kn",
            Self::Ml => "ml",
            Self::Mr => "mr",
            Self::Or => "or",
            Self::Pa => "pa",
            Self::Ta => "ta",
            Self::Te => "te",
        }
    }

    /// Lowercase English name used in the `"<name style>"` prompt.
    pub fn english_name(self) -> &'static str {
        match self {
            Self::As => "assamese",
            Self::Bn => "bengali",
            Self::En => "english",
            Self::Gu => "gujarati",
            Self::Hi => "hindi",
            Self::Kn => "kannada",
            Self::Ml => "malayalam",
            Self::Mr => "marathi",
            Self::Or => "odia",
            Self::Pa => "punjabi",
            Self::Ta => "tamil",
            Self::Te => "telugu",
        }
    }

    /// Parse a lowercase ISO code.
    pub fn parse(code: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|lang| lang.code() == code)
    }
}

/// Writer style: only the prompt string changes (`native` vs `mixed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HimaStyle {
    /// `"<language native>"`.
    Native,
    /// `"<language mixed>"`.
    Mixed,
}

impl HimaStyle {
    /// Prompt token (`native` / `mixed`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Mixed => "mixed",
        }
    }
}

/// Typed engine options. Yaml parsing is the next PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HimaConfig {
    /// `None` is auto-detect: omit the `"<language style>"` prompt tokens.
    pub language: Option<HimaLanguage>,
    /// Prompt style. The reference default is `native`.
    pub style: HimaStyle,
}

impl Default for HimaConfig {
    fn default() -> Self {
        Self {
            language: None,
            style: HimaStyle::Native,
        }
    }
}

/// `"<english native>"` prompt the reference encodes when language is set.
pub fn language_prompt(language: HimaLanguage, style: HimaStyle) -> String {
    format!("<{} {}>", language.english_name(), style.as_str())
}

/// `hima_onnx.clean_text`: NFC, Malayalam chillu folding, strip tags, lowercase.
pub fn clean_text(text: &str) -> String {
    let nfc: String = text.nfc().collect();
    let folded = fold_chillu(&nfc);
    let stripped = strip_tags(&folded);
    let chars: Vec<char> = stripped.chars().collect();
    let mut out = String::new();
    for (i, ch) in chars.iter().copied().enumerate() {
        if JOINERS.contains(&ch) {
            out.push(ch);
        } else if (ch == '\'' || ch == '\u{2019}')
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_alphabetic()
            && chars[i + 1].is_alphabetic()
        {
            out.push('\'');
        } else if ch.is_letter() || ch.is_mark() || ch.is_number() {
            out.push(ch);
        } else {
            out.push(' ');
        }
    }
    out.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn fold_chillu(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        let prev_virama = i > 0 && chars[i - 1] == VIRAMA;
        if let Some(mapped) = chillu_of(ch) {
            if !prev_virama && i + 1 < chars.len() && chars[i + 1] == VIRAMA {
                let mut next = i + 2;
                if next < chars.len() && chars[next] == ZWJ {
                    next += 1;
                }
                let blocked = chars
                    .get(next)
                    .copied()
                    .is_some_and(|c| (MALAYALAM_START..=MALAYALAM_END).contains(&c));
                if !blocked {
                    out.push(mapped);
                    i = next;
                    continue;
                }
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn chillu_of(ch: char) -> Option<char> {
    match ch {
        'ണ' => Some('ൺ'),
        'ന' => Some('ൻ'),
        'ര' => Some('ർ'),
        'ല' => Some('ൽ'),
        'ള' => Some('ൾ'),
        'ക' => Some('ൿ'),
        _ => None,
    }
}

fn strip_tags(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let open = chars[i];
        let close = match open {
            '<' => Some('>'),
            '[' => Some(']'),
            _ => None,
        };
        if let Some(close) = close {
            if let Some(rel) = chars[i + 1..].iter().position(|c| *c == close) {
                out.push(' ');
                i += rel + 2;
                continue;
            }
        }
        out.push(open);
        i += 1;
    }
    out
}

/// Repeat-unit size at the end of `p`, or 0. Matches `hima_onnx._loop_length`.
pub fn loop_length(p: &[i64]) -> usize {
    for size in 1..9 {
        let times = if size == 1 { 6 } else { 4 };
        if p.len() >= size * times && ends_with_repeat(p, size, times) {
            return size;
        }
    }
    for size in 9..81 {
        if p.len() >= size * 3 && ends_with_repeat(p, size, 3) {
            return size;
        }
    }
    0
}

fn ends_with_repeat(p: &[i64], size: usize, times: usize) -> bool {
    let start = p.len() - size * times;
    let unit = &p[p.len() - size..];
    p[start..]
        .chunks_exact(size)
        .take(times)
        .all(|chunk| chunk == unit)
}

/// Trim trailing repeats, matching `hima_onnx._cut_loop`.
pub fn cut_loop(mut p: Vec<i64>) -> Vec<i64> {
    let size = loop_length(&p);
    if size == 0 {
        return p;
    }
    let unit = p[p.len() - size..].to_vec();
    while p.len() >= 2 * size && p[p.len() - 2 * size..p.len() - size] == unit {
        p.truncate(p.len() - size);
    }
    p
}

/// Split `samples` into pieces of at most [`MAX_CHUNK_SAMPLES`], cutting at
/// the quietest 20 ms energy frame in the last two seconds of each 30 s window.
pub fn split_long_audio(samples: &[f32]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let remain = samples.len() - start;
        if remain <= MAX_CHUNK_SAMPLES {
            ranges.push(start..samples.len());
            break;
        }
        let hard = start + MAX_CHUNK_SAMPLES;
        let search_from = hard.saturating_sub(SPLIT_WINDOW).max(start + 1);
        let split = quietest_offset(&samples[search_from..hard])
            .map(|offset| search_from + offset)
            .filter(|split| *split > start)
            .unwrap_or(hard);
        ranges.push(start..split);
        start = split;
    }
    ranges
}

fn quietest_offset(window: &[f32]) -> Option<usize> {
    if window.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_e = f32::INFINITY;
    let mut i = 0;
    while i < window.len() {
        let end = (i + ENERGY_FRAME).min(window.len());
        if end - i < ENERGY_FRAME / 2 {
            break;
        }
        let energy: f32 = window[i..end].iter().map(|s| s * s).sum();
        if energy < best_e {
            best_e = energy;
            best_i = i;
        }
        i += ENERGY_FRAME;
    }
    Some(best_i)
}

/// Graphs plus tables loaded from a model directory (or built by tests).
pub(crate) struct HimaWeights {
    embed: Vec<u16>,
    embed_rows: usize,
    keep_ids: Vec<i64>,
    sot: Vec<f32>,
    filters: Vec<f32>,
    tokenizer: Tokenizer,
}

/// In-process Hima-ASR adapter (batch finalize). Sessions are injected as
/// [`OnnxSession`] so coverage tests can script inference without weights.
pub struct HimaStt {
    encoder: Box<dyn OnnxSession>,
    decoder: Box<dyn OnnxSession>,
    /// Packed f16 rows, length `embed_rows * EMBED_DIM`.
    embed: Vec<u16>,
    embed_rows: usize,
    keep_ids: Vec<i64>,
    row: HashMap<i64, usize>,
    sot: Vec<f32>,
    filters: Vec<f32>,
    tokenizer: Tokenizer,
    config: HimaConfig,
}

impl HimaStt {
    /// Load int8 encoder/decoder plus shared npy/json from `dir`.
    ///
    /// Graphs may sit in `dir` or `dir/onnx/`. Shared files (`embed.npy`,
    /// `keep_ids.json`, `start_of_text.npy`, `mel_filters.npy`, `tokenizer.json`)
    /// are resolved the way `hima_onnx._find` walks parents. `.onnx_data`
    /// must sit next to each `.onnx`.
    ///
    /// The pinned int8 graphs use `MatMulNBits` with `bits=8`. CPU support for
    /// that landed in ONNX Runtime 1.22; workspace `ort` 2.0.0-rc.9 still
    /// vendors 1.20 (4-bit only), so `OrtSession::load` fails until that
    /// crate is bumped. Mock tests never load the graphs. Coverage loads the
    /// checked-in Relu fixture under the int8 filenames so `from_dir` still
    /// runs `OrtSession::load` without the real weights.
    pub fn from_dir(dir: &Path, config: HimaConfig) -> Result<Self> {
        let encoder_path = find_graph(dir, ENCODER_FILE)?;
        let decoder_path = find_graph(dir, DECODER_FILE)?;
        let encoder = load_session(&encoder_path, "hima encoder")?;
        let decoder = load_session(&decoder_path, "hima decoder")?;
        let embed_path = find_shared(dir, "embed.npy")?;
        let packed = npy::read_f16_packed(&std::fs::read(&embed_path)?)?;
        if packed.shape.len() != 2 || packed.shape[1] != EMBED_DIM {
            return Err(provider(format!(
                "embed.npy shape {:?} is not (N, {EMBED_DIM})",
                packed.shape
            )));
        }
        let keep_path = find_shared(dir, "keep_ids.json")?;
        let keep_ids: Vec<i64> = serde_json::from_slice(&std::fs::read(&keep_path)?)
            .map_err(|err| provider(format!("keep_ids.json: {err}")))?;
        let sot_path = find_shared(dir, "start_of_text.npy")?;
        let sot = npy::read_f32(&std::fs::read(&sot_path)?)?;
        if sot.shape != [EMBED_DIM] {
            return Err(provider(format!(
                "start_of_text.npy shape {:?} is not ({EMBED_DIM},)",
                sot.shape
            )));
        }
        let filters = load_mel_filters(&find_shared(dir, "mel_filters.npy")?)?;
        let tok_path = find_shared(dir, "tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|err| {
            provider(format!(
                "could not load tokenizer {}: {err}",
                tok_path.display()
            ))
        })?;
        Self::from_parts(
            encoder,
            decoder,
            HimaWeights {
                embed: packed.data,
                embed_rows: packed.shape[0],
                keep_ids,
                sot: sot.data,
                filters,
                tokenizer,
            },
            config,
        )
    }

    /// Assemble an adapter from injected sessions. Production passes
    /// [`OrtSession`]; coverage tests pass scripted mocks.
    pub(crate) fn from_parts(
        encoder: Box<dyn OnnxSession>,
        decoder: Box<dyn OnnxSession>,
        weights: HimaWeights,
        config: HimaConfig,
    ) -> Result<Self> {
        let HimaWeights {
            embed,
            embed_rows,
            keep_ids,
            sot,
            filters,
            tokenizer,
        } = weights;
        validate_contract(encoder.as_ref(), decoder.as_ref())?;
        if embed_rows == 0 || embed.len() != embed_rows * EMBED_DIM {
            return Err(provider(format!(
                "embed is {} f16 values, expected {embed_rows}×{EMBED_DIM}",
                embed.len()
            )));
        }
        if keep_ids.len() != embed_rows {
            return Err(provider(format!(
                "keep_ids has {} rows, embed has {embed_rows}",
                keep_ids.len()
            )));
        }
        if sot.len() != EMBED_DIM {
            return Err(provider("start_of_text is not 640 f32 values"));
        }
        if filters.len() != N_FREQ * N_MELS {
            return Err(provider("mel filterbank has the wrong length"));
        }
        let mut row = HashMap::with_capacity(keep_ids.len());
        for (i, token) in keep_ids.iter().copied().enumerate() {
            row.insert(token, i);
        }
        if !row.contains_key(&BOS) || !row.contains_key(&EOS) {
            return Err(provider("keep_ids must include BOS=2 and EOS=1"));
        }
        Ok(Self {
            encoder,
            decoder,
            embed,
            embed_rows,
            keep_ids,
            row,
            sot,
            filters,
            tokenizer,
            config,
        })
    }

    /// Configured language (`None` = auto-detect).
    pub fn language(&self) -> Option<HimaLanguage> {
        self.config.language
    }

    /// Prompt style.
    pub fn style(&self) -> HimaStyle {
        self.config.style
    }

    /// Replace language/style without reloading weights.
    pub fn with_config(mut self, config: HimaConfig) -> Self {
        self.config = config;
        self
    }

    fn transcribe_pcm(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mut parts = Vec::new();
        for range in split_long_audio(pcm) {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            parts.push(self.decode_chunk(&pcm[range], cancel)?);
        }
        Ok(parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn decode_chunk(&mut self, pcm: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let mel = log_mel(pcm, &self.filters)?;
        let n = encoder_output_frames(mel.n_valid, mel.n_frames);
        let mel_tensor = OnnxTensor::f32(vec![1, N_MELS as i64, mel.n_frames as i64], mel.values);
        let n_frames = OnnxTensor::i64(Vec::new(), vec![n as i64]);
        let sound_out = self
            .encoder
            .run(&[("mel", &mel_tensor), ("n_frames", &n_frames)], &["sound"])?;
        let sound = sound_out
            .into_iter()
            .next()
            .ok_or_else(|| provider("encoder returned no sound"))?;
        let sound_data = sound
            .f32_data()
            .ok_or_else(|| provider("encoder sound is not f32"))?;
        if sound_data.len() % EMBED_DIM != 0 {
            return Err(provider(format!(
                "encoder sound length {} is not a multiple of {EMBED_DIM}",
                sound_data.len()
            )));
        }
        let rows = sound_data.len() / EMBED_DIM;
        if rows < n {
            return Err(provider(format!(
                "encoder sound has {rows} rows, expected at least {n}"
            )));
        }
        let sound_rows = &sound_data[..n * EMBED_DIM];
        self.write(sound_rows, n, cancel)
    }

    fn write(&mut self, sound: &[f32], n_sound: usize, cancel: &Cancel) -> Result<String> {
        let mut x_data = Vec::new();
        x_data.extend(self.embed_token(BOS)?);
        x_data.extend_from_slice(sound);
        x_data.extend_from_slice(&self.sot);
        if self.config.language.is_some() {
            for id in self.prompt_token_ids()? {
                x_data.extend(self.embed_token(id)?);
            }
        }
        let mut x = OnnxTensor::f32(
            vec![(x_data.len() / EMBED_DIM) as i64, EMBED_DIM as i64],
            x_data,
        );
        let mut pk = empty_kv();
        let mut pv = empty_kv();
        let mut pos: i64 = 0;
        let mut out: Vec<i64> = Vec::new();
        let limit = MAX_PIECES.min((1.5 * n_sound as f64) as usize + 8);
        for _ in 0..=limit {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let pos0 = OnnxTensor::i64(Vec::new(), vec![pos]);
            let step = self.decoder.run(
                &[("x", &x), ("pos0", &pos0), ("past_k", &pk), ("past_v", &pv)],
                &["logits", "present_k", "present_v"],
            )?;
            if step.len() < 3 {
                return Err(provider("decoder returned too few outputs"));
            }
            let logits = &step[0];
            pk = step[1].clone();
            pv = step[2].clone();
            let seq = match &x.shape[..] {
                [seq, _] => *seq,
                _ => {
                    return Err(provider("decoder x is not [T, 640]"));
                }
            };
            pos += seq;
            let j = argmax(logits)?;
            let j = usize::try_from(j).map_err(|_| provider("argmax overflow"))?;
            let tok = *self
                .keep_ids
                .get(j)
                .ok_or_else(|| provider(format!("keep_ids missing row {j}")))?;
            if tok == EOS {
                break;
            }
            out.push(tok);
            if out.len() >= limit {
                break;
            }
            if loop_length(&out) > 0 {
                out = cut_loop(out);
                break;
            }
            let row = self.embed_row(j)?;
            x = OnnxTensor::f32(vec![1, EMBED_DIM as i64], row);
        }
        let ids: Vec<u32> = out
            .iter()
            .filter_map(|id| u32::try_from(*id).ok())
            .collect();
        let raw = self
            .tokenizer
            .decode(&ids, false)
            .map_err(|err| provider(format!("tokenizer decode: {err}")))?;
        Ok(clean_text(&raw))
    }

    fn prompt_token_ids(&self) -> Result<Vec<i64>> {
        let Some(language) = self.config.language else {
            return Ok(Vec::new());
        };
        let text = language_prompt(language, self.config.style);
        let encoding = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|err| provider(format!("tokenizer encode {text:?}: {err}")))?;
        encoding
            .get_ids()
            .iter()
            .map(|&id| {
                let token = i64::from(id);
                if !self.row.contains_key(&token) {
                    return Err(provider(format!("prompt token {token} is not in keep_ids")));
                }
                Ok(token)
            })
            .collect()
    }

    fn embed_token(&self, token: i64) -> Result<Vec<f32>> {
        let row = *self
            .row
            .get(&token)
            .ok_or_else(|| provider(format!("token {token} is not in keep_ids")))?;
        self.embed_row(row)
    }

    fn embed_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.embed_rows {
            return Err(provider(format!("embed row {row} is out of range")));
        }
        let start = row * EMBED_DIM;
        Ok(self.embed[start..start + EMBED_DIM]
            .iter()
            .copied()
            .map(npy::f16_to_f32)
            .collect())
    }

    /// Prompt embedding rows for tests: BOS + `n_sound` dummy sound + SOT + language.
    #[cfg(test)]
    pub(crate) fn prompt_width(&self, n_sound: usize) -> Result<usize> {
        let extra = self.prompt_token_ids()?.len();
        Ok(1 + n_sound + 1 + extra)
    }
}

fn load_session(path: &Path, label: &str) -> Result<Box<dyn OnnxSession>> {
    OrtSession::load(path, PROVIDER_NAME, label).map(|session| Box::new(session) as _)
}

fn find_graph(dir: &Path, name: &str) -> Result<PathBuf> {
    let direct = dir.join(name);
    if direct.is_file() {
        return Ok(direct);
    }
    let nested = dir.join("onnx").join(name);
    if nested.is_file() {
        return Ok(nested);
    }
    Err(provider(format!(
        "{name} not found in {} or {}/onnx",
        dir.display(),
        dir.display()
    )))
}

fn find_shared(dir: &Path, name: &str) -> Result<PathBuf> {
    let mut d = dir.to_path_buf();
    for _ in 0..3 {
        let candidate = d.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        match d.parent() {
            Some(parent) => d = parent.to_path_buf(),
            None => break,
        }
    }
    Err(provider(format!("{name} not found near {}", dir.display())))
}

fn empty_kv() -> OnnxTensor {
    OnnxTensor::f32(vec![KV_LAYERS as i64, 1, 0, KV_DIM as i64], Vec::new())
}

fn argmax(logits: &OnnxTensor) -> Result<i64> {
    match &logits.data {
        OnnxData::F32(data) if !data.is_empty() => {
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

fn validate_contract(encoder: &dyn OnnxSession, decoder: &dyn OnnxSession) -> Result<()> {
    if encoder.input_names() != ["mel", "n_frames"] {
        return Err(provider(format!(
            "unexpected encoder inputs: {}",
            encoder.input_names().join(", ")
        )));
    }
    if !encoder.output_names().iter().any(|name| name == "sound") {
        return Err(provider("encoder must emit sound"));
    }
    if decoder.input_names() != ["x", "pos0", "past_k", "past_v"] {
        return Err(provider(format!(
            "unexpected decoder inputs: {}",
            decoder.input_names().join(", ")
        )));
    }
    let outs = decoder.output_names();
    if outs.len() < 3 || outs[0] != "logits" || outs[1] != "present_k" || outs[2] != "present_v" {
        return Err(provider("decoder must emit logits, present_k, present_v"));
    }
    Ok(())
}

impl Stt for HimaStt {
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
        let language = self
            .config
            .language
            .map(HimaLanguage::code)
            .unwrap_or("auto")
            .to_string();
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language,
        })
    }

    fn supports_partials(&self) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "hima_tests.rs"]
mod hima_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use syllabix_native::audio::{i16_to_f32, read_wav};

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hima")
    }

    fn jfk_wav() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../syllabix-core/tests/fixtures/stt/jfk.wav")
    }

    fn jfk_f32() -> Vec<f32> {
        let bytes = std::fs::read(jfk_wav()).expect("jfk.wav");
        let wav = read_wav(std::io::Cursor::new(bytes)).expect("wav");
        assert_eq!(wav.format.sample_rate_hz, SAMPLE_RATE_HZ);
        assert_eq!(wav.format.channels, 1);
        i16_to_f32(&wav.samples)
    }

    fn filters() -> Vec<f32> {
        load_mel_filters(&fixture_dir().join("mel_filters.npy")).expect("mel_filters")
    }

    #[test]
    fn post_frame_count_matches_python() {
        assert_eq!(post_frame_count(0), 0);
        assert_eq!(post_frame_count(1), 1);
        assert_eq!(post_frame_count(50), 7);
        assert_eq!(post_frame_count(100), 13);
        assert_eq!(post_frame_count(200), 25);
        assert_eq!(post_frame_count(201), 26);
    }

    #[test]
    fn short_clip_is_padded_inside_log_mel() {
        let audio = jfk_f32();
        let clip: Vec<f32> = audio
            .iter()
            .copied()
            .take((0.3 * SAMPLE_RATE_HZ as f32) as usize)
            .collect();
        assert_eq!(clip.len(), 4800);
        let mel = log_mel(&clip, &filters()).expect("short");
        assert_eq!(mel.n_mels, 128);
        assert_eq!(mel.n_valid, 50);
        assert_eq!(mel.n_frames, 100);
        assert_eq!(mel.values.len(), 128 * 100);
    }

    #[test]
    fn first_two_seconds_match_python_golden() {
        // rustfft f64 vs numpy.fft is well under this; 1e-4 still flags a
        // wrong hop, window, log floor, or filter layout.
        const TOL: f32 = 1e-4;
        let audio = jfk_f32();
        let clip = &audio[..2 * SAMPLE_RATE_HZ as usize];
        let mel = log_mel(clip, &filters()).expect("log_mel");
        assert_eq!(mel.n_mels, 128);
        assert_eq!(mel.n_valid, 200);
        assert_eq!(mel.n_frames, 200);

        let golden_bytes =
            std::fs::read(fixture_dir().join("jfk_first2s_logmel.npy")).expect("golden npy");
        let golden = npy::read_f32(&golden_bytes).expect("golden parse");
        assert_eq!(golden.shape, [128, 200]);
        assert_eq!(mel.values.len(), golden.data.len());
        let mut max_abs = 0.0f32;
        for (a, b) in mel.values.iter().zip(golden.data.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(max_abs < TOL, "max abs diff {max_abs} exceeds {TOL}");
    }

    #[test]
    fn load_mel_filters_rejects_wrong_shape() {
        let data = 1f32.to_le_bytes();
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n";
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        let len = u16::try_from(dict.len()).unwrap();
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(dict.as_bytes());
        bytes.extend_from_slice(&data);
        let err = load_mel_filters_bytes(&bytes).expect_err("shape");
        assert!(err.to_string().contains("mel_filters.npy shape"), "{err}");
    }

    #[test]
    fn log_mel_rejects_wrong_filter_len() {
        let err = log_mel(&[0.0; MIN_SAMPLES], &[0.0; 4]).expect_err("filters");
        assert!(err.to_string().contains("filterbank"), "{err}");
    }

    #[test]
    fn missing_filter_file_is_io() {
        let err = load_mel_filters(std::path::Path::new("/no/such/mel_filters.npy"))
            .expect_err("missing");
        assert!(matches!(err, Error::Io(_)), "{err}");
    }

    #[test]
    fn provider_name_is_hima() {
        assert_eq!(PROVIDER_NAME, "hima");
        assert_eq!(N_FREQ, 201);
    }

    #[test]
    fn encoder_output_frames_matches_python() {
        assert_eq!(encoder_output_frames(50, 100), 7);
        assert_eq!(encoder_output_frames(100, 100), 13);
        assert_eq!(encoder_output_frames(200, 200), 26);
        assert_eq!(
            encoder_output_frames(201, 300),
            post_frame_count(100) + post_frame_count(100) + post_frame_count(1)
        );
    }

    #[test]
    fn language_prompt_matches_reference() {
        assert_eq!(
            language_prompt(HimaLanguage::Hi, HimaStyle::Native),
            "<hindi native>"
        );
        assert_eq!(
            language_prompt(HimaLanguage::En, HimaStyle::Mixed),
            "<english mixed>"
        );
        assert_eq!(HimaLanguage::parse("or"), Some(HimaLanguage::Or));
        assert_eq!(HimaLanguage::parse("xx"), None);
        assert_eq!(HimaLanguage::ALL.len(), 12);
        assert_eq!(HimaConfig::default().language, None);
        assert_eq!(HimaConfig::default().style, HimaStyle::Native);
        for lang in HimaLanguage::ALL {
            assert_eq!(HimaLanguage::parse(lang.code()), Some(lang));
            assert!(!lang.english_name().is_empty());
            let _ = language_prompt(lang, HimaStyle::Mixed);
            let _ = language_prompt(lang, HimaStyle::Native);
        }
        assert_eq!(HimaStyle::Mixed.as_str(), "mixed");
        assert_eq!(HimaStyle::Native.as_str(), "native");
    }

    #[test]
    fn clean_text_nfc_chillu_tags_and_case() {
        assert_eq!(clean_text("Hello, WORLD!"), "hello world");
        assert_eq!(clean_text("don't"), "don't");
        assert_eq!(clean_text("don’t"), "don't");
        assert_eq!(clean_text("<lang>hi</lang> [noise] there"), "hi there");
        assert_eq!(clean_text("keep [unclosed"), "keep unclosed");
        // Preceding virama blocks chillu folding.
        assert_eq!(clean_text("്ന്"), "്ന്");
        // Malayalam: ന + virama at end of word folds to chillu ൻ.
        assert_eq!(clean_text("അവന്"), "അവൻ");
        assert_eq!(clean_text("അവന്‍"), "അവൻ");
        // Conjuncts (virama + Malayalam letter) are left alone.
        let conjunct = format!("ന്{}", "മ");
        assert_eq!(clean_text(&conjunct), conjunct);
        let nfc = "e\u{0301}";
        assert_eq!(clean_text(nfc), "é");
        assert_eq!(clean_text(""), "");
        assert_eq!(clean_text("a\u{200c}b"), "a\u{200c}b");
        assert_eq!(clean_text("ണ്"), "ൺ");
        assert_eq!(clean_text("ര്"), "ർ");
        assert_eq!(clean_text("ല്"), "ൽ");
        assert_eq!(clean_text("ള്"), "ൾ");
        assert_eq!(clean_text("ക്"), "ൿ");
    }

    #[test]
    fn loop_guards_match_python() {
        assert_eq!(loop_length(&[1, 2, 3]), 0);
        assert_eq!(loop_length(&[7, 7, 7, 7, 7, 7]), 1);
        assert_eq!(loop_length(&[7, 7, 7, 7, 7]), 0);
        assert_eq!(loop_length(&[1, 2, 1, 2, 1, 2, 1, 2]), 2);
        let long: Vec<i64> = (0..9).cycle().take(27).collect();
        assert_eq!(loop_length(&long), 9);
        assert_eq!(cut_loop(vec![7, 7, 7, 7, 7, 7]), vec![7]);
        assert_eq!(cut_loop(vec![1, 2, 1, 2, 1, 2, 1, 2]), vec![1, 2]);
        assert_eq!(cut_loop(vec![3, 4, 5]), vec![3, 4, 5]);
    }

    #[test]
    fn split_long_audio_prefers_a_quiet_point() {
        let mut samples = vec![0.5f32; MAX_CHUNK_SAMPLES + SAMPLE_RATE_HZ as usize];
        let quiet = 28 * SAMPLE_RATE_HZ as usize + SAMPLE_RATE_HZ as usize / 2;
        for sample in samples.iter_mut().skip(quiet).take(ENERGY_FRAME) {
            *sample = 0.0;
        }
        let ranges = split_long_audio(&samples);
        assert_eq!(ranges.len(), 2);
        assert!(
            ranges[0].end < MAX_CHUNK_SAMPLES,
            "should not hard-cut at 30 s when a quieter frame exists, got {}",
            ranges[0].end
        );
        assert_eq!(ranges[0].end, quiet);
        assert_eq!(ranges[1].start, quiet);
        assert_eq!(split_long_audio(&[0.1; 1000]).len(), 1);
        let flat = vec![0.5f32; MAX_CHUNK_SAMPLES + 160];
        let hard = split_long_audio(&flat);
        assert_eq!(hard.len(), 2);
        // Equal energy: the first 20 ms frame of the last two seconds wins.
        assert_eq!(hard[0].end, MAX_CHUNK_SAMPLES - SPLIT_WINDOW);
        assert_eq!(quietest_offset(&[]), None);
    }

    #[test]
    fn split_long_audio_tiles_without_gaps_or_duplicates() {
        for len in [
            0,
            1,
            MAX_CHUNK_SAMPLES,
            MAX_CHUNK_SAMPLES + 1,
            MAX_CHUNK_SAMPLES * 2 + 123,
        ] {
            let samples = vec![0.1f32; len];
            let ranges = split_long_audio(&samples);
            if len == 0 {
                assert!(ranges.is_empty());
                continue;
            }
            assert_eq!(ranges.first().map(|r| r.start), Some(0));
            assert_eq!(ranges.last().map(|r| r.end), Some(len));
            assert!(ranges.windows(2).all(|w| w[0].end == w[1].start));
            assert!(ranges.iter().all(|r| r.len() <= MAX_CHUNK_SAMPLES));
        }
    }
}
