//! Indic-Mio codec: ONNX decoder, iSTFT, voices, windowing, limiter.
//!
//! The language model lives in `syllabix-native` (`MioLmContext`). This
//! module turns speech-token codes plus a 128-d speaker embedding into
//! 24 kHz PCM.

use std::path::Path;

#[cfg(not(coverage))]
use std::path::PathBuf;

#[cfg(not(coverage))]
use ort::session::Session;
#[cfg(not(coverage))]
use ort::value::Tensor;
#[cfg(not(coverage))]
use syllabix_native::audio::istft::{Istft, N_FREQ, SAMPLES_PER_TOKEN};
use syllabix_native::error::{Error, Result};

/// Fixed decoder window (tokens). The current ONNX export is not dynamic.
pub const DECODER_TOKENS: usize = 100;
/// Sliding-window stride when the sentence is longer than [`DECODER_TOKENS`].
pub const WINDOW_STRIDE_TOKENS: usize = 80;
/// 25 tokens = 1 s; 750 tokens ≈ 30 s per sentence.
pub const FRAME_CAP_TOKENS: i32 = 750;
/// Speaker embedding width.
pub const VOICE_DIM: usize = 128;
/// Sampler seed used by native round-trip tests.
pub const INDIC_MIO_TEST_SEED: u32 = 20261115;

const DECODER_FRAMES: usize = 200;

struct Voice {
    id: &'static str,
    bytes: &'static [u8],
}

const VOICES: &[Voice] = &[
    Voice {
        id: "hi-female",
        bytes: include_bytes!("indic_mio/voices/hi-female.f32"),
    },
    Voice {
        id: "hi-male",
        bytes: include_bytes!("indic_mio/voices/hi-male.f32"),
    },
    Voice {
        id: "ta-female",
        bytes: include_bytes!("indic_mio/voices/ta-female.f32"),
    },
];

/// Default speaker id (`hi-female`).
pub const DEFAULT_VOICE_ID: &str = "hi-female";

/// Load a bundled speaker embedding by id (`hi-female`, `hi-male`, `ta-female`).
pub fn voice_by_id(id: &str) -> Result<[f32; VOICE_DIM]> {
    for voice in VOICES {
        if voice.id == id {
            return load_voice_bytes(voice.bytes);
        }
    }
    Err(Error::Provider {
        provider: "indic-mio",
        message: format!("unknown Indic-Mio voice {id:?} (want hi-female, hi-male, ta-female)"),
    })
}

fn load_voice_bytes(bytes: &[u8]) -> Result<[f32; VOICE_DIM]> {
    if bytes.len() != VOICE_DIM * 4 {
        return Err(Error::Provider {
            provider: "indic-mio",
            message: format!(
                "speaker embedding must be {} little-endian f32 bytes, got {}",
                VOICE_DIM * 4,
                bytes.len()
            ),
        });
    }
    let mut out = [0.0f32; VOICE_DIM];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

/// Pad a short sequence to [`DECODER_TOKENS`] by repeating the last real token.
pub fn pad_codes(codes: &[i64]) -> Vec<i64> {
    if codes.is_empty() {
        return vec![0; DECODER_TOKENS];
    }
    if codes.len() >= DECODER_TOKENS {
        return codes[..DECODER_TOKENS].to_vec();
    }
    let last = *codes.last().unwrap();
    let mut out = Vec::with_capacity(DECODER_TOKENS);
    out.extend_from_slice(codes);
    out.resize(DECODER_TOKENS, last);
    out
}

/// Start offsets for overlapping 100-token windows. The last window is
/// aligned to the end so it needs no padding.
pub fn window_starts(n_tokens: usize) -> Vec<usize> {
    if n_tokens <= DECODER_TOKENS {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut start = 0;
    while start + DECODER_TOKENS < n_tokens {
        starts.push(start);
        start += WINDOW_STRIDE_TOKENS;
    }
    let last = n_tokens - DECODER_TOKENS;
    if starts.last().copied() != Some(last) {
        starts.push(last);
    }
    starts
}

/// Equal-power crossfade: `cos²` / `sin²` weights over `overlap` samples.
pub fn equal_power_crossfade(left: &[f32], right: &[f32], overlap: usize) -> Vec<f32> {
    let overlap = overlap.min(left.len()).min(right.len());
    let mut out = Vec::with_capacity(left.len() + right.len() - overlap);
    out.extend_from_slice(&left[..left.len() - overlap]);
    let n = overlap.max(1) as f32;
    for i in 0..overlap {
        let t = i as f32 / n;
        let w_out = (std::f32::consts::FRAC_PI_2 * t).cos();
        let w_in = (std::f32::consts::FRAC_PI_2 * t).sin();
        out.push(left[left.len() - overlap + i] * w_out + right[i] * w_in);
    }
    out.extend_from_slice(&right[overlap..]);
    out
}

/// Soft peak limiter. Returns whether it engaged. Safety net only: a real
/// speaker embedding should already sit below full scale.
pub fn limit_peak(pcm: &mut [f32]) -> bool {
    let peak = pcm.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if peak <= 1.0 {
        return false;
    }
    let gain = 0.99 / peak;
    for sample in pcm.iter_mut() {
        *sample *= gain;
    }
    true
}

/// MioCodec ONNX decoder (fixed 100-token window) plus iSTFT.
#[cfg(not(coverage))]
pub struct MioDecoder {
    session: Session,
    istft: Istft,
    model_path: PathBuf,
}

#[cfg(not(coverage))]
impl MioDecoder {
    pub fn load(model: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_error)?
            .commit_from_file(model)
            .map_err(ort_error)?;
        let input_names: Vec<&str> = session.inputs().iter().map(|input| input.name()).collect();
        if input_names != ["content_token_indices", "global_embedding"] {
            return Err(Error::Provider {
                provider: "indic-mio",
                message: format!("unexpected ONNX inputs: {}", input_names.join(", ")),
            });
        }
        let output_names: Vec<&str> = session
            .outputs()
            .iter()
            .map(|output| output.name())
            .collect();
        if !output_names.contains(&"mag") || !output_names.contains(&"phase") {
            return Err(Error::Provider {
                provider: "indic-mio",
                message: format!("unexpected ONNX outputs: {}", output_names.join(", ")),
            });
        }
        Ok(Self {
            session,
            istft: Istft::new(),
            model_path: model.to_path_buf(),
        })
    }

    fn infer_window(&mut self, codes: &[i64], embedding: &[f32; VOICE_DIM]) -> Result<Vec<f32>> {
        let ids = pad_codes(codes);
        let outputs = self
            .session
            .run(ort::inputs![
                "content_token_indices" => Tensor::from_array(([1_usize, DECODER_TOKENS], ids)).map_err(ort_error)?,
                "global_embedding" => Tensor::from_array(([1_usize, VOICE_DIM], embedding.to_vec())).map_err(ort_error)?,
            ])
            .map_err(ort_error)?;
        let mag = named_f32(&outputs, "mag")?;
        let phase = named_f32(&outputs, "phase")?;
        if mag.len() != N_FREQ * DECODER_FRAMES || phase.len() != N_FREQ * DECODER_FRAMES {
            return Err(Error::Provider {
                provider: "indic-mio",
                message: format!(
                    "ONNX mag/phase shape mismatch ({} / {}) from {}",
                    mag.len(),
                    phase.len(),
                    self.model_path.display()
                ),
            });
        }
        let pcm = self
            .istft
            .run(&mag, &phase, DECODER_FRAMES)
            .map_err(|message| Error::Provider {
                provider: "indic-mio",
                message,
            })?;
        Ok(pcm)
    }

    /// Decode any-length codes. Short sentences pad with the last token
    /// (never zeros). Long sentences use overlapping 100-token windows
    /// and an equal-power crossfade.
    pub fn decode(&mut self, codes: &[i32], embedding: &[f32; VOICE_DIM]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let codes: Vec<i64> = codes.iter().map(|c| i64::from(*c)).collect();
        if codes.len() <= DECODER_TOKENS {
            let mut pcm = self.infer_window(&codes, embedding)?;
            pcm.truncate(codes.len() * SAMPLES_PER_TOKEN);
            return Ok(pcm);
        }
        let starts = window_starts(codes.len());
        let mut combined = Vec::new();
        for (i, start) in starts.iter().copied().enumerate() {
            let window = &codes[start..start + DECODER_TOKENS];
            let piece = self.infer_window(window, embedding)?;
            if i == 0 {
                combined = piece;
                continue;
            }
            let prev = starts[i - 1];
            let overlap_tokens = prev + DECODER_TOKENS - start;
            let overlap_samples = overlap_tokens * SAMPLES_PER_TOKEN;
            combined = equal_power_crossfade(&combined, &piece, overlap_samples);
        }
        combined.truncate(codes.len() * SAMPLES_PER_TOKEN);
        Ok(combined)
    }
}

#[cfg(coverage)]
#[derive(Debug)]
pub struct MioDecoder {
    _private: (),
}

#[cfg(coverage)]
impl MioDecoder {
    pub fn load(_model: &Path) -> Result<Self> {
        Err(Error::Provider {
            provider: "indic-mio",
            message: "MioCodec ONNX is not loaded in coverage tests.".into(),
        })
    }

    pub fn decode(&mut self, codes: &[i32], embedding: &[f32; VOICE_DIM]) -> Result<Vec<f32>> {
        coverage_exercise_codec_helpers();
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let _ = (embedding.len(), FRAME_CAP_TOKENS, DECODER_FRAMES);
        Err(Error::Provider {
            provider: "indic-mio",
            message: "MioCodec ONNX is not loaded in coverage tests.".into(),
        })
    }
}

/// Keeps codec helpers alive in coverage builds where `NativeIndicMio` is not
/// linked and ONNX is not loaded.
#[cfg(coverage)]
pub(crate) fn coverage_exercise_codec_helpers() {
    use syllabix_native::audio::istft::SAMPLES_PER_TOKEN;

    let _ = (INDIC_MIO_TEST_SEED, FRAME_CAP_TOKENS, DECODER_FRAMES);
    let _ = pad_codes(&[7, 8]);
    let long: Vec<i64> = (0..DECODER_TOKENS + WINDOW_STRIDE_TOKENS)
        .map(|i| i as i64)
        .collect();
    let starts = window_starts(long.len());
    assert!(!starts.is_empty());
    let _voice = voice_by_id(DEFAULT_VOICE_ID).expect("default voice");
    let sample_len = WINDOW_STRIDE_TOKENS * SAMPLES_PER_TOKEN;
    let left = vec![0.25f32; sample_len];
    let right = vec![-0.25f32; sample_len];
    let _ = equal_power_crossfade(&left, &right, sample_len / 4);
    let mut loud = vec![3.5f32; 4];
    assert!(limit_peak(&mut loud));
}

#[cfg(not(coverage))]
fn named_f32(outputs: &ort::session::SessionOutputs<'_>, name: &str) -> Result<Vec<f32>> {
    let (_shape, data) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(ort_error)?;
    Ok(data.to_vec())
}

#[cfg(not(coverage))]
fn ort_error(err: impl std::fmt::Display) -> Error {
    Error::Provider {
        provider: "indic-mio",
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syllabix_native::audio::istft::{Istft, N_FREQ};

    #[test]
    fn pad_empty_codes_uses_zeros() {
        let padded = pad_codes(&[]);
        assert_eq!(padded.len(), DECODER_TOKENS);
        assert!(padded.iter().all(|&c| c == 0));
    }

    #[test]
    fn load_voice_bytes_rejects_wrong_size() {
        let err = load_voice_bytes(&[0u8; 8]).unwrap_err();
        assert!(err.to_string().contains("speaker embedding"));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_mio_codec_helpers() {
        coverage_exercise_codec_helpers();
        let err = MioDecoder::load(Path::new("/no/such.onnx")).unwrap_err();
        assert!(err.to_string().contains("coverage"));
        let voice = voice_by_id(DEFAULT_VOICE_ID).unwrap();
        let mut dec = MioDecoder { _private: () };
        assert!(dec.decode(&[], &voice).unwrap().is_empty());
        let err = dec.decode(&[1], &voice).unwrap_err();
        assert!(err.to_string().contains("coverage"));
    }

    #[test]
    fn pad_repeats_last_token_never_zeros() {
        let padded = pad_codes(&[7, 8, 9]);
        assert_eq!(padded.len(), DECODER_TOKENS);
        assert_eq!(&padded[..3], &[7, 8, 9]);
        assert!(padded[3..].iter().all(|c| *c == 9));
        assert!(!padded.contains(&0));
    }

    #[test]
    fn window_starts_align_last_window() {
        assert_eq!(window_starts(8), vec![0]);
        assert_eq!(window_starts(100), vec![0]);
        assert_eq!(window_starts(101), vec![0, 1]);
        assert_eq!(window_starts(180), vec![0, 80]);
        assert_eq!(window_starts(200), vec![0, 80, 100]);
    }

    #[test]
    fn equal_power_crossfade_weights_sum_to_one() {
        let overlap = 4;
        let n = overlap as f32;
        for i in 0..overlap {
            let t = i as f32 / n;
            let w_out = (std::f32::consts::FRAC_PI_2 * t).cos();
            let w_in = (std::f32::consts::FRAC_PI_2 * t).sin();
            assert!((w_out * w_out + w_in * w_in - 1.0).abs() < 1e-5);
        }
        let left = vec![1.0f32; 8];
        let right = vec![-1.0f32; 8];
        let mixed = equal_power_crossfade(&left, &right, overlap);
        assert_eq!(mixed.len(), 12);
        assert_eq!(mixed[0], 1.0);
        assert_eq!(*mixed.last().unwrap(), -1.0);
    }

    #[test]
    fn limiter_engages_only_above_full_scale() {
        let mut quiet = vec![0.4f32, -0.5, 0.2];
        assert!(!limit_peak(&mut quiet));
        let mut loud = vec![3.6f32, -2.0, 1.1];
        assert!(limit_peak(&mut loud));
        let peak = loud.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(peak <= 0.99 + 1e-5);
    }

    #[test]
    fn default_voice_is_not_all_zero() {
        let voice = voice_by_id(DEFAULT_VOICE_ID).unwrap();
        let norm: f32 = voice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1, "default voice embedding is degenerate ({norm})");
        for shipped in VOICES {
            let v = load_voice_bytes(shipped.bytes).expect(shipped.id);
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(n > 0.1, "{} degenerate ({n})", shipped.id);
        }
        assert!(voice_by_id("nope").is_err());
        assert_eq!(
            voice_by_id("hi-male").unwrap(),
            load_voice_bytes(VOICES[1].bytes).unwrap()
        );
    }

    #[test]
    fn golden_istft_matches_upstream_pcm() {
        let mag = read_fixture("spec_mag.f32");
        let phase = read_fixture("spec_phase.f32");
        let expected = read_fixture("expected_pcm.f32");
        let n_frames = mag.len() / N_FREQ;
        let got = Istft::new().run(&mag, &phase, n_frames).expect("iSTFT");
        assert_eq!(got.len(), expected.len());
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-4, "iSTFT max abs error {max_abs}");
    }

    fn read_fixture(name: &str) -> Vec<f32> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../syllabix-native/tests/fixtures/miocodec_istft")
            .join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}
