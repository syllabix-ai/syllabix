//! Hima-ASR audio frontend: log-mel features matching `hima_onnx._log_mel`.
//!
//! Pinned to Prahaladha/hima-asr-ONNX revision
//! `18fa854f8ecaf2f48fe7416035643f86da3834d8`. Weights, config, and engine
//! wiring are later PRs.

use std::path::Path;

use crate::npy;
use crate::stft::power_spectrogram;
use syllabix_native::error::{Error, Result};

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
}
