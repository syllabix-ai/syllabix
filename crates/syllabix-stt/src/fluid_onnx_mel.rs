//! NeMo log-mel frontend for Fluid Speech Pro ONNX (slaney mel, `realfft` STFT).

use realfft::RealFftPlanner;
use syllabix_native::error::{Error, Result};

const N_FFT: usize = 512;
const HOP: usize = 160;
const WIN_LEN: usize = 400;
pub(crate) const N_MELS: usize = 128;
pub(crate) const N_FREQ: usize = N_FFT / 2 + 1;
const PREEMPH: f32 = 0.97;
const SIDE_PAD: usize = 256;
const LN_EPS: f32 = 1.0 / 16_777_216.0;

fn mel_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "fluid-onnx-mel",
        message: message.into(),
    }
}

fn hz_to_mel_slaney(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if freq < min_log_hz {
        freq / f_sp
    } else {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// Torchaudio `melscale_fbanks` with `norm="slaney"` and `mel_scale="slaney"`.
pub fn slaney_mel_filterbank(
    n_freq: usize,
    n_mels: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    assert_eq!(n_freq, N_FREQ);
    let f_min = fmin as f64;
    let f_max = fmax as f64;
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);
    let mut mel_pts = vec![0.0f64; n_mels + 2];
    for (i, slot) in mel_pts.iter_mut().enumerate() {
        *slot = mel_min + (mel_max - mel_min) * (i as f64) / (n_mels as f64 + 1.0);
    }
    let hz_pts: Vec<f64> = mel_pts.iter().map(|&m| mel_to_hz_slaney(m)).collect();
    let mut fb = vec![0.0f32; n_mels * n_freq];
    let sr = sample_rate as f64;
    for m in 0..n_mels {
        let left = hz_pts[m];
        let center = hz_pts[m + 1];
        let right = hz_pts[m + 2];
        let enorm = 2.0 / (right - left);
        for k in 0..n_freq {
            let freq = (k as f64) * sr / (n_fft_bin_denom());
            let weight = if freq < left || freq >= right {
                0.0
            } else if freq < center {
                if center == left {
                    0.0
                } else {
                    (freq - left) / (center - left)
                }
            } else if right == center {
                0.0
            } else {
                (right - freq) / (right - center)
            };
            fb[m * n_freq + k] = (weight * enorm) as f32;
        }
    }
    fb
}

fn n_fft_bin_denom() -> f64 {
    N_FFT as f64
}

fn symmetric_hann_window() -> Vec<f32> {
    let mut hann = vec![0.0f32; WIN_LEN];
    let denom = (WIN_LEN - 1) as f32;
    for (i, slot) in hann.iter_mut().enumerate() {
        let phase = 2.0 * std::f32::consts::PI * (i as f32) / denom;
        *slot = 0.5 - 0.5 * phase.cos();
    }
    let mut win = vec![0.0f32; N_FFT];
    let offset = (N_FFT - WIN_LEN) / 2;
    win[offset..offset + WIN_LEN].copy_from_slice(&hann);
    win
}

fn preemphasis(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for i in 1..samples.len() {
        out.push(samples[i] - PREEMPH * samples[i - 1]);
    }
    out
}

fn const_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len() + 2 * pad);
    out.extend(std::iter::repeat_n(0.0f32, pad));
    out.extend_from_slice(samples);
    out.extend(std::iter::repeat_n(0.0f32, pad));
    out
}

fn power_stft(padded: &[f32]) -> Result<Vec<f32>> {
    if padded.len() < N_FFT {
        return Err(mel_err("signal shorter than n_fft after padding"));
    }
    let n_frames = 1 + (padded.len() - N_FFT) / HOP;
    let window = symmetric_hann_window();
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    let mut values = Vec::with_capacity(n_frames * N_FREQ);
    for frame in 0..n_frames {
        let start = frame * HOP;
        for i in 0..N_FFT {
            input[i] = f64::from(padded[start + i] * window[i]);
        }
        r2c.process(&mut input, &mut spectrum)
            .map_err(|err| mel_err(format!("rfft failed: {err}")))?;
        for bin in &spectrum {
            values.push((bin.re * bin.re + bin.im * bin.im) as f32);
        }
    }
    Ok(values)
}

/// 16 kHz mono f32 in; log-mel features, frame count, and valid length out.
pub fn nemo_log_mel(samples: &[f32], filters: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
    if filters.len() != N_MELS * N_FREQ {
        return Err(mel_err(format!(
            "mel filterbank is {} bins, expected {}",
            filters.len(),
            N_MELS * N_FREQ
        )));
    }
    let n = samples.len();
    let valid_len = n / HOP;
    let n_frames = 1 + n / HOP;
    let emphasized = preemphasis(samples);
    let padded = const_pad(&emphasized, SIDE_PAD);
    let spec = power_stft(&padded)?;
    if spec.len() != n_frames * N_FREQ {
        return Err(mel_err("unexpected spectrogram frame count"));
    }
    let mut mel = vec![0.0f32; N_MELS * n_frames];
    for t in 0..n_frames {
        for m in 0..N_MELS {
            let mut acc = 0.0f32;
            for k in 0..N_FREQ {
                acc += spec[t * N_FREQ + k] * filters[m * N_FREQ + k];
            }
            mel[m * n_frames + t] = (acc + LN_EPS).ln();
        }
    }
    if valid_len > 0 {
        let denom = if valid_len > 1 {
            (valid_len - 1) as f32
        } else {
            1.0
        };
        for m in 0..N_MELS {
            let row = m * n_frames;
            let slice = &mel[row..row + valid_len];
            let mean = slice.iter().sum::<f32>() / valid_len as f32;
            let var = slice
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / denom;
            let scale = (var.sqrt() + 1e-5).recip();
            for t in 0..valid_len {
                mel[row + t] = (mel[row + t] - mean) * scale;
            }
            for t in valid_len..n_frames {
                mel[row + t] = 0.0;
            }
        }
    }
    Ok((mel, n_frames, valid_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;

    const SAMPLE_RATE: u32 = 16_000;

    #[test]
    fn frame_count_and_valid_length() {
        let filters = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE, 0.0, 8000.0);
        let samples = vec![0.0f32; 16_000];
        let (_, frames, valid) = nemo_log_mel(&samples, &filters).expect("mel");
        assert_eq!(frames, 101);
        assert_eq!(valid, 100);
    }

    #[test]
    fn filterbank_shape_and_row_sums() {
        let fb = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE, 0.0, 8000.0);
        assert_eq!(fb.len(), N_MELS * N_FREQ);
        // Spot-check slaney column sums against onnx_asr `fbanks.npz` (`nemo128`).
        let expect = [0.028_377_54_f32, 0.028_377_54, 0.030_684_11, 0.031_943_69];
        for (m, want) in [0usize, 32, 64, 127].into_iter().zip(expect) {
            let col_sum: f32 = (0..N_FREQ).map(|k| fb[m * N_FREQ + k]).sum();
            assert!((col_sum - want).abs() < 1e-5, "mel col {m} sum {col_sum}");
        }
    }

    #[test]
    fn jfk_full_wav_matches_onnx_asr_reference() {
        const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/fluid/jfk_full_nemo128.f32");
        let expected: Vec<f32> = FIXTURE
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let jfk = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../syllabix-core/tests/fixtures/stt/jfk.wav"),
        )
        .expect("jfk.wav");
        let wav = syllabix_native::audio::read_wav(std::io::Cursor::new(jfk)).expect("wav");
        let pcm: Vec<f32> = wav.samples.iter().map(|s| *s as f32 / 32768.0).collect();
        let filters = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE, 0.0, 8000.0);
        let (got, frames, valid) = nemo_log_mel(&pcm, &filters).expect("mel");
        assert_eq!(valid, pcm.len() / 160);
        assert_eq!(got.len(), expected.len());
        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1e-3,
            "max diff {max_diff} frames {frames} valid {valid}"
        );
    }

    #[test]
    fn jfk_first_second_matches_onnx_asr_reference() {
        const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/fluid/jfk_1s_nemo128.f32");
        const SOURCE: &str = include_str!("../tests/fixtures/fluid/SOURCE.txt");
        assert!(SOURCE.contains("onnx_asr"));
        let expected: Vec<f32> = FIXTURE
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(expected.len(), 128 * 101);
        let jfk = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../syllabix-core/tests/fixtures/stt/jfk.wav"),
        )
        .expect("jfk.wav");
        let wav = syllabix_native::audio::read_wav(std::io::Cursor::new(jfk)).expect("wav");
        let clip: Vec<f32> = wav
            .samples
            .iter()
            .map(|s| *s as f32 / 32768.0)
            .take(16_000)
            .collect();
        let filters = slaney_mel_filterbank(N_FREQ, N_MELS, SAMPLE_RATE, 0.0, 8000.0);
        let (got, frames, valid) = nemo_log_mel(&clip, &filters).expect("mel");
        assert_eq!(frames, 101);
        assert_eq!(valid, 100);
        assert_eq!(got.len(), expected.len());
        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff <= 1e-3, "max diff {max_diff}");
        let expect_rows = [-1.5972712_f32, -2.7600634, -2.538801];
        for (row, want) in expect_rows.into_iter().enumerate() {
            for frame in 0..3 {
                let v = got[row * frames + frame];
                assert!(
                    (v - want).abs() < 1e-3,
                    "row {row} frame {frame} = {v}, expected {want}"
                );
            }
        }
    }
}
