//! Inverse STFT matching MioCodec `ISTFT` (`padding="same"`).
//!
//! Periodic Hann 1920, `irfft` with backward normalisation (realfft's
//! inverse is unnormalised, so divide by `n_fft`), hop 480, trim 720
//! samples on each side, then divide by the overlap-added `hann²`
//! envelope.

use super::stft::periodic_hann;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner};
use std::sync::Arc;

/// FFT size used by the 24 kHz MioCodec decoder.
pub const N_FFT: usize = 1920;
/// Hop size in samples.
pub const HOP: usize = 480;
/// `n_fft / 2 + 1`.
pub const N_FREQ: usize = N_FFT / 2 + 1;
/// `(n_fft - hop) / 2` — PyTorch `"same"` trim on each end.
pub const SAME_TRIM: usize = (N_FFT - HOP) / 2;
/// Decoder tokens produce this many 24 kHz samples each.
pub const SAMPLES_PER_TOKEN: usize = 960;

/// Inverse STFT with a reusable realfft plan.
pub struct Istft {
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    window_sq: Vec<f32>,
}

impl Default for Istft {
    fn default() -> Self {
        Self::new()
    }
}

impl Istft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let inverse = planner.plan_fft_inverse(N_FFT);
        let window = periodic_hann(N_FFT);
        let window_sq: Vec<f32> = window.iter().map(|w| w * w).collect();
        Self {
            inverse,
            window,
            window_sq,
        }
    }

    /// `mag` and `phase` are row-major `[n_freq, n_frames]` (`[961, F]`).
    /// Returns `n_frames * hop` samples after the `"same"` trim and envelope.
    pub fn run(&self, mag: &[f32], phase: &[f32], n_frames: usize) -> Result<Vec<f32>, String> {
        let expected = N_FREQ * n_frames;
        if mag.len() != expected || phase.len() != expected {
            return Err(format!(
                "iSTFT expected {expected} mag/phase bins for {n_frames} frames, got {} / {}",
                mag.len(),
                phase.len()
            ));
        }
        if n_frames == 0 {
            return Ok(Vec::new());
        }

        let ola_len = (n_frames - 1) * HOP + N_FFT;
        let mut y = vec![0.0f32; ola_len];
        let mut env = vec![0.0f32; ola_len];
        let mut spectrum = self.inverse.make_input_vec();
        let mut time = self.inverse.make_output_vec();
        let scale = 1.0 / N_FFT as f32;

        for frame in 0..n_frames {
            for (freq, bin) in spectrum.iter_mut().enumerate() {
                let idx = freq * n_frames + frame;
                let m = mag[idx];
                let p = phase[idx];
                *bin = Complex::new(m * p.cos(), m * p.sin());
            }
            // realfft requires DC and Nyquist to be purely real, matching numpy irfft.
            spectrum[0].im = 0.0;
            if let Some(nyquist) = spectrum.last_mut() {
                nyquist.im = 0.0;
            }
            self.inverse
                .process(&mut spectrum, &mut time)
                .map_err(|err| err.to_string())?;
            let offset = frame * HOP;
            for i in 0..N_FFT {
                let sample = time[i] * scale * self.window[i];
                y[offset + i] += sample;
                env[offset + i] += self.window_sq[i];
            }
        }

        let end = ola_len - SAME_TRIM;
        if end <= SAME_TRIM {
            return Err("iSTFT output shorter than the same-padding trim".into());
        }
        let mut out = Vec::with_capacity(end - SAME_TRIM);
        for i in SAME_TRIM..end {
            out.push(y[i] / env[i].max(1e-11));
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "cpp"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/miocodec_istft")
            .join(name)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "{}", path.display());
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn golden_istft_matches_upstream_pcm() {
        let mag = read_f32(&fixture("spec_mag.f32"));
        let phase = read_f32(&fixture("spec_phase.f32"));
        let expected = read_f32(&fixture("expected_pcm.f32"));
        assert_eq!(mag.len() % N_FREQ, 0);
        let n_frames = mag.len() / N_FREQ;
        let got = Istft::new().run(&mag, &phase, n_frames).expect("iSTFT");
        assert_eq!(got.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (a, b) in got.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs <= 1e-4,
            "iSTFT max abs error {max_abs} exceeds 1e-4"
        );
    }
}
