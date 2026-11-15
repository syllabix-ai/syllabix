//! Centered real STFT power spectrum, shared by ONNX speech frontends.
//!
//! Parameterised by `n_fft`, window length, and hop. The window is a periodic
//! Hann of `window_len`, zero-padded to `n_fft` when it is shorter (NeMo-style
//! 400-sample window in a 512-point FFT). Hima-specific last-frame drop, mel
//! filter, and log scaling live in [`crate::hima`].

use realfft::RealFftPlanner;
use syllabix_native::error::{Error, Result};

fn stft_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "stft",
        message: message.into(),
    }
}

/// Power spectrogram from a centered real STFT.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSpectrogram {
    /// `n_fft / 2 + 1` rFFT bins.
    pub n_freq: usize,
    /// Number of STFT frames (no hima-specific last-frame drop).
    pub n_frames: usize,
    /// Row-major `[n_frames][n_freq]` power (`re² + im²` as f32).
    pub values: Vec<f32>,
}

/// Periodic Hann of `window_len`, zero-padded on the right to `n_fft`.
pub fn padded_periodic_hann(n_fft: usize, window_len: usize) -> Result<Vec<f32>> {
    if n_fft < 2 {
        return Err(stft_err("n_fft must be at least 2"));
    }
    if window_len == 0 || window_len > n_fft {
        return Err(stft_err(format!(
            "window length {window_len} must be in 1..=n_fft ({n_fft})"
        )));
    }
    let mut win = vec![0.0f32; n_fft];
    let denom = window_len as f32;
    for (i, slot) in win.iter_mut().take(window_len).enumerate() {
        let phase = 2.0 * std::f32::consts::PI * (i as f32) / denom;
        *slot = 0.5 - 0.5 * phase.cos();
    }
    Ok(win)
}

/// `numpy.pad(x, (pad, pad), mode="reflect")` for a 1-D buffer.
pub fn reflect_pad(samples: &[f32], pad: usize) -> Result<Vec<f32>> {
    if pad == 0 {
        return Ok(samples.to_vec());
    }
    if samples.len() < 2 {
        return Err(stft_err(
            "reflect pad needs at least two samples when pad > 0",
        ));
    }
    let n = samples.len();
    let mut out = vec![0.0; n + 2 * pad];
    for (i, slot) in out.iter_mut().enumerate() {
        let src = i as isize - pad as isize;
        *slot = samples[reflect_index(src, n)];
    }
    Ok(out)
}

fn reflect_index(index: isize, n: usize) -> usize {
    let period = (2 * (n - 1)) as isize;
    let mut k = index.rem_euclid(period);
    if k >= n as isize {
        k = period - k;
    }
    k as usize
}

/// Centered power STFT: reflect-pad `n_fft/2` on each side, hop `hop`.
///
/// FFT math is f64 (matching `numpy.fft.rfft`) and power is stored as f32.
pub fn power_spectrogram(
    samples: &[f32],
    n_fft: usize,
    window_len: usize,
    hop: usize,
) -> Result<PowerSpectrogram> {
    if hop == 0 {
        return Err(stft_err("hop must be greater than 0"));
    }
    let window = padded_periodic_hann(n_fft, window_len)?;
    let pad = n_fft / 2;
    let xp = reflect_pad(samples, pad)?;
    if xp.len() < n_fft {
        return Err(stft_err("signal is shorter than n_fft after padding"));
    }
    let n_frames = 1 + (xp.len() - n_fft) / hop;
    let n_freq = n_fft / 2 + 1;
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    let mut values = Vec::with_capacity(n_frames * n_freq);
    for frame in 0..n_frames {
        let start = frame * hop;
        for i in 0..n_fft {
            input[i] = f64::from(xp[start + i] * window[i]);
        }
        r2c.process(&mut input, &mut spectrum)
            .map_err(|err| stft_err(format!("rfft failed: {err}")))?;
        for bin in &spectrum {
            values.push((bin.re * bin.re + bin.im * bin.im) as f32);
        }
    }
    Ok(PowerSpectrogram {
        n_freq,
        n_frames,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hann_is_zero_at_both_ends_and_pads_when_short() {
        let win = padded_periodic_hann(400, 400).expect("hann");
        assert_eq!(win.len(), 400);
        assert_eq!(win[0], 0.0);
        assert!(win[200] > 0.99);
        let padded = padded_periodic_hann(512, 400).expect("nemo-style");
        assert_eq!(padded.len(), 512);
        assert_eq!(&padded[400..], &[0.0; 112]);
        assert_eq!(&padded[..400], &win[..]);
    }

    #[test]
    fn reflect_pad_matches_numpy_edges() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let padded = reflect_pad(&x, 3).expect("pad");
        assert_eq!(padded, [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn power_spectrogram_shape_follows_n_fft_and_hop() {
        let samples = vec![0.1f32; 1600];
        let spec = power_spectrogram(&samples, 400, 400, 160).expect("stft");
        assert_eq!(spec.n_freq, 201);
        // reflect pad 200+200, n_frames = 1 + (2000-400)/160 = 11
        assert_eq!(spec.n_frames, 11);
        assert_eq!(spec.values.len(), 11 * 201);

        let nemo = power_spectrogram(&samples, 512, 400, 160).expect("nemo");
        assert_eq!(nemo.n_freq, 257);
        assert!(nemo.n_frames > 0);
    }

    #[test]
    fn rejects_bad_stft_params() {
        assert!(padded_periodic_hann(1, 1).is_err());
        assert!(padded_periodic_hann(400, 0).is_err());
        assert!(padded_periodic_hann(400, 401).is_err());
        assert!(power_spectrogram(&[0.0; 8], 400, 400, 0).is_err());
        assert!(reflect_pad(&[1.0], 2).is_err());
    }
}
