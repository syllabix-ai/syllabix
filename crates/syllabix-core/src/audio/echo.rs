//! Full-duplex acoustic echo cancellation.
//!
//! The speaker callback records the samples actually submitted to the audio
//! device. [`EchoController`] converts that render reference and microphone
//! capture to 16 kHz mono, processes synchronized 10 ms blocks through WebRTC
//! AEC3, and returns cleaned 16 kHz capture samples for ASR. Silero performs
//! its own separate 8 kHz conversion in the VAD adapter.

use std::collections::VecDeque;
use std::sync::Arc;

use sonora::config::EchoCanceller;
use sonora::{AudioProcessing, Config, StreamConfig};

use crate::audio::convert::{PcmConverter, PcmFormat};
use crate::audio::ring::SampleRing;
use crate::error::{Error, Result};
use crate::types::DEFAULT_SAMPLE_RATE_HZ;

/// WebRTC audio processing uses 10 ms blocks.
pub const AEC_FRAME_SAMPLES: usize = DEFAULT_SAMPLE_RATE_HZ as usize / 100;
const MAX_PENDING_BLOCKS: usize = 50;
const MAX_PENDING_SAMPLES: usize = AEC_FRAME_SAMPLES * MAX_PENDING_BLOCKS;
const PLAYBACK_ENERGY_FLOOR: f32 = 1.0e-7;

/// Read-only render tap shared by the speaker callback and capture worker.
pub struct EchoReference {
    pub(crate) ring: Arc<SampleRing>,
    pub(crate) device_format: PcmFormat,
}

impl EchoReference {
    pub(crate) fn new(ring: Arc<SampleRing>, device_format: PcmFormat) -> Self {
        Self {
            ring,
            device_format,
        }
    }

    /// Samples discarded because capture processing did not drain fast enough.
    pub fn dropped_samples(&self) -> u64 {
        self.ring.overruns()
    }
}

/// State of AEC's automatic render/capture delay calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoCalibration {
    /// No non-silent speaker reference has arrived yet.
    WaitingForPlayback,
    /// Speaker audio is present and AEC3 is estimating the echo path.
    Calibrating,
    /// AEC3 has produced a delay estimate.
    Active,
    /// Render-reference samples were lost; headphones are recommended.
    Degraded,
}

/// Sonora AEC3 wrapper operating on the existing 16 kHz capture/ASR stream.
pub struct EchoController {
    processor: AudioProcessing,
    reference: EchoReference,
    render_converter: PcmConverter,
    render_pending: VecDeque<f32>,
    capture_pending: VecDeque<f32>,
    clean_pending: VecDeque<f32>,
    silent_render: Vec<f32>,
    last_dropped_samples: u64,
    far_end_blocks: u64,
    calibration: EchoCalibration,
}

impl EchoController {
    /// Enable AEC3 and attach the exact speaker callback reference.
    pub fn new(reference: EchoReference) -> Result<Self> {
        let stream = StreamConfig::new(DEFAULT_SAMPLE_RATE_HZ, 1);
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        };
        let mut processor = AudioProcessing::builder()
            .config(config)
            .capture_config(stream)
            .render_config(stream)
            .echo_detector(true)
            .build();
        // The reference is tapped at the output callback, so queueing delay is
        // already excluded. AEC3 estimates the remaining acoustic-path delay.
        processor
            .set_stream_delay_ms(0)
            .map_err(|err| echo_error("set stream delay", err))?;
        let render_converter = PcmConverter::new(reference.device_format, PcmFormat::v0())?;
        let last_dropped_samples = reference.dropped_samples();
        Ok(Self {
            processor,
            reference,
            render_converter,
            render_pending: VecDeque::with_capacity(MAX_PENDING_SAMPLES),
            capture_pending: VecDeque::with_capacity(AEC_FRAME_SAMPLES * 4),
            clean_pending: VecDeque::with_capacity(AEC_FRAME_SAMPLES * 4),
            silent_render: vec![0.0; AEC_FRAME_SAMPLES],
            last_dropped_samples,
            far_end_blocks: 0,
            calibration: EchoCalibration::WaitingForPlayback,
        })
    }

    /// Process arbitrary-length 16 kHz mono capture samples.
    ///
    /// Output is emitted only in complete 10 ms blocks; callers retain their
    /// existing frame splitter for the 512-sample ASR capture contract.
    pub fn process_capture(&mut self, capture: &[f32]) -> Result<Vec<f32>> {
        self.capture_pending.extend(capture.iter().copied());
        self.drain_render_reference();

        while self.capture_pending.len() >= AEC_FRAME_SAMPLES {
            self.process_one_block()?;
        }

        Ok(self.clean_pending.drain(..).collect())
    }

    /// Current automatic calibration state.
    pub fn calibration(&self) -> EchoCalibration {
        self.calibration
    }

    /// Number of non-silent 10 ms speaker blocks observed.
    pub fn far_end_blocks(&self) -> u64 {
        self.far_end_blocks
    }

    fn drain_render_reference(&mut self) {
        let available = self.reference.ring.occupancy();
        if available == 0 {
            return;
        }
        let mut device_samples = vec![0.0; available.min(self.reference.ring.capacity())];
        let read = self.reference.ring.try_pop_slice(&mut device_samples);
        let converted = self.render_converter.push(&device_samples[..read]);
        self.render_pending.extend(converted);

        if self.render_pending.len() > MAX_PENDING_SAMPLES {
            let discard = self.render_pending.len() - MAX_PENDING_SAMPLES;
            self.render_pending.drain(..discard);
            self.calibration = EchoCalibration::Degraded;
        }
        let dropped = self.reference.dropped_samples();
        if dropped > self.last_dropped_samples {
            self.last_dropped_samples = dropped;
            self.calibration = EchoCalibration::Degraded;
        }
    }

    fn process_one_block(&mut self) -> Result<()> {
        let render = if self.render_pending.len() >= AEC_FRAME_SAMPLES {
            self.render_pending
                .drain(..AEC_FRAME_SAMPLES)
                .collect::<Vec<_>>()
        } else {
            self.silent_render.clone()
        };
        if mean_square(&render) > PLAYBACK_ENERGY_FLOOR {
            self.far_end_blocks += 1;
            if self.calibration == EchoCalibration::WaitingForPlayback {
                self.calibration = EchoCalibration::Calibrating;
            }
        }

        let mut render_output = vec![0.0; AEC_FRAME_SAMPLES];
        self.processor
            .process_render_f32(&[&render], &mut [&mut render_output])
            .map_err(|err| echo_error("process speaker reference", err))?;

        let capture = self
            .capture_pending
            .drain(..AEC_FRAME_SAMPLES)
            .collect::<Vec<_>>();
        let mut clean = vec![0.0; AEC_FRAME_SAMPLES];
        self.processor
            .process_capture_f32(&[&capture], &mut [&mut clean])
            .map_err(|err| echo_error("process microphone capture", err))?;
        self.clean_pending.extend(clean);

        if self.calibration == EchoCalibration::Calibrating
            && self.processor.statistics().delay_ms.is_some()
        {
            self.calibration = EchoCalibration::Active;
        }
        Ok(())
    }
}

fn mean_square(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32
}

fn echo_error(action: &str, err: sonora::Error) -> Error {
    Error::InvalidAudio {
        message: format!("echo control could not {action}: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(format: PcmFormat, capacity: usize) -> (Arc<SampleRing>, EchoReference) {
        let ring = Arc::new(SampleRing::new(capacity));
        (
            Arc::clone(&ring),
            EchoReference::new(Arc::clone(&ring), format),
        )
    }

    #[test]
    fn ten_ms_blocks_preserve_near_end_without_playback() {
        let (_ring, reference) = reference(PcmFormat::v0(), 3_200);
        let mut echo = EchoController::new(reference).expect("echo");
        let capture = (0..AEC_FRAME_SAMPLES * 2 + 17)
            .map(|sample| (sample as f32 * 0.11).sin() * 0.2)
            .collect::<Vec<_>>();
        let clean = echo.process_capture(&capture).expect("clean capture");
        assert_eq!(clean.len(), AEC_FRAME_SAMPLES * 2);
        assert!(mean_square(&clean) > 0.01);
        assert_eq!(echo.calibration(), EchoCalibration::WaitingForPlayback);
    }

    #[test]
    fn render_reference_converts_device_stereo_to_v0() {
        let format = PcmFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let (ring, reference) = reference(format, 48_000);
        let mut echo = EchoController::new(reference).expect("echo");
        let stereo = vec![0.25; 480 * 2];
        assert_eq!(ring.try_push_slice(&stereo), stereo.len());
        let capture = vec![0.05; AEC_FRAME_SAMPLES];
        let clean = echo.process_capture(&capture).expect("clean capture");
        assert_eq!(clean.len(), AEC_FRAME_SAMPLES);
        assert_eq!(echo.far_end_blocks(), 1);
        assert_ne!(echo.calibration(), EchoCalibration::WaitingForPlayback);
    }

    #[test]
    fn lost_reference_marks_calibration_degraded() {
        let (ring, reference) = reference(PcmFormat::v0(), AEC_FRAME_SAMPLES);
        let mut echo = EchoController::new(reference).expect("echo");
        let render = vec![0.2; AEC_FRAME_SAMPLES * 2];
        assert_eq!(ring.try_push_slice(&render), AEC_FRAME_SAMPLES);
        assert!(ring.overruns() > 0);
        echo.process_capture(&vec![0.1; AEC_FRAME_SAMPLES])
            .expect("clean capture");
        assert_eq!(echo.calibration(), EchoCalibration::Degraded);
    }

    #[test]
    fn delayed_far_end_echo_is_suppressed_and_near_end_survives() {
        const DELAY_BLOCKS: usize = 8;
        const ADAPT_BLOCKS: usize = 600;
        const MEASURE_BLOCKS: usize = 200;

        let (ring, reference) = reference(PcmFormat::v0(), AEC_FRAME_SAMPLES * 4);
        let mut echo = EchoController::new(reference).expect("echo");
        let mut history = VecDeque::from(vec![vec![0.0; AEC_FRAME_SAMPLES]; DELAY_BLOCKS]);
        let mut echo_input_power = 0.0;
        let mut echo_output_power = 0.0;

        for block in 0..(ADAPT_BLOCKS + MEASURE_BLOCKS) {
            let render = speech_like_block(block);
            assert_eq!(ring.try_push_slice(&render), render.len());
            let delayed = history.pop_front().expect("delay line");
            history.push_back(render);
            let capture = delayed
                .iter()
                .map(|sample| sample * 0.55)
                .collect::<Vec<_>>();
            let clean = echo.process_capture(&capture).expect("cancel echo");
            if block >= ADAPT_BLOCKS {
                echo_input_power += mean_square(&capture);
                echo_output_power += mean_square(&clean);
            }
        }
        assert!(
            echo_output_power < echo_input_power * 0.35,
            "AEC left too much far-end energy: input={echo_input_power} output={echo_output_power}"
        );

        let mut near_input_power = 0.0;
        let mut near_output_power = 0.0;
        for block in 0..MEASURE_BLOCKS {
            let render = speech_like_block(block + ADAPT_BLOCKS + MEASURE_BLOCKS);
            assert_eq!(ring.try_push_slice(&render), render.len());
            let delayed = history.pop_front().expect("delay line");
            history.push_back(render);
            let near = near_end_block(block);
            let capture = delayed
                .iter()
                .zip(near.iter())
                .map(|(far, near)| far * 0.55 + near)
                .collect::<Vec<_>>();
            let clean = echo.process_capture(&capture).expect("double talk");
            near_input_power += mean_square(&near);
            near_output_power += mean_square(&clean);
        }
        assert!(
            near_output_power > near_input_power * 0.2,
            "AEC removed real near-end speech: near={near_input_power} output={near_output_power}"
        );
    }

    fn speech_like_block(block: usize) -> Vec<f32> {
        (0..AEC_FRAME_SAMPLES)
            .map(|offset| {
                let n = (block * AEC_FRAME_SAMPLES + offset) as f32;
                ((n * 0.071).sin() + (n * 0.037).sin() * 0.6 + (n * 0.013).cos() * 0.3) * 0.22
            })
            .collect()
    }

    fn near_end_block(block: usize) -> Vec<f32> {
        (0..AEC_FRAME_SAMPLES)
            .map(|offset| {
                let n = (block * AEC_FRAME_SAMPLES + offset) as f32;
                (n * 0.109).sin() * 0.18
            })
            .collect()
    }
}
