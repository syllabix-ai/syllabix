//! Full-duplex acoustic echo cancellation.
//!
//! The speaker callback records the samples actually submitted to the audio
//! device. [`EchoController`] converts that render reference and microphone
//! capture to 16 kHz mono, processes synchronized 10 ms blocks through WebRTC
//! AEC3, and returns cleaned 16 kHz capture samples for ASR. Silero performs
//! its own separate 8 kHz conversion in the VAD adapter.

use std::collections::VecDeque;
use std::sync::Arc;

use sonora::config::{EchoCanceller, HighPassFilter};
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
/// Hold capture at most this long while a partial speaker reference is in flight.
const MAX_CAPTURE_HOLD_SAMPLES: usize = AEC_FRAME_SAMPLES * 8;
/// Cross-correlation window before applying a one-shot delay hint (500 ms).
const DELAY_ESTIMATE_SAMPLES: usize = AEC_FRAME_SAMPLES * 50;
/// Search 0..=300 ms of acoustic delay between the render tap and the mic.
const DELAY_SEARCH_SAMPLES: usize = AEC_FRAME_SAMPLES * 30;

/// Read-only render tap shared by the speaker callback and capture worker.
pub struct EchoReference {
    pub(crate) ring: Arc<SampleRing>,
    pub(crate) device_format: PcmFormat,
}

impl EchoReference {
    /// Wrap a speaker-callback tap. `device_format` is the PCM layout in the ring.
    pub fn new(ring: Arc<SampleRing>, device_format: PcmFormat) -> Self {
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

/// Time-aligned 16 kHz mono dump of one AEC session (debug / fixtures).
#[derive(Debug, Clone)]
pub struct EchoRecording {
    /// Speaker-callback reference after conversion to 16 kHz mono.
    pub render: Vec<f32>,
    /// Microphone capture before AEC, 16 kHz mono.
    pub capture: Vec<f32>,
    /// AEC output fed to Silero / ASR, 16 kHz mono.
    pub clean: Vec<f32>,
    /// Times capture was held waiting for a partial render block.
    pub render_starved_holds: u64,
    /// One-shot cross-correlation delay hint applied to AEC3, if any.
    pub estimated_delay_ms: Option<i32>,
    /// Last AEC3 instantaneous delay statistic.
    pub aec_delay_ms: Option<i32>,
    pub calibration: EchoCalibration,
    pub far_end_blocks: u64,
}

/// Sonora AEC3 wrapper operating on the existing 16 kHz capture/ASR stream.
pub struct EchoController {
    processor: AudioProcessing,
    reference: EchoReference,
    render_converter: PcmConverter,
    render_pending: VecDeque<f32>,
    capture_pending: VecDeque<f32>,
    clean_pending: VecDeque<f32>,
    last_dropped_samples: u64,
    far_end_blocks: u64,
    render_starved_holds: u64,
    calibration: EchoCalibration,
    recording: bool,
    rec_render: Vec<f32>,
    rec_capture: Vec<f32>,
    rec_clean: Vec<f32>,
    delay_render: Vec<f32>,
    delay_capture: Vec<f32>,
    delay_applied: bool,
    estimated_delay_ms: Option<i32>,
    last_render_block: Vec<f32>,
}

impl EchoController {
    /// Enable AEC3 and attach the exact speaker callback reference.
    pub fn new(reference: EchoReference) -> Result<Self> {
        let stream = StreamConfig::new(DEFAULT_SAMPLE_RATE_HZ, 1);
        let config = Config {
            echo_canceller: Some(EchoCanceller::default()),
            high_pass_filter: Some(HighPassFilter::default()),
            ..Default::default()
        };
        let mut processor = AudioProcessing::builder()
            .config(config)
            .capture_config(stream)
            .render_config(stream)
            .echo_detector(true)
            .build();
        // The reference is tapped at the output callback, so software queueing
        // delay is already excluded. AEC3 still needs the acoustic/DAC delay.
        // 0 is the starting hint; a one-shot cross-correlation updates it.
        let _ = processor.set_stream_delay_ms(0);
        let render_converter = PcmConverter::new(reference.device_format, PcmFormat::v0())?;
        let last_dropped_samples = reference.dropped_samples();
        Ok(Self {
            processor,
            reference,
            render_converter,
            render_pending: VecDeque::with_capacity(MAX_PENDING_SAMPLES),
            capture_pending: VecDeque::with_capacity(AEC_FRAME_SAMPLES * 4),
            clean_pending: VecDeque::with_capacity(AEC_FRAME_SAMPLES * 4),
            last_dropped_samples,
            far_end_blocks: 0,
            render_starved_holds: 0,
            calibration: EchoCalibration::WaitingForPlayback,
            recording: false,
            rec_render: Vec::new(),
            rec_capture: Vec::new(),
            rec_clean: Vec::new(),
            delay_render: Vec::new(),
            delay_capture: Vec::new(),
            delay_applied: false,
            estimated_delay_ms: None,
            last_render_block: vec![0.0; AEC_FRAME_SAMPLES],
        })
    }

    /// Keep 16 kHz render/capture/clean PCM for a debug dump.
    pub fn enable_recording(&mut self) {
        self.recording = true;
    }

    /// Snapshot of recorded PCM and AEC diagnostics.
    pub fn recording(&self) -> EchoRecording {
        EchoRecording {
            render: self.rec_render.clone(),
            capture: self.rec_capture.clone(),
            clean: self.rec_clean.clone(),
            render_starved_holds: self.render_starved_holds,
            estimated_delay_ms: self.estimated_delay_ms,
            aec_delay_ms: self.processor.statistics().delay_ms,
            calibration: self.calibration,
            far_end_blocks: self.far_end_blocks,
        }
    }

    /// Process arbitrary-length 16 kHz mono capture samples.
    ///
    /// Output is emitted only in complete 10 ms blocks; callers retain their
    /// existing frame splitter for the 512-sample ASR capture contract.
    ///
    /// Render is pushed into AEC3 as soon as a 10 ms speaker block exists.
    /// Capture is not paired with invented silence when the mic is briefly
    /// ahead of the speaker callback.
    pub fn process_capture(&mut self, capture: &[f32]) -> Result<Vec<f32>> {
        self.capture_pending.extend(capture.iter().copied());
        self.drain_and_process_render()?;

        while self.capture_pending.len() >= AEC_FRAME_SAMPLES {
            if self.should_hold_capture() {
                self.render_starved_holds += 1;
                break;
            }
            self.drain_and_process_render()?;
            self.process_one_capture_block()?;
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

    fn drain_and_process_render(&mut self) -> Result<()> {
        self.drain_render_reference();
        while self.render_pending.len() >= AEC_FRAME_SAMPLES {
            let render = self
                .render_pending
                .drain(..AEC_FRAME_SAMPLES)
                .collect::<Vec<_>>();
            self.process_render_block(&render)?;
        }
        Ok(())
    }

    fn should_hold_capture(&self) -> bool {
        if self.capture_pending.len() >= MAX_CAPTURE_HOLD_SAMPLES {
            return false;
        }
        if self.render_pending.len() >= AEC_FRAME_SAMPLES {
            return false;
        }
        !self.render_pending.is_empty() || self.reference.ring.occupancy() > 0
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

    fn process_render_block(&mut self, render: &[f32]) -> Result<()> {
        if mean_square(render) > PLAYBACK_ENERGY_FLOOR {
            self.far_end_blocks += 1;
            if self.calibration == EchoCalibration::WaitingForPlayback {
                self.calibration = EchoCalibration::Calibrating;
            }
        }

        let mut render_output = vec![0.0; AEC_FRAME_SAMPLES];
        self.processor
            .process_render_f32(&[render], &mut [&mut render_output])
            .map_err(|err| echo_error("process speaker reference", err))?;

        self.last_render_block = render.to_vec();
        if !self.delay_applied && self.calibration == EchoCalibration::Calibrating {
            self.delay_render.extend_from_slice(render);
        }
        Ok(())
    }

    fn process_one_capture_block(&mut self) -> Result<()> {
        let capture = self
            .capture_pending
            .drain(..AEC_FRAME_SAMPLES)
            .collect::<Vec<_>>();
        if !self.delay_applied && self.calibration == EchoCalibration::Calibrating {
            self.delay_capture.extend_from_slice(&capture);
            self.maybe_apply_delay_estimate()?;
        }

        let mut clean = vec![0.0; AEC_FRAME_SAMPLES];
        self.processor
            .process_capture_f32(&[&capture], &mut [&mut clean])
            .map_err(|err| echo_error("process microphone capture", err))?;
        self.clean_pending.extend(clean.iter().copied());

        if self.recording {
            self.rec_render.extend_from_slice(&self.last_render_block);
            self.rec_capture.extend_from_slice(&capture);
            self.rec_clean.extend_from_slice(&clean);
        }

        if self.calibration == EchoCalibration::Calibrating
            && (self.delay_applied || self.processor.statistics().delay_ms.is_some())
        {
            self.calibration = EchoCalibration::Active;
        }
        Ok(())
    }

    fn maybe_apply_delay_estimate(&mut self) -> Result<()> {
        if self.delay_applied {
            return Ok(());
        }
        if self.delay_render.len() < DELAY_ESTIMATE_SAMPLES
            || self.delay_capture.len() < DELAY_ESTIMATE_SAMPLES
        {
            return Ok(());
        }
        let lag = best_delay_lag(
            &self.delay_render,
            &self.delay_capture,
            DELAY_SEARCH_SAMPLES,
        );
        let delay_ms = (lag as i32 * 1000) / DEFAULT_SAMPLE_RATE_HZ as i32;
        let _ = self.processor.set_stream_delay_ms(delay_ms);
        self.estimated_delay_ms = Some(delay_ms);
        self.delay_applied = true;
        self.delay_render.clear();
        self.delay_capture.clear();
        Ok(())
    }
}

fn mean_square(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32
}

/// Lag (in samples) that best aligns delayed capture with render.
fn best_delay_lag(render: &[f32], capture: &[f32], max_lag: usize) -> usize {
    let mut best_lag = 0usize;
    let mut best_score = f32::MIN;
    let max_lag = max_lag.min(capture.len().saturating_sub(AEC_FRAME_SAMPLES));
    for lag in 0..=max_lag {
        let n = render.len().min(capture.len().saturating_sub(lag));
        if n < AEC_FRAME_SAMPLES * 10 {
            break;
        }
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += render[i] * capture[i + lag];
        }
        let score = acc / n as f32;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }
    best_lag
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

    #[test]
    fn capture_without_playback_does_not_feed_silent_render() {
        let (_ring, reference) = reference(PcmFormat::v0(), AEC_FRAME_SAMPLES * 4);
        let mut echo = EchoController::new(reference).expect("echo");
        echo.enable_recording();
        let clean = echo
            .process_capture(&speech_like_block(0))
            .expect("near-end only");
        assert_eq!(clean.len(), AEC_FRAME_SAMPLES);
        let rec = echo.recording();
        assert_eq!(rec.render.len(), rec.capture.len());
        assert!(
            rec.render.iter().all(|s| *s == 0.0),
            "idle speakers must not enqueue invented far-end into AEC3"
        );
        assert_eq!(echo.far_end_blocks(), 0);
    }

    #[test]
    fn partial_render_block_holds_capture() {
        let (ring, reference) = reference(PcmFormat::v0(), AEC_FRAME_SAMPLES * 4);
        let mut echo = EchoController::new(reference).expect("echo");
        let half = vec![0.2; AEC_FRAME_SAMPLES / 2];
        assert_eq!(ring.try_push_slice(&half), half.len());
        let clean = echo
            .process_capture(&speech_like_block(0))
            .expect("wait for rest of render");
        assert!(
            clean.is_empty(),
            "must not pair capture with a half render block"
        );
        assert_eq!(ring.try_push_slice(&half), half.len());
        let clean = echo.process_capture(&[]).expect("complete render");
        assert_eq!(clean.len(), AEC_FRAME_SAMPLES);
        assert_eq!(echo.far_end_blocks(), 1);
    }

    #[test]
    fn cross_correlation_finds_eighty_ms_acoustic_delay() {
        let render: Vec<f32> = (0..AEC_FRAME_SAMPLES * 80)
            .map(|n| ((n as f32) * 0.071).sin() * 0.3)
            .collect();
        let delay = AEC_FRAME_SAMPLES * 8;
        let mut capture = vec![0.0; delay];
        capture.extend(render.iter().map(|s| s * 0.5));
        let lag = super::best_delay_lag(&render, &capture, AEC_FRAME_SAMPLES * 30);
        assert!(
            (lag as i32 - delay as i32).abs() <= 2,
            "expected ~{delay} samples, got {lag}"
        );
    }
}
