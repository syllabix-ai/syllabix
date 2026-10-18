//! Silero ONNX voice-activity detection for 16 kHz mono pipeline audio.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

use ort::session::Session;

use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Vad;
use crate::types::{
    AudioFrame, TurnId, Utterance, VadEvent, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};
use crate::Cancel;

/// Silero probability at or above which a frame begins or continues speech.
pub const SPEECH_THRESHOLD: f32 = 0.6;

/// A turn starts only after this much contiguous speech (twelve 32 ms frames).
pub const MIN_SPEECH: Duration = Duration::from_millis(384);

/// Silence needed to close an utterance (twenty-five 32 ms frames = 800 ms).
pub const END_SILENCE: Duration = Duration::from_millis(800);

/// Frame duration: 512 samples at 16 kHz.
pub const FRAME_DURATION: Duration = Duration::from_millis(32);

/// Frames of speech that meet [`MIN_SPEECH`].
pub const MIN_SPEECH_FRAMES: usize = 12;

/// Frames of silence that meet [`END_SILENCE`].
pub const END_SILENCE_FRAMES: usize = 25;

/// Dead-band margin below [`SPEECH_THRESHOLD`] that still holds a turn open.
///
/// While a turn is open, only frames scoring below
/// [`VadSettings::silence_threshold`] count toward the closing hangover. Frames
/// between that edge and the threshold (quiet consonants, brief dips) neither
/// reset silence nor advance it, so borderline frames do not cut the user off.
pub const HYSTERESIS: f32 = 0.15;

/// Floor for [`VadSettings::silence_threshold`].
///
/// Without it, a `speech_threshold` at or below [`HYSTERESIS`] would put the
/// hangover edge at or below zero, where `probability < edge` never holds and
/// an open turn could never close.
pub const SILENCE_FLOOR: f32 = 0.01;

/// Idle non-speech that clears Silero's recurrent state.
///
/// Silero carries an LSTM state between frames, so room noise while nobody
/// talks accumulates in the model's memory and biases the first real frame of
/// the next turn. Clearing the state after this much contiguous idle
/// non-speech makes that first frame start from the neutral initial state.
pub const IDLE_RESET: Duration = Duration::from_millis(500);

/// Post-AEC audio prepended onto the Whisper utterance before the first
/// at-or-above-threshold frame.
pub const WHISPER_PREROLL: Duration = Duration::from_millis(200);

/// Samples in [`WHISPER_PREROLL`] at the capture rate.
pub const PREROLL_SAMPLES: usize =
    (DEFAULT_SAMPLE_RATE_HZ as usize) * (WHISPER_PREROLL.as_millis() as usize) / 1000;

/// Silero turn policy. Frame size and 8 kHz pair-average stay fixed.
#[derive(Debug, Clone, PartialEq)]
pub struct VadSettings {
    /// Silero probability at or above which a frame is speech.
    pub speech_threshold: f32,
    /// Contiguous speech required to open a turn.
    pub min_speech: Duration,
    /// Silence required to close a turn.
    pub end_silence: Duration,
    /// Post-AEC audio prepended before the first speech frame.
    pub preroll: Duration,
}

impl VadSettings {
    /// Built-in detector settings.
    pub fn v0() -> Self {
        Self {
            speech_threshold: SPEECH_THRESHOLD,
            min_speech: MIN_SPEECH,
            end_silence: END_SILENCE,
            preroll: WHISPER_PREROLL,
        }
    }

    /// Samples of preroll at the capture rate.
    pub fn preroll_samples(&self) -> usize {
        (DEFAULT_SAMPLE_RATE_HZ as usize) * (self.preroll.as_millis() as usize) / 1000
    }

    /// Probability below which a frame counts toward the closing hangover.
    ///
    /// `speech_threshold - HYSTERESIS`, floored at [`SILENCE_FLOOR`] so a very
    /// low `speech_threshold` still lets the hangover advance on near-silence.
    pub fn silence_threshold(&self) -> f32 {
        (self.speech_threshold - HYSTERESIS).max(SILENCE_FLOOR)
    }
}

/// In-process Silero VAD backed by ONNX Runtime.
///
/// The detector accepts fixed-size frames: 512 samples of
/// 16 kHz mono PCM. It pair-averages each frame to Silero's 8 kHz / 256-sample
/// window (the ONNX 512 / `sr=16000` branch scores ~0.003). It retains
/// Silero's recurrent state between frames.
pub struct SileroVad {
    scorer: Box<dyn ProbabilityScorer>,
    settings: VadSettings,
    next_turn: u64,
    /// Recent post-AEC frames that are not part of the open speech clip.
    history: VecDeque<AudioFrame>,
    /// Snapshot of [`WHISPER_PREROLL`] taken at the first speech frame.
    preroll: Vec<AudioFrame>,
    /// Frames for the open (or still-tentative) utterance, including hangover.
    current: Vec<AudioFrame>,
    /// Set once [`MIN_SPEECH`] has elapsed; `None` means frames are tentative.
    active: Option<TurnId>,
    speech: Duration,
    silence: Duration,
    /// Consecutive non-speech while idle, for the [`IDLE_RESET`] state clear.
    idle: Duration,
}

impl SileroVad {
    /// Load the cached `silero_vad.onnx` asset with ONNX Runtime.
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self> {
        let scorer = OrtScorer::load(path.as_ref())?;
        Ok(Self::with_scorer(Box::new(scorer)))
    }

    /// Resolve the manifest's Silero asset, downloading it on a cache miss,
    /// then load it with ONNX Runtime.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let asset = cache
            .manifest()
            .asset("silero")
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the silero asset".into(),
            })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_model_path(path)
    }

    /// Apply yaml (or built-in) turn policy. Scoring path is unchanged.
    pub fn with_settings(mut self, settings: VadSettings) -> Self {
        self.settings = settings;
        self
    }

    fn with_scorer(scorer: Box<dyn ProbabilityScorer>) -> Self {
        Self {
            scorer,
            settings: VadSettings::v0(),
            next_turn: 0,
            history: VecDeque::new(),
            preroll: Vec::new(),
            current: Vec::new(),
            active: None,
            speech: Duration::ZERO,
            silence: Duration::ZERO,
            idle: Duration::ZERO,
        }
    }

    fn probability(&mut self, frame: &AudioFrame) -> Result<f32> {
        frame.validate()?;
        if frame.samples.len() != FRAME_SAMPLES {
            return Err(Error::InvalidAudio {
                message: format!(
                    "Silero requires {FRAME_SAMPLES} samples per frame; got {}",
                    frame.samples.len()
                ),
            });
        }
        self.scorer.score(frame)
    }

    /// Return the Silero probability for one validated pipeline frame.
    pub fn debug_probability(&mut self, frame: &AudioFrame) -> Result<f32> {
        self.probability(frame)
    }

    fn promote(&mut self) -> Option<VadEvent> {
        if self.active.is_some() || self.speech < self.settings.min_speech {
            return None;
        }
        let turn = TurnId(self.next_turn);
        self.next_turn += 1;
        self.active = Some(turn);
        Some(VadEvent::SpeechStart { turn })
    }

    fn finish(&mut self) -> Option<VadEvent> {
        self.silence = Duration::ZERO;
        self.speech = Duration::ZERO;
        self.idle = Duration::ZERO;
        let mut frames = std::mem::take(&mut self.preroll);
        frames.extend(std::mem::take(&mut self.current));
        let event = self.active.take().map(|turn| VadEvent::SpeechEnd {
            utterance: Utterance { turn, frames },
        });
        if event.is_some() {
            // Recurrent hangover otherwise suppresses the next user turn
            // (identical or similar speech stays below the threshold).
            self.scorer.reset();
        }
        event
    }

    fn push_history(&mut self, frame: AudioFrame) {
        self.history.push_back(frame);
        loop {
            let total: usize = self.history.iter().map(|frame| frame.samples.len()).sum();
            let Some(front) = self.history.front() else {
                break;
            };
            if total.saturating_sub(front.samples.len()) >= self.settings.preroll_samples() {
                self.history.pop_front();
            } else {
                break;
            }
        }
    }

    fn drop_tentative(&mut self, silence: AudioFrame) {
        let leftover = std::mem::take(&mut self.current);
        for frame in leftover {
            self.push_history(frame);
        }
        self.push_history(silence);
        self.preroll.clear();
        self.speech = Duration::ZERO;
        self.silence = Duration::ZERO;
        self.idle = Duration::ZERO;
    }
}

impl Vad for SileroVad {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().vad.as_str()
    }

    fn push_frame(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>> {
        let probability = self.probability(&frame)?;
        let speech = probability >= self.settings.speech_threshold;
        let mut events = Vec::new();

        if speech {
            self.silence = Duration::ZERO;
            self.idle = Duration::ZERO;
            if self.current.is_empty() {
                self.preroll = preroll_frames(&self.history, self.settings.preroll_samples());
            }
            self.current.push(frame);
            self.speech += FRAME_DURATION;
            if let Some(event) = self.promote() {
                events.push(event);
            }
        } else if self.active.is_some() {
            // Keep every post-start frame on the Whisper clip. Below-threshold
            // audio is often a quiet consonant or a short pause, not
            // end-of-turn, and the closing hangover itself still contains word
            // endings. Only frames clearly below the threshold
            // (`silence_threshold`) advance the hangover; the dead-band in
            // between holds silence where it is.
            self.current.push(frame.clone());
            self.push_history(frame);
            if probability < self.settings.silence_threshold() {
                self.silence += FRAME_DURATION;
            }
            // Count elapsed Duration, not integer frames. Close on the first
            // frame where silence is ≥ `end_silence` (25 × 32 ms = 800 ms).
            if self.silence >= self.settings.end_silence {
                if let Some(event) = self.finish() {
                    events.push(event);
                }
            }
        } else if !self.current.is_empty() {
            self.idle = Duration::ZERO;
            self.drop_tentative(frame);
        } else {
            // Truly idle: no turn open and nothing tentative. Accumulate
            // non-speech so room noise does not build up in Silero's recurrent
            // state and bias the next turn's opening frames.
            self.idle += FRAME_DURATION;
            if self.idle >= IDLE_RESET {
                self.scorer.reset();
                self.idle = Duration::ZERO;
            }
            self.push_history(frame);
        }

        Ok(events)
    }

    fn flush(&mut self) -> Result<Vec<VadEvent>> {
        Ok(self.finish().into_iter().collect())
    }
}

impl SileroVad {
    /// Clear recurrent state and any in-flight utterance.
    pub fn reset(&mut self) {
        self.scorer.reset();
        self.next_turn = 0;
        self.history.clear();
        self.preroll.clear();
        self.current.clear();
        self.active = None;
        self.speech = Duration::ZERO;
        self.silence = Duration::ZERO;
        self.idle = Duration::ZERO;
    }
}

fn preroll_frames(history: &VecDeque<AudioFrame>, preroll_samples: usize) -> Vec<AudioFrame> {
    let total: usize = history.iter().map(|frame| frame.samples.len()).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut skip = total.saturating_sub(preroll_samples);
    let mut out = Vec::new();
    for frame in history {
        if skip >= frame.samples.len() {
            skip -= frame.samples.len();
            continue;
        }
        let start = skip;
        skip = 0;
        let samples = frame.samples[start..].to_vec();
        let capture_pcm = frame.capture_pcm.as_ref().map(|pcm| {
            if start < pcm.len() {
                pcm[start..].to_vec()
            } else {
                Vec::new()
            }
        });
        out.push(AudioFrame {
            seq: frame.seq,
            sample_rate_hz: frame.sample_rate_hz,
            channels: frame.channels,
            samples,
            capture_pcm,
        });
    }
    out
}

trait ProbabilityScorer: Send {
    fn score(&mut self, frame: &AudioFrame) -> Result<f32>;
    fn reset(&mut self);
}

struct OrtScorer {
    session: Session,
    state: Vec<f32>,
}

impl OrtScorer {
    fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_error)?
            .commit_from_file(path)
            .map_err(ort_error)?;
        let input_names: Vec<&str> = session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        if input_names != ["input", "state", "sr"] {
            return Err(Error::Provider {
                provider: "silero",
                message: format!("unexpected ONNX inputs: {}", input_names.join(", ")),
            });
        }
        let output_names: Vec<&str> = session
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect();
        if output_names != ["output", "stateN"] {
            return Err(Error::Provider {
                provider: "silero",
                message: format!("unexpected ONNX outputs: {}", output_names.join(", ")),
            });
        }
        Ok(Self {
            session,
            state: vec![0.0; 2 * 128],
        })
    }
}

impl ProbabilityScorer for OrtScorer {
    fn score(&mut self, frame: &AudioFrame) -> Result<f32> {
        // silero_vad.onnx (v6.2.1): 512 samples at sr=16000 scores ~0.003.
        // Two 256-sample 16 kHz windows hear speech but split speech.wav
        // across a pause longer than the hangover. Pair-average to 8 kHz.
        debug_assert_eq!(frame.sample_rate_hz, DEFAULT_SAMPLE_RATE_HZ);
        let audio: Vec<f32> = frame
            .samples
            .chunks_exact(2)
            .map(|pair| (f32::from(pair[0]) + f32::from(pair[1])) / (2.0 * 32_768.0))
            .collect();
        let outputs = self
            .session
            .run(
                ort::inputs![
                    "input" => ([1_usize, 256], audio),
                    "state" => ([2_usize, 1, 128], self.state.clone()),
                    "sr" => ((), vec![8_000_i64]),
                ]
                .map_err(ort_error)?,
            )
            .map_err(ort_error)?;
        let probability = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .first()
            .copied()
            .ok_or_else(|| Error::Provider {
                provider: "silero",
                message: "ONNX output probability was empty".into(),
            })?;
        self.state = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?
            .iter()
            .copied()
            .collect();
        Ok(probability)
    }

    fn reset(&mut self) {
        self.state = vec![0.0; 2 * 128];
    }
}

fn ort_error(error: ort::Error) -> Error {
    Error::Provider {
        provider: "silero",
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Vad;
    use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ};

    struct ScriptedScorer(Vec<f32>);

    impl ProbabilityScorer for ScriptedScorer {
        fn score(&mut self, _frame: &AudioFrame) -> Result<f32> {
            Ok(self.0.remove(0))
        }

        fn reset(&mut self) {
            self.0.clear();
        }
    }

    fn frame(seq: u64) -> AudioFrame {
        AudioFrame::new(
            seq,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![100; FRAME_SAMPLES],
        )
        .unwrap()
    }

    fn push_seq(vad: &mut SileroVad, seq: u64) -> Vec<VadEvent> {
        vad.push_frame(frame(seq)).unwrap()
    }

    #[test]
    fn speech_and_silence_boundaries_follow_fixed_thresholds() {
        let probabilities = std::iter::once(0.2)
            .chain(std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES))
            .chain(std::iter::repeat_n(0.1, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        let total = 1 + MIN_SPEECH_FRAMES + END_SILENCE_FRAMES;
        for seq in 0..total as u64 {
            events.extend(push_seq(&mut vad, seq));
            if seq + 1 < (1 + MIN_SPEECH_FRAMES) as u64 {
                assert!(
                    events.is_empty(),
                    "SpeechStart must wait for {MIN_SPEECH:?}"
                );
            }
            if seq + 1 == (1 + MIN_SPEECH_FRAMES) as u64 {
                assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
            }
            if seq + 1 == (1 + MIN_SPEECH_FRAMES + END_SILENCE_FRAMES - 1) as u64 {
                assert_eq!(
                    events.len(),
                    1,
                    "{} silence frames are under {END_SILENCE:?}",
                    END_SILENCE_FRAMES - 1
                );
            }
        }

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], VadEvent::SpeechStart { turn: TurnId(0) });
        let VadEvent::SpeechEnd { utterance } = &events[1] else {
            panic!("expected a speech end");
        };
        assert_eq!(utterance.turn, TurnId(0));
        assert_eq!(
            utterance.frames.len(),
            1 + MIN_SPEECH_FRAMES + END_SILENCE_FRAMES
        );
        assert_eq!(utterance.frames[0].seq, 0, "leading silence is preroll");
        assert_eq!(utterance.frames[1].seq, 1);
        assert_eq!(
            utterance.frames[MIN_SPEECH_FRAMES].seq,
            MIN_SPEECH_FRAMES as u64
        );
        assert_eq!(
            utterance.frames.last().map(|f| f.seq),
            Some((1 + MIN_SPEECH_FRAMES + END_SILENCE_FRAMES - 1) as u64),
            "hangover frames stay on the Whisper clip"
        );
        assert_eq!(
            utterance.pcm().len(),
            (1 + MIN_SPEECH_FRAMES + END_SILENCE_FRAMES) * FRAME_SAMPLES
        );
        assert!(FRAME_DURATION * (END_SILENCE_FRAMES as u32 - 1) < END_SILENCE);
        assert!(FRAME_DURATION * END_SILENCE_FRAMES as u32 >= END_SILENCE);
    }

    #[test]
    fn below_threshold_frames_during_an_open_turn_stay_on_the_whisper_clip() {
        let dip = 3;
        let probabilities = std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES)
            .chain(std::iter::repeat_n(0.1, dip))
            .chain(std::iter::repeat_n(0.9, 2))
            .chain(std::iter::repeat_n(0.1, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        for (seq, fill) in std::iter::repeat_n(100_i16, MIN_SPEECH_FRAMES)
            .chain(std::iter::repeat_n(7, dip))
            .chain(std::iter::repeat_n(100, 2))
            .chain(std::iter::repeat_n(0, END_SILENCE_FRAMES))
            .enumerate()
        {
            events.extend(vad.push_frame(marked_frame(seq as u64, fill)).unwrap());
        }
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, VadEvent::SpeechStart { .. }))
                .count(),
            1,
            "a short dip must not close the turn"
        );
        let Some(VadEvent::SpeechEnd { utterance }) = events.last() else {
            panic!("expected speech end, got {events:?}");
        };
        let pcm = utterance.pcm();
        let dip_pcm =
            &pcm[MIN_SPEECH_FRAMES * FRAME_SAMPLES..(MIN_SPEECH_FRAMES + dip) * FRAME_SAMPLES];
        assert!(
            dip_pcm.iter().all(|s| *s == 7),
            "intra-utterance below-threshold audio must reach Whisper, got {:?}",
            dip_pcm.iter().copied().take(8).collect::<Vec<_>>()
        );
    }

    fn marked_frame(seq: u64, fill: i16) -> AudioFrame {
        AudioFrame::new(
            seq,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![fill; FRAME_SAMPLES],
        )
        .unwrap()
    }

    #[test]
    fn whisper_utterance_includes_200ms_preroll() {
        let preroll_frames = 8;
        let probabilities = std::iter::repeat_n(0.1, preroll_frames)
            .chain(std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES))
            .chain(std::iter::repeat_n(0.1, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        let mut seq = 0_u64;
        for _ in 0..preroll_frames {
            events.extend(
                vad.push_frame(marked_frame(seq, i16::try_from(seq).unwrap()))
                    .unwrap(),
            );
            seq += 1;
        }
        for _ in 0..MIN_SPEECH_FRAMES {
            events.extend(vad.push_frame(marked_frame(seq, 100)).unwrap());
            seq += 1;
        }
        for _ in 0..END_SILENCE_FRAMES {
            events.extend(vad.push_frame(marked_frame(seq, 0)).unwrap());
            seq += 1;
        }
        let Some(VadEvent::SpeechEnd { utterance }) = events.last() else {
            panic!("expected speech end, got {events:?}");
        };
        let pcm = utterance.pcm();
        assert!(pcm.len() >= PREROLL_SAMPLES + MIN_SPEECH_FRAMES * FRAME_SAMPLES);
        let preroll = &pcm[..PREROLL_SAMPLES];
        assert!(
            preroll.iter().all(|s| *s != 100 && *s != 0),
            "preroll must be the audio before the first speech frame"
        );
        let speech = &pcm[PREROLL_SAMPLES..PREROLL_SAMPLES + MIN_SPEECH_FRAMES * FRAME_SAMPLES];
        assert!(speech.iter().all(|s| *s == 100));
        assert_eq!(preroll.len(), PREROLL_SAMPLES);
        assert_eq!(WHISPER_PREROLL, Duration::from_millis(200));
    }

    #[test]
    fn yaml_min_speech_can_require_more_frames() {
        let settings = VadSettings {
            min_speech: Duration::from_millis(400),
            ..VadSettings::v0()
        };
        let probabilities = std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES + 1).collect();
        let mut vad =
            SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities))).with_settings(settings);
        let mut events = Vec::new();
        for seq in 0..MIN_SPEECH_FRAMES as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert!(
            events.is_empty(),
            "400 ms min speech must ignore a 384 ms burst"
        );
        events.extend(push_seq(&mut vad, MIN_SPEECH_FRAMES as u64));
        assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
    }

    #[test]
    fn dead_band_frames_do_not_advance_the_closing_hangover() {
        // Speech, then END_SILENCE_FRAMES-1 frames in the dead-band
        // (threshold - HYSTERESIS <= p < threshold). Those frames must not
        // advance the hangover, so the turn stays open until a truly silent
        // frame pushes the count over.
        let dead_band = SPEECH_THRESHOLD - HYSTERESIS / 2.0;
        let probabilities = std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES)
            .chain(std::iter::repeat_n(dead_band, END_SILENCE_FRAMES))
            .chain(std::iter::repeat_n(0.0, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        let mut seq = 0_u64;
        for _ in 0..MIN_SPEECH_FRAMES {
            events.extend(push_seq(&mut vad, seq));
            seq += 1;
        }
        assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
        events.clear();
        for _ in 0..END_SILENCE_FRAMES {
            events.extend(push_seq(&mut vad, seq));
            seq += 1;
        }
        assert!(
            events.is_empty(),
            "dead-band frames must not close the turn"
        );
        for _ in 0..END_SILENCE_FRAMES {
            events.extend(push_seq(&mut vad, seq));
            seq += 1;
        }
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, VadEvent::SpeechEnd { .. }))
                .count(),
            1,
            "truly silent frames still close the turn"
        );
    }

    #[test]
    fn dead_band_frames_hold_accumulated_silence() {
        // Ten truly silent frames accumulate 320 ms of the hangover. Dead-band
        // frames must leave that count untouched, so only fifteen more silent
        // frames (not a fresh twenty-five) close the turn. If the dead-band
        // reset the counter, the turn would still be open at the end.
        let dead_band = SPEECH_THRESHOLD - HYSTERESIS / 2.0;
        let partial = 10;
        let probabilities = std::iter::repeat_n(0.9, MIN_SPEECH_FRAMES)
            .chain(std::iter::repeat_n(0.0, partial))
            .chain(std::iter::repeat_n(dead_band, END_SILENCE_FRAMES))
            .chain(std::iter::repeat_n(0.0, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        let mut seq = 0_u64;
        for _ in 0..MIN_SPEECH_FRAMES + partial + END_SILENCE_FRAMES {
            events.extend(push_seq(&mut vad, seq));
            seq += 1;
        }
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, VadEvent::SpeechEnd { .. })),
            "dead-band frames must not close the turn"
        );
        for _ in 0..END_SILENCE_FRAMES - partial {
            events.extend(push_seq(&mut vad, seq));
            seq += 1;
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, VadEvent::SpeechEnd { .. })),
            "silence accumulated before the dead-band must survive it"
        );
    }

    #[test]
    fn low_threshold_still_closes_the_turn() {
        // A speech_threshold at or below HYSTERESIS would put
        // `speech_threshold - HYSTERESIS` at or below zero without the floor,
        // leaving the hangover edge where `probability < edge` never holds.
        let settings = VadSettings {
            speech_threshold: 0.1,
            ..VadSettings::v0()
        };
        assert!(settings.silence_threshold() > 0.0);
        let probabilities = std::iter::repeat_n(0.2, MIN_SPEECH_FRAMES)
            .chain(std::iter::repeat_n(0.0, END_SILENCE_FRAMES))
            .collect();
        let mut vad =
            SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities))).with_settings(settings);
        let mut events = Vec::new();
        for seq in 0..(MIN_SPEECH_FRAMES + END_SILENCE_FRAMES) as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, VadEvent::SpeechEnd { .. })),
            "a low threshold must still let the hangover close the turn"
        );
    }

    #[test]
    fn idle_non_speech_clears_the_recurrent_state() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct CountingScorer {
            resets: Arc<AtomicUsize>,
        }
        impl ProbabilityScorer for CountingScorer {
            fn score(&mut self, _frame: &AudioFrame) -> Result<f32> {
                Ok(0.0)
            }
            fn reset(&mut self) {
                self.resets.fetch_add(1, Ordering::SeqCst);
            }
        }

        let resets = Arc::new(AtomicUsize::new(0));
        let mut vad = SileroVad::with_scorer(Box::new(CountingScorer {
            resets: Arc::clone(&resets),
        }));
        let idle_frames = (IDLE_RESET.as_millis() as usize).div_ceil(32);
        for seq in 0..(idle_frames - 1) as u64 {
            assert!(push_seq(&mut vad, seq).is_empty());
        }
        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "state clears only after a full {IDLE_RESET:?} of idle non-speech"
        );
        assert!(push_seq(&mut vad, (idle_frames - 1) as u64).is_empty());
        assert_eq!(
            resets.load(Ordering::SeqCst),
            1,
            "{IDLE_RESET:?} of idle non-speech clears the recurrent state"
        );
    }

    #[test]
    fn idle_reset_does_not_fire_while_a_turn_is_open() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct ReplayScorer {
            values: Vec<f32>,
            index: usize,
            resets: Arc<AtomicUsize>,
        }
        impl ProbabilityScorer for ReplayScorer {
            fn score(&mut self, _frame: &AudioFrame) -> Result<f32> {
                let value = self.values[self.index];
                self.index += 1;
                Ok(value)
            }
            fn reset(&mut self) {
                self.resets.fetch_add(1, Ordering::SeqCst);
            }
        }

        // More than IDLE_RESET of trailing silence, yet fewer frames than
        // END_SILENCE needs to close, so the turn stays open.
        let idle_frames = (IDLE_RESET.as_millis() as usize).div_ceil(32);
        assert!(idle_frames < END_SILENCE_FRAMES);
        let mut values = vec![0.9; MIN_SPEECH_FRAMES];
        values.extend(std::iter::repeat_n(0.0, idle_frames));
        let resets = Arc::new(AtomicUsize::new(0));
        let mut vad = SileroVad::with_scorer(Box::new(ReplayScorer {
            values,
            index: 0,
            resets: Arc::clone(&resets),
        }));
        let mut events = Vec::new();
        for seq in 0..(MIN_SPEECH_FRAMES + idle_frames) as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, VadEvent::SpeechEnd { .. })),
            "the turn is still open"
        );
        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "idle reset must not run while a turn is open"
        );
    }

    #[test]
    fn idle_counter_restarts_after_a_speech_frame() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct ReplayScorer {
            values: Vec<f32>,
            index: usize,
            resets: Arc<AtomicUsize>,
        }
        impl ProbabilityScorer for ReplayScorer {
            fn score(&mut self, _frame: &AudioFrame) -> Result<f32> {
                let value = self.values[self.index];
                self.index += 1;
                Ok(value)
            }
            fn reset(&mut self) {
                self.resets.fetch_add(1, Ordering::SeqCst);
            }
        }

        // Two idle runs shorter than IDLE_RESET, split by a tentative speech
        // frame that never opens a turn. The speech frame clears the idle
        // counter, so the runs do not add up and no reset fires.
        let run = (IDLE_RESET.as_millis() as usize).div_ceil(32) - 1;
        let mut values = vec![0.0; run];
        values.push(0.9);
        values.extend(std::iter::repeat_n(0.0, run));
        let resets = Arc::new(AtomicUsize::new(0));
        let mut vad = SileroVad::with_scorer(Box::new(ReplayScorer {
            values,
            index: 0,
            resets: Arc::clone(&resets),
        }));
        for seq in 0..(run + 1 + run) as u64 {
            assert!(push_seq(&mut vad, seq).is_empty());
        }
        assert_eq!(
            resets.load(Ordering::SeqCst),
            0,
            "a speech frame must restart the idle counter"
        );
    }

    #[test]
    fn speech_shorter_than_min_duration_is_dropped() {
        let n = MIN_SPEECH_FRAMES - 1;
        let probabilities = std::iter::repeat_n(0.9, n)
            .chain(std::iter::once(0.1))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        for seq in 0..=n as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert!(events.is_empty());
        assert!(vad.flush().unwrap().is_empty());
    }

    #[test]
    fn flush_closes_active_speech_without_waiting_for_silence() {
        let mut vad =
            SileroVad::with_scorer(Box::new(ScriptedScorer(vec![0.8; MIN_SPEECH_FRAMES])));
        let mut events = Vec::new();
        for seq in 0..MIN_SPEECH_FRAMES as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
        let events = vad.flush().unwrap();
        assert!(
            matches!(events.as_slice(), [VadEvent::SpeechEnd { utterance }] if utterance.frames.len() == MIN_SPEECH_FRAMES)
        );
    }

    #[test]
    fn wrong_window_size_is_rejected_before_onnx_inference() {
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(vec![])));
        let short =
            AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, vec![1; 16]).unwrap();
        let err = vad.push_frame(short).unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
        assert!(err.to_string().contains("512 samples"));
    }

    #[test]
    fn name_matches_v0_and_reset_clears_turn() {
        struct ReplayScorer {
            values: Vec<f32>,
            index: usize,
        }
        impl ProbabilityScorer for ReplayScorer {
            fn score(&mut self, _frame: &AudioFrame) -> Result<f32> {
                let value = self.values[self.index];
                self.index += 1;
                Ok(value)
            }
            fn reset(&mut self) {
                self.index = 0;
            }
        }
        let mut vad = SileroVad::with_scorer(Box::new(ReplayScorer {
            values: vec![0.9; MIN_SPEECH_FRAMES],
            index: 0,
        }));
        assert_eq!(vad.name(), "silero");
        let mut events = Vec::new();
        for seq in 0..MIN_SPEECH_FRAMES as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
        vad.reset();
        events.clear();
        for seq in 0..MIN_SPEECH_FRAMES as u64 {
            events.extend(push_seq(&mut vad, seq));
        }
        assert_eq!(events, vec![VadEvent::SpeechStart { turn: TurnId(0) }]);
    }

    #[test]
    fn missing_model_file_is_a_provider_error() {
        let err = match SileroVad::from_model_path("/no/such/silero_vad.onnx") {
            Err(err) => err,
            Ok(_) => panic!("missing model path should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: "silero",
                ..
            }
        ));
    }

    #[test]
    fn from_cache_requires_silero_asset() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-vad-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = crate::models::ModelCache::new(
            root,
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match SileroVad::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("silero"));
    }
}
