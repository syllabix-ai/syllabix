//! Silero ONNX voice-activity detection for the v0 16 kHz mono audio contract.

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
pub const SPEECH_THRESHOLD: f32 = 0.5;

/// A turn starts only after this much contiguous speech (four 32 ms frames).
pub const MIN_SPEECH: Duration = Duration::from_millis(100);

/// Silence needed to close an utterance (eleven 32 ms frames = 352 ms).
pub const END_SILENCE: Duration = Duration::from_millis(350);

/// v0 hop: 512 samples at 16 kHz.
pub const FRAME_DURATION: Duration = Duration::from_millis(32);

/// Frames of speech that meet [`MIN_SPEECH`].
pub const MIN_SPEECH_FRAMES: usize = 4;

/// Frames of silence that meet [`END_SILENCE`].
pub const END_SILENCE_FRAMES: usize = 11;

/// In-process Silero VAD backed by ONNX Runtime.
///
/// The detector accepts only the fixed v0 capture contract: 512 samples of
/// 16 kHz mono PCM. It pair-averages each frame to Silero's 8 kHz / 256-sample
/// window (the ONNX 512 / `sr=16000` branch scores ~0.003). It retains
/// Silero's recurrent state between frames.
pub struct SileroVad {
    scorer: Box<dyn ProbabilityScorer>,
    next_turn: u64,
    /// Speech frames for the open (or still-tentative) utterance.
    current: Vec<AudioFrame>,
    /// Set once [`MIN_SPEECH`] has elapsed; `None` means frames are tentative.
    active: Option<TurnId>,
    speech: Duration,
    silence: Duration,
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

    fn with_scorer(scorer: Box<dyn ProbabilityScorer>) -> Self {
        Self {
            scorer,
            next_turn: 0,
            current: Vec::new(),
            active: None,
            speech: Duration::ZERO,
            silence: Duration::ZERO,
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

    /// Test helper: Silero probability for one v0 frame.
    pub fn debug_probability(&mut self, frame: &AudioFrame) -> Result<f32> {
        self.probability(frame)
    }

    fn promote(&mut self) -> Option<VadEvent> {
        if self.active.is_some() || self.speech < MIN_SPEECH {
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
        let frames = std::mem::take(&mut self.current);
        let event = self.active.take().map(|turn| VadEvent::SpeechEnd {
            utterance: Utterance { turn, frames },
        });
        if event.is_some() {
            // Recurrent hangover otherwise suppresses the next user turn
            // (identical or similar speech stays below 0.5).
            self.scorer.reset();
        }
        event
    }
}

impl Vad for SileroVad {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().vad.as_str()
    }

    fn push_frame(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>> {
        let speech = self.probability(&frame)? >= SPEECH_THRESHOLD;
        let mut events = Vec::new();

        if speech {
            self.silence = Duration::ZERO;
            self.current.push(frame);
            self.speech += FRAME_DURATION;
            if let Some(event) = self.promote() {
                events.push(event);
            }
        } else if self.active.is_some() {
            self.silence += FRAME_DURATION;
            if self.silence >= END_SILENCE {
                if let Some(event) = self.finish() {
                    events.push(event);
                }
            }
        } else if !self.current.is_empty() {
            // Tentative speech never reached min duration; drop it.
            self.current.clear();
            self.speech = Duration::ZERO;
            self.silence = Duration::ZERO;
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
        self.current.clear();
        self.active = None;
        self.speech = Duration::ZERO;
        self.silence = Duration::ZERO;
    }
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
        // across a pause longer than 350 ms hangover. Pair-average to 8 kHz.
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
                    "10 silence frames are under {END_SILENCE:?}"
                );
            }
        }

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], VadEvent::SpeechStart { turn: TurnId(0) });
        let VadEvent::SpeechEnd { utterance } = &events[1] else {
            panic!("expected a speech end");
        };
        assert_eq!(utterance.turn, TurnId(0));
        assert_eq!(utterance.frames.len(), MIN_SPEECH_FRAMES);
        assert_eq!(utterance.frames[0].seq, 1);
        assert_eq!(
            utterance.frames[MIN_SPEECH_FRAMES - 1].seq,
            MIN_SPEECH_FRAMES as u64
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
