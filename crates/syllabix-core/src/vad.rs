//! Silero ONNX voice-activity detection for the v0 16 kHz mono audio contract.

use std::path::Path;

use ort::session::Session;

use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Vad;
use crate::types::{AudioFrame, TurnId, Utterance, VadEvent, FRAME_SAMPLES};
use crate::Cancel;

/// Silero probability at or above which a frame begins or continues speech.
pub const SPEECH_THRESHOLD: f32 = 0.5;

/// Silence needed to close an utterance. Ten 32 ms frames give a 320 ms endpoint.
pub const END_SILENCE_FRAMES: usize = 10;

/// In-process Silero VAD backed by ONNX Runtime.
///
/// The detector accepts only the fixed v0 capture contract: 512 samples of
/// 16 kHz mono PCM. It retains Silero's recurrent state between frames.
pub struct SileroVad {
    scorer: Box<dyn ProbabilityScorer>,
    next_turn: u64,
    current: Option<(TurnId, Vec<AudioFrame>)>,
    silence_frames: usize,
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
            current: None,
            silence_frames: 0,
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

    fn start(&mut self, frame: AudioFrame) -> VadEvent {
        let turn = TurnId(self.next_turn);
        self.next_turn += 1;
        self.current = Some((turn, vec![frame]));
        VadEvent::SpeechStart { turn }
    }

    fn finish(&mut self) -> Option<VadEvent> {
        self.silence_frames = 0;
        let event = self
            .current
            .take()
            .map(|(turn, frames)| VadEvent::SpeechEnd {
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
            self.silence_frames = 0;
            if let Some((_, frames)) = self.current.as_mut() {
                frames.push(frame);
            } else {
                events.push(self.start(frame));
            }
        } else if self.current.is_some() {
            self.silence_frames += 1;
            if self.silence_frames >= END_SILENCE_FRAMES {
                if let Some(event) = self.finish() {
                    events.push(event);
                }
            }
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
        self.current = None;
        self.silence_frames = 0;
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
        // silero_vad.onnx (v6.2.1) scores 256-sample 8 kHz windows. v0 frames
        // are 512 samples at 16 kHz; pair-average to the native window.
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

    #[test]
    fn speech_and_silence_boundaries_follow_fixed_thresholds() {
        let probabilities = std::iter::once(0.2)
            .chain([0.5, 0.9])
            .chain(std::iter::repeat_n(0.1, END_SILENCE_FRAMES))
            .collect();
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(probabilities)));
        let mut events = Vec::new();
        for seq in 0..(END_SILENCE_FRAMES + 3) as u64 {
            events.extend(vad.push_frame(frame(seq)).unwrap());
        }

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], VadEvent::SpeechStart { turn: TurnId(0) });
        let VadEvent::SpeechEnd { utterance } = &events[1] else {
            panic!("expected a speech end");
        };
        assert_eq!(utterance.turn, TurnId(0));
        assert_eq!(utterance.frames.len(), 2);
        assert_eq!(utterance.frames[0].seq, 1);
        assert_eq!(utterance.frames[1].seq, 2);
    }

    #[test]
    fn flush_closes_active_speech_without_waiting_for_silence() {
        let mut vad = SileroVad::with_scorer(Box::new(ScriptedScorer(vec![0.8])));
        assert_eq!(
            vad.push_frame(frame(0)).unwrap(),
            vec![VadEvent::SpeechStart { turn: TurnId(0) }]
        );
        let events = vad.flush().unwrap();
        assert!(
            matches!(events.as_slice(), [VadEvent::SpeechEnd { utterance }] if utterance.frames.len() == 1)
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
            values: vec![0.9, 0.9],
            index: 0,
        }));
        assert_eq!(vad.name(), "silero");
        assert_eq!(
            vad.push_frame(frame(0)).unwrap(),
            vec![VadEvent::SpeechStart { turn: TurnId(0) }]
        );
        vad.reset();
        assert_eq!(
            vad.push_frame(frame(1)).unwrap(),
            vec![VadEvent::SpeechStart { turn: TurnId(0) }]
        );
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
