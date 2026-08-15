//! Audio-frame, utterance, transcript, token-stream, and synthesized-audio types.

use crate::error::{Error, Result};

/// v0 capture/playback rate. Whisper, Silero, and Kokoro adapters share it.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;

/// v0 is mono. Native I/O converts device layouts to this.
pub const DEFAULT_CHANNELS: u16 = 1;

/// ~32 ms at 16 kHz. Matches a typical Silero window without coupling to ONNX yet.
pub const FRAME_SAMPLES: usize = 512;

/// Identifies one user turn from VAD speech-start through assistant playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnId(pub u64);

/// Monotonic id for an assistant generation. Bumped on barge-in cancel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenerationId(pub u64);

/// One PCM frame from capture (or a fake source).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// Strictly increasing capture sequence.
    pub seq: u64,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count. v0 frames are mono.
    pub channels: u16,
    /// Interleaved PCM16 samples.
    pub samples: Vec<i16>,
}

impl AudioFrame {
    /// Build a frame after checking the v0 PCM contract.
    pub fn new(seq: u64, sample_rate_hz: u32, channels: u16, samples: Vec<i16>) -> Result<Self> {
        let frame = Self {
            seq,
            sample_rate_hz,
            channels,
            samples,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Reject empty buffers and non-mono / wrong-rate frames.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate_hz != DEFAULT_SAMPLE_RATE_HZ {
            return Err(Error::InvalidAudio {
                message: format!(
                    "sample rate {} Hz is not the v0 rate {DEFAULT_SAMPLE_RATE_HZ}",
                    self.sample_rate_hz
                ),
            });
        }
        if self.channels != DEFAULT_CHANNELS {
            return Err(Error::InvalidAudio {
                message: format!("channel count {} is not the v0 mono layout", self.channels),
            });
        }
        if self.samples.is_empty() {
            return Err(Error::InvalidAudio {
                message: "audio frame has no samples".into(),
            });
        }
        Ok(())
    }

    /// True when any sample is non-zero. Fake VAD uses this as speech.
    pub fn has_energy(&self) -> bool {
        self.samples.iter().any(|s| *s != 0)
    }
}

/// Completed user speech segment emitted by VAD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utterance {
    /// Turn assigned at speech-start.
    pub turn: TurnId,
    /// Frames from speech-start through the last speech frame (silence not included).
    pub frames: Vec<AudioFrame>,
}

impl Utterance {
    /// Concatenate PCM from every frame, preserving frame order.
    pub fn pcm(&self) -> Vec<i16> {
        self.frames
            .iter()
            .flat_map(|frame| frame.samples.iter().copied())
            .collect()
    }
}

/// Speech-start / speech-stop events. Barge-in later keys off `SpeechStart`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadEvent {
    /// Mic crossed into speech; a new turn id is reserved.
    SpeechStart {
        /// New turn.
        turn: TurnId,
    },
    /// Mic returned to silence (or flush) and the utterance is complete.
    SpeechEnd {
        /// Speech frames for this turn.
        utterance: Utterance,
    },
}

/// STT result for one user turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Turn this transcript belongs to.
    pub turn: TurnId,
    /// Recognized text. Empty string is allowed only if a provider emits it.
    pub text: String,
}

/// One streamed LLM token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk {
    /// Turn this token belongs to.
    pub turn: TurnId,
    /// Generation that produced the token. Stale generations are flushed.
    pub generation: GenerationId,
    /// Zero-based index within this generation.
    pub index: u32,
    /// Token text, including spaces when the fake/real tokenizer emits them.
    pub text: String,
    /// True on the last token of this generation.
    pub is_last: bool,
}

/// One streamed TTS audio chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizedAudio {
    /// Turn this chunk belongs to.
    pub turn: TurnId,
    /// Generation that produced the chunk. Stale generations are flushed.
    pub generation: GenerationId,
    /// Zero-based index within this generation.
    pub index: u32,
    /// PCM16 mono at [`DEFAULT_SAMPLE_RATE_HZ`].
    pub samples: Vec<i16>,
    /// True on the last chunk of this generation.
    pub is_last: bool,
}

/// Rolling history entry passed into the LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryTurn {
    /// User transcript for the turn.
    pub user: Transcript,
    /// Assistant text already spoken (or generated) for that turn.
    pub assistant: String,
}

/// One finished user+assistant cycle collected by the in-memory loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedTurn {
    /// Turn id.
    pub id: TurnId,
    /// User transcript.
    pub user_text: String,
    /// Concatenated assistant tokens.
    pub assistant_text: String,
    /// Number of LLM tokens emitted (including the last marker token).
    pub token_count: usize,
    /// Number of TTS chunks played (including the last marker chunk).
    pub audio_chunks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_rate_and_empty_samples() {
        let err = AudioFrame::new(0, 8_000, 1, vec![1]).unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
        let err = AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, 2, vec![1, 2]).unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
        let err = AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, 1, vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
    }

    #[test]
    fn energy_detects_speech_vs_silence() {
        let speech = AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, 1, vec![0, 3, 0]).unwrap();
        let silence = AudioFrame::new(1, DEFAULT_SAMPLE_RATE_HZ, 1, vec![0, 0]).unwrap();
        assert!(speech.has_energy());
        assert!(!silence.has_energy());
    }

    #[test]
    fn utterance_pcm_preserves_frame_order() {
        let utterance = Utterance {
            turn: TurnId(0),
            frames: vec![
                AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, 1, vec![1, 2]).unwrap(),
                AudioFrame::new(1, DEFAULT_SAMPLE_RATE_HZ, 1, vec![3]).unwrap(),
            ],
        };
        assert_eq!(utterance.pcm(), vec![1, 2, 3]);
    }
}
