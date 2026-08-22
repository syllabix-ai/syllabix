//! Audio-frame, utterance, transcript, token-stream, and synthesized-audio types.

use crate::error::{Error, Result};
use std::time::Duration;

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
    /// Pre-AEC PCM for `--turn-debug`, same length as `samples` when present.
    pub capture_pcm: Option<Vec<i16>>,
}

impl AudioFrame {
    /// Build a frame after checking the v0 PCM contract.
    pub fn new(seq: u64, sample_rate_hz: u32, channels: u16, samples: Vec<i16>) -> Result<Self> {
        let frame = Self {
            seq,
            sample_rate_hz,
            channels,
            samples,
            capture_pcm: None,
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
    /// 200 ms post-AEC preroll (when available) plus every frame after speech-start
    /// until end-of-utterance, including below-threshold dips and the 350 ms hangover.
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

/// Speech-start / speech-stop events. `--barge-in` keys off `SpeechStart`.
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
    /// Effective STT language code: the configured id, or the code detected
    /// from this utterance when yaml selected `auto`.
    pub language: String,
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
    /// Stage clocks for the TUI latency line.
    pub timings: TurnTimings,
}

/// Per-turn clocks shown in the `run` TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TurnTimings {
    /// Utterance ready → transcript.
    pub stt: Duration,
    /// LLM start → first token.
    pub ttft: Duration,
    /// LLM start → first audio chunk played.
    pub ttfb: Duration,
    /// Utterance ready → last audio chunk played.
    pub total: Duration,
}

impl TurnTimings {
    /// One-line TUI footer: `STT 120ms  TTFT 80ms  TTFB 210ms  total 1.10s`.
    pub fn format_line(&self) -> String {
        format!(
            "STT {}  TTFT {}  TTFB {}  total {}",
            format_duration(self.stt),
            format_duration(self.ttft),
            format_duration(self.ttfb),
            format_duration(self.total)
        )
    }
}

fn format_duration(duration: Duration) -> String {
    let millis = duration.as_secs_f64() * 1000.0;
    if millis < 1000.0 {
        format!("{millis:.0}ms")
    } else {
        format!("{:.2}s", duration.as_secs_f64())
    }
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

    #[test]
    fn timings_line_uses_ms_under_one_second() {
        let line = TurnTimings {
            stt: Duration::from_millis(12),
            ttft: Duration::from_millis(80),
            ttfb: Duration::from_millis(210),
            total: Duration::from_millis(1100),
        }
        .format_line();
        assert_eq!(line, "STT 12ms  TTFT 80ms  TTFB 210ms  total 1.10s");
    }
}
