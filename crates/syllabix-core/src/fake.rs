//! In-memory providers. Same names as the blessed v0 stack; no alternate backends.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::providers::{AudioSink, Llm, Stt, Tts, Vad};
use crate::types::{
    AudioFrame, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId, Utterance, VadEvent,
    DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};

/// Recorded LLM invocation for history-order tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmCall {
    /// Prior completed turns.
    pub history_len: usize,
    /// User texts in `history`, in order.
    pub history_user_texts: Vec<String>,
    /// Current user text.
    pub user_text: String,
}

/// Energy-based VAD used until Silero is wired.
pub struct FakeVad {
    next_turn: u64,
    current: Option<(TurnId, Vec<AudioFrame>)>,
}

impl FakeVad {
    /// Idle detector.
    pub fn new() -> Self {
        Self {
            next_turn: 0,
            current: None,
        }
    }
}

impl Default for FakeVad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad for FakeVad {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().vad.as_str()
    }

    fn push_frame(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>> {
        frame.validate()?;
        if frame.has_energy() {
            self.push_speech(frame)
        } else {
            Ok(self.end_speech().into_iter().collect())
        }
    }

    fn flush(&mut self) -> Result<Vec<VadEvent>> {
        Ok(self.end_speech().into_iter().collect())
    }
}

impl FakeVad {
    fn push_speech(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>> {
        let mut events = Vec::new();
        if self.current.is_none() {
            let turn = TurnId(self.next_turn);
            self.next_turn += 1;
            events.push(VadEvent::SpeechStart { turn });
            self.current = Some((turn, Vec::new()));
        }
        if let Some((_, frames)) = self.current.as_mut() {
            frames.push(frame);
        }
        Ok(events)
    }

    fn end_speech(&mut self) -> Option<VadEvent> {
        self.current
            .take()
            .map(|(turn, frames)| VadEvent::SpeechEnd {
                utterance: Utterance { turn, frames },
            })
    }
}

/// Maps an utterance to `turn-{id:03}`. Kept for the 30-turn in-memory tests.
pub struct FakeStt;

impl Default for FakeStt {
    fn default() -> Self {
        Self
    }
}

impl Stt for FakeStt {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().stt.as_str()
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no frames".into(),
            });
        }
        Ok(Transcript {
            turn: utterance.turn,
            text: format!("turn-{:03}", utterance.turn.0),
        })
    }
}

/// Echo LLM: streams `echo:` + user text as individual characters, then a terminator.
pub struct FakeLlm {
    delay_per_token: Duration,
    calls: Arc<Mutex<Vec<LlmCall>>>,
}

impl FakeLlm {
    /// Instant tokens.
    pub fn new() -> Self {
        Self {
            delay_per_token: Duration::ZERO,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Slow tokens so tests can cancel mid-generation.
    pub fn with_delay(delay_per_token: Duration) -> Self {
        Self {
            delay_per_token,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Shared call log. Clone the `Arc` before moving the LLM into the pipeline.
    pub fn call_log(&self) -> Arc<Mutex<Vec<LlmCall>>> {
        Arc::clone(&self.calls)
    }
}

impl Default for FakeLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl Llm for FakeLlm {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().llm.as_str()
    }

    fn generate(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        let generation = cancel.generation();
        {
            let mut calls = self.calls.lock().expect("llm call log");
            calls.push(LlmCall {
                history_len: history.len(),
                history_user_texts: history.iter().map(|t| t.user.text.clone()).collect(),
                user_text: user.text.clone(),
            });
        }

        let reply = format!("echo:{}", user.text);
        let mut chars: Vec<String> = reply.chars().map(|c| c.to_string()).collect();
        if chars.is_empty() {
            chars.push(String::new());
        }
        let last = chars.len() - 1;
        for (index, text) in chars.into_iter().enumerate() {
            if cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            if !self.delay_per_token.is_zero() {
                std::thread::sleep(self.delay_per_token);
            }
            if cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            on_token(TokenChunk {
                turn: user.turn,
                generation,
                index: index as u32,
                text,
                is_last: index == last,
            })?;
        }
        Ok(())
    }
}

/// Each token becomes a tiny PCM chunk (UTF-8 bytes as samples).
pub struct FakeTts;

impl Default for FakeTts {
    fn default() -> Self {
        Self
    }
}

impl Tts for FakeTts {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().tts.as_str()
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<SynthesizedAudio> {
        if cancel.is_stale(token.generation) {
            return Err(Error::Cancelled);
        }
        let mut samples: Vec<i16> = token.text.bytes().map(i16::from).collect();
        if samples.is_empty() {
            samples.push(1);
        }
        Ok(SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index: token.index,
            samples,
            is_last: token.is_last,
        })
    }
}

/// Collects played audio for assertions. Does not open a device.
#[derive(Debug, Default)]
pub struct CollectingSink {
    /// Chunks in the order `play` was called, after stale generations were skipped.
    pub chunks: Vec<SynthesizedAudio>,
}

impl AudioSink for CollectingSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            return Ok(());
        }
        self.chunks.push(audio);
        Ok(())
    }
}

/// Build scripted mic frames: `turns` × (`speech_frames` of energy + `silence_frames` of zeros).
pub fn scripted_frames(
    turns: usize,
    speech_frames: usize,
    silence_frames: usize,
) -> Vec<AudioFrame> {
    assert!(
        speech_frames > 0,
        "each turn needs at least one speech frame"
    );
    let mut frames = Vec::with_capacity(turns * (speech_frames + silence_frames));
    let mut seq = 0_u64;
    for turn in 0..turns {
        let energy = i16::try_from(turn + 1).unwrap_or(i16::MAX);
        for _ in 0..speech_frames {
            frames.push(
                AudioFrame::new(
                    seq,
                    DEFAULT_SAMPLE_RATE_HZ,
                    DEFAULT_CHANNELS,
                    vec![energy; FRAME_SAMPLES],
                )
                .expect("scripted speech frame"),
            );
            seq += 1;
        }
        for _ in 0..silence_frames {
            frames.push(
                AudioFrame::new(
                    seq,
                    DEFAULT_SAMPLE_RATE_HZ,
                    DEFAULT_CHANNELS,
                    vec![0; FRAME_SAMPLES],
                )
                .expect("scripted silence frame"),
            );
            seq += 1;
        }
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Vad;

    #[test]
    fn vad_emits_start_and_end_per_turn() {
        let mut vad = FakeVad::new();
        let frames = scripted_frames(2, 2, 1);
        let mut events = Vec::new();
        for frame in frames {
            events.extend(vad.push_frame(frame).unwrap());
        }
        events.extend(vad.flush().unwrap());
        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                VadEvent::SpeechStart { turn } => Some(*turn),
                VadEvent::SpeechEnd { .. } => None,
            })
            .collect();
        let ends: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                VadEvent::SpeechEnd { utterance } => Some(utterance.turn),
                VadEvent::SpeechStart { .. } => None,
            })
            .collect();
        assert_eq!(starts, vec![TurnId(0), TurnId(1)]);
        assert_eq!(ends, vec![TurnId(0), TurnId(1)]);
    }

    #[test]
    fn fake_names_match_builtin_stack() {
        let d = BuiltinDefaults::v0();
        assert_eq!(FakeVad::new().name(), d.vad.as_str());
        assert_eq!(FakeStt.name(), d.stt.as_str());
        assert_eq!(FakeLlm::new().name(), d.llm.as_str());
        assert_eq!(FakeTts.name(), d.tts.as_str());
    }

    #[test]
    fn stt_labels_turns_in_order() {
        let mut stt = FakeStt;
        let cancel = Cancel::new();
        let frames = scripted_frames(1, 1, 0);
        let utterance = Utterance {
            turn: TurnId(7),
            frames,
        };
        let t = stt.transcribe(&utterance, &cancel).unwrap();
        assert_eq!(t.text, "turn-007");
    }
}
