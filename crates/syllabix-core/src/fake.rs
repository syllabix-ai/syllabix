//! In-memory providers for deterministic pipeline tests.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::providers::{AudioSink, Llm, Stt, Tts, Vad};
use crate::turn_debug::PlaybackWatch;
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

/// Energy-based VAD for deterministic in-memory tests.
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
            // No preroll buffer here; the seed is the promote frame itself so
            // streaming STT still starts from the same onset as the utterance.
            events.push(VadEvent::SpeechStart {
                turn,
                seed: vec![frame.clone()],
            });
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
            language: BuiltinDefaults::v0().language.to_string(),
        })
    }
}

/// Returns queued hypotheses in order. Used to test empty-STT skip.
pub struct ScriptedStt {
    texts: Vec<String>,
    index: usize,
    language: String,
}

impl ScriptedStt {
    /// One hypothesis per completed VAD utterance, in order.
    pub fn new(texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            texts: texts.into_iter().map(Into::into).collect(),
            index: 0,
            language: BuiltinDefaults::v0().language.to_string(),
        }
    }

    /// Language stamped onto every transcript.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }
}

impl Stt for ScriptedStt {
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
        let text = self
            .texts
            .get(self.index)
            .cloned()
            .ok_or_else(|| Error::Provider {
                provider: self.name(),
                message: "scripted stt exhausted".into(),
            })?;
        self.index += 1;
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: self.language.clone(),
        })
    }
}

/// First `transcribe` returns a provider error; later calls use [`FakeStt`].
pub struct FailOnceStt {
    inner: FakeStt,
    remaining: u32,
}

impl Default for FailOnceStt {
    fn default() -> Self {
        Self {
            inner: FakeStt,
            remaining: 1,
        }
    }
}

impl Stt for FailOnceStt {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if self.remaining > 0 {
            self.remaining -= 1;
            return Err(Error::Provider {
                provider: self.name(),
                message: "scripted stt failure".into(),
            });
        }
        self.inner.transcribe(utterance, cancel)
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

/// First `generate` returns a provider error; later calls use [`FakeLlm`].
pub struct FailOnceLlm {
    inner: FakeLlm,
    remaining: u32,
}

impl FailOnceLlm {
    /// Fail the first generate, then echo.
    pub fn new() -> Self {
        Self {
            inner: FakeLlm::new(),
            remaining: 1,
        }
    }
}

impl Default for FailOnceLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl Llm for FailOnceLlm {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn generate(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        if self.remaining > 0 {
            self.remaining -= 1;
            return Err(Error::Provider {
                provider: self.name(),
                message: "scripted llm failure".into(),
            });
        }
        self.inner.generate(history, user, cancel, on_token)
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
    ) -> Result<Vec<SynthesizedAudio>> {
        if cancel.is_stale(token.generation) {
            return Err(Error::Cancelled);
        }
        let mut samples: Vec<i16> = token.text.bytes().map(i16::from).collect();
        if samples.is_empty() {
            samples.push(1);
        }
        Ok(vec![SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index: token.index,
            samples,
            is_last: token.is_last,
        }])
    }
}

/// Fails every chunk of the first turn, then uses [`FakeTts`].
pub struct FailOnceTts {
    inner: FakeTts,
    failed_turn: Option<TurnId>,
}

impl Default for FailOnceTts {
    fn default() -> Self {
        Self {
            inner: FakeTts,
            failed_turn: None,
        }
    }
}

impl Tts for FailOnceTts {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        if self.failed_turn.is_none() || self.failed_turn == Some(token.turn) {
            self.failed_turn = Some(token.turn);
            return Err(Error::Provider {
                provider: self.name(),
                message: "scripted tts failure".into(),
            });
        }
        self.inner.synthesize_chunk(token, cancel)
    }
}

/// Collects played audio for assertions. Does not open a device.
#[derive(Debug, Default)]
pub struct CollectingSink {
    /// Chunks in the order `play` was called, after stale generations were skipped.
    pub chunks: Vec<SynthesizedAudio>,
    /// Times [`AudioSink::interrupt`] ran (barge-in flush).
    pub interrupted: usize,
    watch: Option<PlaybackWatch>,
}

impl CollectingSink {
    /// Collect audio while reporting playback edges to a timeline watch —
    /// the fixture stand-in for the native speaker callback.
    pub fn with_playback_watch(watch: Option<PlaybackWatch>) -> Self {
        Self {
            chunks: Vec::new(),
            interrupted: 0,
            watch,
        }
    }
}

impl AudioSink for CollectingSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            return Ok(());
        }
        if let Some(watch) = &self.watch {
            // Synchronous "device": consuming the chunk is one callback tick;
            // the final chunk drains immediately after.
            watch.begin_turn(audio.turn);
            watch.on_callback(audio.samples.len());
            if audio.is_last {
                watch.on_callback(0);
            }
        }
        self.chunks.push(audio);
        Ok(())
    }

    fn interrupt(&mut self) {
        self.interrupted += 1;
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
    use crate::providers::{AudioSink, Stt, Vad};

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
                VadEvent::SpeechStart { turn, .. } => Some(*turn),
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

    #[test]
    fn fake_stt_rejects_empty_utterance_and_shutdown() {
        let mut stt = FakeStt;
        let cancel = Cancel::new();
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(0),
                    frames: vec![],
                },
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
        cancel.shutdown();
        let frames = scripted_frames(1, 1, 0);
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames,
                },
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn scripted_stt_returns_queued_hypotheses() {
        let mut stt = ScriptedStt::new(["", " hello "]);
        let frames = scripted_frames(1, 1, 0);
        let first = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(0),
                    frames: frames.clone(),
                },
                &Cancel::new(),
            )
            .unwrap();
        assert_eq!(first.text, "");
        let second = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames,
                },
                &Cancel::new(),
            )
            .unwrap();
        assert_eq!(second.text, " hello ");
    }

    #[test]
    fn collecting_sink_records_chunks_until_cancel() {
        let mut sink = CollectingSink::default();
        let cancel = Cancel::new();
        let audio = SynthesizedAudio {
            turn: TurnId(0),
            generation: cancel.generation(),
            index: 0,
            samples: vec![1, 2],
            is_last: true,
        };
        sink.play(audio.clone(), &cancel).unwrap();
        assert_eq!(sink.chunks.len(), 1);
        sink.interrupt();
        assert_eq!(sink.interrupted, 1);
        cancel.shutdown();
        assert!(matches!(sink.play(audio, &cancel), Err(Error::Cancelled)));
    }
}
