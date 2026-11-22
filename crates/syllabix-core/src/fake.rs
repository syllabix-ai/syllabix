//! In-memory providers for deterministic pipeline tests.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::providers::{AudioSink, Llm, Stt, Tts, Vad};
use crate::s2s::S2sEngine;
use crate::turn_debug::PlaybackWatch;
use crate::types::{
    AudioFrame, GenerationId, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId,
    Utterance, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
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
        crate::defaults::LlmProvider::Local.as_str()
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeAssistPhase {
    Idle,
    Audio,
    Text,
}

#[derive(Clone, Debug)]
pub struct FakeS2sCloseRecord {
    pub phase: FakeAssistPhase,
    pub marker: bool,
    pub closed: bool,
}

type FakeS2sSeenLog = Arc<Mutex<Vec<(TurnId, Vec<TurnId>)>>>;

/// Stable assistant text streamed by [`FakeS2s`].
pub const FAKE_S2S_REPLY: &str = "hello from fake s2s";

/// Stable user transcript from [`FakeS2s::transcribe`].
pub const FAKE_S2S_TRANSCRIPT: &str = "please pass the stew for dinner";

const FAKE_S2S_TEXT_PIECES: &[&str] = &["hello ", "from ", "fake ", "s2s"];

/// Recorded native S2S operations for tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeS2sOp {
    Generate(TurnId),
    Transcribe {
        turn: TurnId,
        language: Option<String>,
    },
    Speak {
        turn: TurnId,
        text: String,
        language: Option<String>,
        voice: Option<String>,
    },
}

/// Delayed echo S2S: streams the user utterance PCM back as assistant audio.
pub struct FakeS2s {
    delay: Duration,
    audio_delay: Duration,
    text_delay: Duration,
    call_log: Arc<Mutex<Vec<TurnId>>>,
    history: Arc<Mutex<Vec<TurnId>>>,
    seen: FakeS2sSeenLog,
    close_log: Arc<Mutex<Vec<(TurnId, FakeS2sCloseRecord)>>>,
    ops_log: Arc<Mutex<Vec<FakeS2sOp>>>,
    last_text: String,
    speak_sample_cap: Option<usize>,
}

impl FakeS2s {
    /// Instant echo after VAD commit.
    pub fn new() -> Self {
        Self {
            delay: Duration::ZERO,
            audio_delay: Duration::ZERO,
            text_delay: Duration::ZERO,
            call_log: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
            close_log: Arc::new(Mutex::new(Vec::new())),
            ops_log: Arc::new(Mutex::new(Vec::new())),
            last_text: String::new(),
            speak_sample_cap: None,
        }
    }

    /// Slow generation so tests can cancel mid-wait or during playback.
    pub fn with_delay(delay: Duration) -> Self {
        Self {
            delay,
            audio_delay: Duration::ZERO,
            text_delay: Duration::ZERO,
            call_log: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
            close_log: Arc::new(Mutex::new(Vec::new())),
            ops_log: Arc::new(Mutex::new(Vec::new())),
            last_text: String::new(),
            speak_sample_cap: None,
        }
    }

    /// Audio-shaped delay then text-shaped delay (for barge-in phase tests).
    pub fn with_staged_delay(audio: Duration, text: Duration) -> Self {
        Self {
            delay: audio.saturating_add(text),
            audio_delay: audio,
            text_delay: text,
            call_log: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(Vec::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
            close_log: Arc::new(Mutex::new(Vec::new())),
            ops_log: Arc::new(Mutex::new(Vec::new())),
            last_text: String::new(),
            speak_sample_cap: None,
        }
    }

    /// Truncate spoken PCM to `cap` samples, matching a capped native backend.
    pub fn with_speak_sample_cap(mut self, cap: usize) -> Self {
        self.speak_sample_cap = Some(cap);
        self
    }

    /// Shared generate log. Clone the `Arc` before moving the engine into the pipeline.
    pub fn call_log(&self) -> Arc<Mutex<Vec<TurnId>>> {
        Arc::clone(&self.call_log)
    }

    pub fn seen_at_generate(&self) -> FakeS2sSeenLog {
        Arc::clone(&self.seen)
    }

    pub fn close_log(&self) -> Arc<Mutex<Vec<(TurnId, FakeS2sCloseRecord)>>> {
        Arc::clone(&self.close_log)
    }

    pub fn ops_log(&self) -> Arc<Mutex<Vec<FakeS2sOp>>> {
        Arc::clone(&self.ops_log)
    }

    fn record_close(&self, turn: TurnId, phase: FakeAssistPhase) {
        self.close_log.lock().expect("close log").push((
            turn,
            FakeS2sCloseRecord {
                phase,
                marker: true,
                closed: true,
            },
        ));
        self.history.lock().expect("s2s history").push(turn);
    }

    fn wait_phase(
        &self,
        generation: crate::types::GenerationId,
        cancel: &Cancel,
        mut remaining: Duration,
    ) -> Result<(), Error> {
        let step = Duration::from_millis(1);
        while remaining > Duration::ZERO {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            let slice = step.min(remaining);
            std::thread::sleep(slice);
            remaining = remaining.saturating_sub(slice);
        }
        Ok(())
    }

    fn emit_text_pieces(
        &mut self,
        on_text: &mut dyn FnMut(&str, bool) -> Result<()>,
        from_index: usize,
        to_index: usize,
        mark_last: bool,
    ) -> Result<(), Error> {
        let pieces = &FAKE_S2S_TEXT_PIECES[from_index..to_index];
        for (i, piece) in pieces.iter().enumerate() {
            self.last_text.push_str(piece);
            let is_last =
                mark_last && i + 1 == pieces.len() && to_index == FAKE_S2S_TEXT_PIECES.len();
            on_text(piece, is_last)?;
        }
        Ok(())
    }
}

impl Default for FakeS2s {
    fn default() -> Self {
        Self::new()
    }
}

impl S2sEngine for FakeS2s {
    fn name(&self) -> &'static str {
        "fake-s2s"
    }

    fn last_text(&self) -> Option<&str> {
        if self.last_text.is_empty() {
            None
        } else {
            Some(&self.last_text)
        }
    }

    fn speak_sample_cap(&self) -> Option<usize> {
        self.speak_sample_cap
    }

    fn generate(
        &mut self,
        utterance: &Utterance,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
        on_text: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no frames".into(),
            });
        }
        self.last_text.clear();
        let generation = cancel.generation();
        let history_at_start = {
            let history = self.history.lock().expect("s2s history");
            history.clone()
        };
        {
            let mut log = self.call_log.lock().expect("s2s call log");
            log.push(utterance.turn);
            let mut seen = self.seen.lock().expect("s2s seen log");
            seen.push((utterance.turn, history_at_start));
            self.ops_log
                .lock()
                .expect("s2s ops log")
                .push(FakeS2sOp::Generate(utterance.turn));
        }

        let text_wait = self.text_delay;
        let text_first = self.audio_delay.is_zero() && text_wait > Duration::ZERO;
        let audio_wait = if self.audio_delay > Duration::ZERO {
            self.audio_delay
        } else if text_first {
            Duration::ZERO
        } else {
            self.delay
        };

        if text_first {
            self.emit_text_pieces(on_text, 0, 1, false)?;
            if let Err(Error::Cancelled) = self.wait_phase(generation, cancel, text_wait) {
                self.record_close(utterance.turn, FakeAssistPhase::Text);
                return Err(Error::Cancelled);
            }
            self.emit_text_pieces(on_text, 1, FAKE_S2S_TEXT_PIECES.len(), true)?;
        } else {
            if let Err(Error::Cancelled) = self.wait_phase(generation, cancel, audio_wait) {
                return Err(Error::Cancelled);
            }

            let pcm = utterance.pcm();
            if text_wait > Duration::ZERO && !pcm.is_empty() {
                on_audio(SynthesizedAudio {
                    turn: utterance.turn,
                    generation,
                    index: 0,
                    samples: pcm[..pcm.len().min(4)].to_vec(),
                    is_last: false,
                })?;
            }

            if text_wait > Duration::ZERO {
                if let Err(Error::Cancelled) = self.wait_phase(generation, cancel, text_wait) {
                    let phase = if text_wait > audio_wait {
                        FakeAssistPhase::Text
                    } else {
                        FakeAssistPhase::Audio
                    };
                    self.record_close(utterance.turn, phase);
                    return Err(Error::Cancelled);
                }
            }
        }

        if !text_first {
            self.emit_text_pieces(on_text, 0, FAKE_S2S_TEXT_PIECES.len(), true)?;
        }

        let pcm = utterance.pcm();

        if cancel.is_shutdown() || cancel.is_stale(generation) {
            if text_wait > Duration::ZERO {
                self.record_close(utterance.turn, FakeAssistPhase::Text);
            }
            return Err(Error::Cancelled);
        }

        on_audio(SynthesizedAudio {
            turn: utterance.turn,
            generation,
            index: if text_first || text_wait.is_zero() {
                0
            } else {
                1
            },
            samples: pcm,
            is_last: true,
        })?;
        self.history
            .lock()
            .expect("s2s history")
            .push(utterance.turn);
        debug_assert_eq!(self.last_text, FAKE_S2S_REPLY);
        Ok(())
    }

    fn transcribe(
        &mut self,
        utterance: &Utterance,
        language: Option<&str>,
        cancel: &Cancel,
    ) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no frames".into(),
            });
        }
        let generation = cancel.generation();
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        self.ops_log
            .lock()
            .expect("s2s ops log")
            .push(FakeS2sOp::Transcribe {
                turn: utterance.turn,
                language: language.map(str::to_string),
            });
        let asr_wait = if self.text_delay > Duration::ZERO {
            self.text_delay
        } else if self.audio_delay > Duration::ZERO {
            Duration::ZERO
        } else {
            self.delay
        };
        if asr_wait > Duration::ZERO {
            self.wait_phase(generation, cancel, asr_wait)?;
        }
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        Ok(FAKE_S2S_TRANSCRIPT.into())
    }

    #[allow(clippy::too_many_arguments)]
    fn speak(
        &mut self,
        turn: TurnId,
        generation: GenerationId,
        text: &str,
        language: Option<&str>,
        voice: Option<&str>,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        self.ops_log
            .lock()
            .expect("s2s ops log")
            .push(FakeS2sOp::Speak {
                turn,
                text: text.into(),
                language: language.map(str::to_string),
                voice: voice.map(str::to_string),
            });
        if self.delay > Duration::ZERO {
            self.wait_phase(generation, cancel, self.delay)?;
        }
        let samples: Vec<i16> = text.bytes().map(i16::from).collect();
        let mut pcm = if samples.is_empty() { vec![1] } else { samples };
        if let Some(cap) = self.speak_sample_cap {
            pcm.truncate(cap);
        }
        on_audio(SynthesizedAudio {
            turn,
            generation,
            index: 0,
            samples: pcm,
            is_last: true,
        })
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
    use crate::providers::{AudioSink, Stt, Tts, Vad};

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
        assert!(FakeTts.backend_id().is_none());
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
    fn fake_s2s_name_is_fake_s2s() {
        assert_eq!(FakeS2s::new().name(), "fake-s2s");
    }

    #[test]
    fn fake_s2s_streams_text_deltas_matching_last_text() {
        let mut engine = FakeS2s::new();
        let cancel = Cancel::new();
        let utterance = Utterance {
            turn: TurnId(0),
            frames: scripted_frames(1, 1, 0),
        };
        let mut pieces = Vec::new();
        engine
            .generate(
                &utterance,
                &cancel,
                &mut |_| Ok(()),
                &mut |text, is_last| {
                    pieces.push((text.to_string(), is_last));
                    Ok(())
                },
            )
            .unwrap();
        let joined: String = pieces.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(joined, FAKE_S2S_REPLY);
        assert_eq!(engine.last_text(), Some(FAKE_S2S_REPLY));
        assert!(pieces.last().is_some_and(|(_, last)| *last));
        assert_eq!(pieces.iter().filter(|(_, last)| *last).count(), 1);
    }

    #[test]
    fn fake_s2s_echoes_utterance_pcm_with_last_chunk() {
        let mut engine = FakeS2s::new();
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let utterance = Utterance {
            turn: TurnId(2),
            frames: scripted_frames(1, 2, 0),
        };
        let expected = utterance.pcm();
        let mut chunks = Vec::new();
        engine
            .generate(
                &utterance,
                &cancel,
                &mut |audio| {
                    chunks.push(audio);
                    Ok(())
                },
                &mut |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples, expected);
        assert!(chunks[0].is_last);
        assert_eq!(chunks[0].turn, TurnId(2));
        assert_eq!(chunks[0].generation, generation);
    }

    #[test]
    fn fake_s2s_transcribe_with_delay_cancellable_mid_wait() {
        use std::sync::Arc;
        use std::thread;

        let cancel = Arc::new(Cancel::new());
        let cancel_worker = Arc::clone(&cancel);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(15));
            cancel_worker.cancel_generation();
        });
        let mut engine = FakeS2s::with_delay(Duration::from_millis(150));
        let utterance = Utterance {
            turn: TurnId(0),
            frames: scripted_frames(1, 1, 0),
        };
        let err = engine.transcribe(&utterance, None, &cancel).unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn fake_s2s_with_delay_cancellable_mid_wait() {
        use std::sync::Arc;
        use std::thread;

        let cancel = Arc::new(Cancel::new());
        let cancel_worker = Arc::clone(&cancel);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(15));
            cancel_worker.cancel_generation();
        });
        let mut engine = FakeS2s::with_delay(Duration::from_millis(150));
        let utterance = Utterance {
            turn: TurnId(0),
            frames: scripted_frames(1, 1, 0),
        };
        let err = engine
            .generate(&utterance, &cancel, &mut |_| Ok(()), &mut |_, _| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn fake_s2s_rejects_empty_utterance_and_shutdown() {
        let mut engine = FakeS2s::new();
        let cancel = Cancel::new();
        let err = engine
            .generate(
                &Utterance {
                    turn: TurnId(0),
                    frames: vec![],
                },
                &cancel,
                &mut |_| Ok(()),
                &mut |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
        cancel.shutdown();
        let err = engine
            .generate(
                &Utterance {
                    turn: TurnId(1),
                    frames: scripted_frames(1, 1, 0),
                },
                &cancel,
                &mut |_| Ok(()),
                &mut |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
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
