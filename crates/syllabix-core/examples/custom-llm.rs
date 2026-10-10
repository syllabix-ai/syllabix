//! Lane 3 — compose stages: custom `Llm` + `run_loop`.
//!
//! Copy-paste starting point for other repos that implement `Llm` (or any
//! stage) and call `run_loop` instead of forking `pipeline.rs`. Uses the
//! Lane 3 crate-root surface listed in `docs/embed.md`.
//!
//! Companion VAD / STT / TTS / sink types are in-example fixtures so this
//! binary compiles and runs without opening a microphone or fetching the
//! launch-stack weights. Production hosts typically keep those stages from
//! `load_real_providers` and swap only the LLM.

use std::sync::mpsc::channel;

use syllabix_core::{
    run_loop, AudioFrame, AudioSink, BuiltinDefaults, Cancel, Error, HistoryTurn, Llm, LoopConfig,
    LoopMode, PipelineStages, Result, RuntimeControls, Stt, SynthesizedAudio, TokenChunk,
    Transcript, Tts, TurnId, Utterance, Vad, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ,
    FRAME_SAMPLES,
};

struct EchoLlm;

impl Llm for EchoLlm {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn generate(
        &mut self,
        _history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        on_token(TokenChunk {
            turn: user.turn,
            generation: cancel.generation(),
            index: 0,
            text: user.text.clone(),
            is_last: true,
        })
    }
}

/// Energy-based VAD: non-zero PCM starts a turn, silence (or flush) ends it.
struct EnergyVad {
    next_turn: u64,
    current: Option<(TurnId, Vec<AudioFrame>)>,
}

impl Vad for EnergyVad {
    fn name(&self) -> &'static str {
        "energy"
    }

    fn push_frame(&mut self, frame: AudioFrame) -> Result<Vec<VadEvent>> {
        if frame.has_energy() {
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
            return Ok(events);
        }
        Ok(self.end_speech().into_iter().collect())
    }

    fn flush(&mut self) -> Result<Vec<VadEvent>> {
        Ok(self.end_speech().into_iter().collect())
    }
}

impl EnergyVad {
    fn end_speech(&mut self) -> Option<VadEvent> {
        self.current
            .take()
            .map(|(turn, frames)| VadEvent::SpeechEnd {
                utterance: Utterance { turn, frames },
            })
    }
}

struct FixedStt;

impl Stt for FixedStt {
    fn name(&self) -> &'static str {
        "fixed"
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(Transcript {
            turn: utterance.turn,
            text: String::from("hello from custom llm"),
            language: String::from("en"),
        })
    }
}

struct PassthroughTts;

impl Tts for PassthroughTts {
    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        if cancel.is_stale(token.generation) {
            return Err(Error::Cancelled);
        }
        Ok(vec![SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index: token.index,
            samples: vec![1],
            is_last: token.is_last,
        }])
    }
}

struct DiscardSink;

impl AudioSink for DiscardSink {
    fn play(&mut self, _audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

fn fixture_turn() -> Result<Vec<AudioFrame>> {
    let speech = vec![1000_i16; FRAME_SAMPLES];
    let silence = vec![0_i16; FRAME_SAMPLES];
    Ok(vec![
        AudioFrame::new(0, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, speech.clone())?,
        AudioFrame::new(1, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, speech)?,
        AudioFrame::new(2, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, silence)?,
    ])
}

fn main() -> Result<()> {
    let cancel = Cancel::new();
    let (events_tx, events_rx) = channel();
    let printer = std::thread::spawn(move || {
        while let Ok(event) = events_rx.recv() {
            println!("event: {event:?}");
        }
    });
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: Some(events_tx),
            turn_debug: None,
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: EnergyVad {
                next_turn: 0,
                current: None,
            },
            stt: FixedStt,
            llm: EchoLlm,
            tts: PassthroughTts,
            sink: DiscardSink,
        },
        fixture_turn()?,
        cancel,
    )?;
    let _ = printer.join();
    println!(
        "conversation ended: {} turn(s), {} skipped",
        report.turns.len(),
        report.skipped_turns
    );
    Ok(())
}
