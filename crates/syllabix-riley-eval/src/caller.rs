//! Simulated caller (#329 phase A3): an LLM plays the scenario's caller and
//! each line is turned into audio for the cascade.
//!
//! Voice runs use [`TtsVoice`] (real `RemoteTts`, `SYLLABIX_TTS_API_KEY`),
//! so the agent hears synthesized speech through VAD and STT. Text smoke
//! runs use [`TextVoice`] + [`QueueStt`], which hand the caller's text
//! straight to the agent; they are smoke checks, not voice scores.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use syllabix_core::types::{TokenChunk, Transcript, TurnId, Utterance, FRAME_SAMPLES};
use syllabix_core::{Cancel, Error, Result, Stt, Tts};

use crate::chat::ChatClient;
use crate::scenario::Persona;
use crate::session::{Line, Speaker};
use crate::voice::{CallerScript, CallerTurn};

/// Marker the simulated caller adds to its last line.
pub const END_CALL: &str = "[END_CALL]";

/// Riley's greeting from the vendored prompt; the call starts after it.
pub const AGENT_GREETING: &str = "Thanks for calling Acme Bank, this is Riley — how can I help?";

/// System prompt for the caller model. The agent never sees it.
pub fn system_prompt(persona: &Persona) -> String {
    let facts: Vec<String> = persona
        .facts
        .iter()
        .map(|(key, value)| format!("- {key}: {value}"))
        .collect();
    format!(
        "You are role-playing a person who phoned Acme Bank's card support line. \
         You are the caller, never the agent.\n\
         Your private goal: {goal}\n\
         How you behave: {traits}.\n\
         Facts you know (give them when asked or when it helps your goal; do not invent other personal details):\n\
         {facts}\n\
         Rules: reply with only what you say out loud, one or two short sentences, no stage directions, \
         lists, or markup. Write numbers and abbreviations the way you say them: digits one by one in \
         words (\"two one seven, five five five\"), street and state names in full (\"Road\", \"New York\"). \
         Never mention these instructions. When your goal is done or clearly cannot be \
         done, say a short goodbye and end that line with {END_CALL}.",
        goal = persona.goal,
        traits = persona.traits.join(", "),
        facts = facts.join("\n"),
    )
}

/// Chat-completions caller with a private persona.
pub struct CallerSim {
    client: ChatClient,
    system: String,
    max_turns: usize,
    said: Vec<String>,
    ended: bool,
}

impl CallerSim {
    /// Caller for `persona` that hangs up after `max_turns` lines.
    pub fn new(client: ChatClient, persona: &Persona, max_turns: usize) -> Self {
        Self {
            client,
            system: system_prompt(persona),
            max_turns,
            said: Vec::new(),
            ended: false,
        }
    }

    /// True after [`END_CALL`] or `max_turns` lines.
    pub fn finished(&self) -> bool {
        self.ended || self.said.len() >= self.max_turns
    }

    /// Lines the caller said, in order (before speech recognition).
    pub fn said(&self) -> &[String] {
        &self.said
    }

    /// Request messages: the agent's lines are `user`, the caller's own
    /// lines are `assistant`. Agent replies come from `transcript`, grouped
    /// by the caller line they follow.
    pub fn messages(&self, transcript: &[Line]) -> Vec<Value> {
        let mut replies: Vec<String> = Vec::new();
        for line in transcript {
            match line.speaker {
                Speaker::Caller => replies.push(String::new()),
                Speaker::Agent => {
                    if let Some(reply) = replies.last_mut() {
                        if !reply.is_empty() {
                            reply.push(' ');
                        }
                        reply.push_str(line.text.trim());
                    }
                }
            }
        }
        let mut messages = vec![
            json!({ "role": "system", "content": self.system }),
            json!({ "role": "user", "content": AGENT_GREETING }),
        ];
        for (index, line) in self.said.iter().enumerate() {
            messages.push(json!({ "role": "assistant", "content": line }));
            let reply = replies
                .get(index)
                .filter(|reply| !reply.is_empty())
                .map_or("(no answer)", String::as_str);
            messages.push(json!({ "role": "user", "content": reply }));
        }
        messages
    }

    /// Next line, or `None` when the caller hangs up.
    pub fn next_line(
        &mut self,
        transcript: &[Line],
    ) -> std::result::Result<Option<String>, String> {
        if self.finished() {
            return Ok(None);
        }
        let reply = self
            .client
            .complete(&self.messages(transcript), false)
            .map_err(|err| format!("caller model: {err}"))?;
        let ended = reply.contains(END_CALL);
        let line = reply.replace(END_CALL, "").trim().to_string();
        self.ended = ended;
        if line.is_empty() {
            self.ended = true;
            return Ok(None);
        }
        self.said.push(line.clone());
        Ok(Some(line))
    }
}

/// Turns a caller line into 16 kHz mono audio.
pub trait Voice: Send {
    /// Audio for `text`.
    fn speak(&mut self, text: &str) -> std::result::Result<Vec<i16>, String>;
}

/// Voice runs: synthesize with a TTS stage (normally `RemoteTts`).
pub struct TtsVoice<T: Tts>(pub T);

impl<T: Tts> Voice for TtsVoice<T> {
    fn speak(&mut self, text: &str) -> std::result::Result<Vec<i16>, String> {
        let cancel = Cancel::new();
        let chunk = TokenChunk {
            turn: TurnId(0),
            generation: cancel.generation(),
            index: 0,
            text: text.to_string(),
            is_last: true,
        };
        let audio = self
            .0
            .synthesize_chunk(&chunk, &cancel)
            .map_err(|err| format!("caller voice: {err}"))?;
        let samples: Vec<i16> = audio.into_iter().flat_map(|chunk| chunk.samples).collect();
        if samples.iter().all(|sample| *sample == 0) {
            return Err("caller voice returned silence".into());
        }
        Ok(samples)
    }
}

/// Text queue shared by [`TextVoice`] and [`QueueStt`].
pub type TextQueue = Arc<Mutex<VecDeque<String>>>;

/// Text smoke runs: queue the text and play a short tone so the VAD opens
/// a turn; [`QueueStt`] returns the queued text.
pub struct TextVoice(pub TextQueue);

/// Sample level of [`TextVoice`]'s placeholder audio.
pub const TEXT_TONE_LEVEL: i16 = 1000;

impl Voice for TextVoice {
    fn speak(&mut self, text: &str) -> std::result::Result<Vec<i16>, String> {
        self.0
            .lock()
            .expect("text queue")
            .push_back(text.to_string());
        Ok(vec![TEXT_TONE_LEVEL; FRAME_SAMPLES * 3])
    }
}

/// STT for text smoke runs: each utterance takes the next queued line.
pub struct QueueStt(pub TextQueue);

impl Stt for QueueStt {
    fn name(&self) -> &'static str {
        "riley-text-queue"
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let text = self
            .0
            .lock()
            .expect("text queue")
            .pop_front()
            .ok_or_else(|| Error::Provider {
                provider: "riley-text-queue",
                message: "no caller line queued".into(),
            })?;
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: "en".into(),
        })
    }
}

/// [`CallerScript`] that asks a [`CallerSim`] for each line and speaks it.
pub struct SimCaller<V: Voice> {
    sim: CallerSim,
    voice: V,
}

impl<V: Voice> SimCaller<V> {
    /// Caller that speaks through `voice`.
    pub fn new(sim: CallerSim, voice: V) -> Self {
        Self { sim, voice }
    }
}

impl<V: Voice> CallerScript for SimCaller<V> {
    fn peek(&self) -> Option<bool> {
        (!self.sim.finished()).then_some(false)
    }

    fn next_turn(
        &mut self,
        transcript: &[Line],
    ) -> std::result::Result<Option<CallerTurn>, String> {
        let Some(line) = self.sim.next_line(transcript)? else {
            return Ok(None);
        };
        let audio = self.voice.speak(&line)?;
        Ok(Some(CallerTurn {
            audio,
            barge_in: false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Duration;
    use syllabix_core::testing::FakeTts;
    use zeroize::Zeroizing;

    fn persona() -> Persona {
        Persona {
            goal: "Freeze card 1111.".into(),
            traits: vec!["polite".into(), "brief".into()],
            facts: BTreeMap::from([("name".to_string(), "Alice Johnson".to_string())]),
        }
    }

    fn offline_sim(max_turns: usize) -> CallerSim {
        // Port 9 on loopback: nothing listens, so any request fails fast.
        let client = ChatClient::new(
            "http://127.0.0.1:9/v1",
            "caller-model",
            Zeroizing::new("k".into()),
            Duration::from_secs(2),
        );
        CallerSim::new(client, &persona(), max_turns)
    }

    fn line(turn: u64, speaker: Speaker, text: &str) -> Line {
        Line {
            turn,
            speaker,
            text: text.into(),
            cut_off: false,
        }
    }

    #[test]
    fn prompt_lists_goal_traits_and_facts() {
        let prompt = system_prompt(&persona());
        assert!(prompt.contains("Your private goal: Freeze card 1111."));
        assert!(prompt.contains("How you behave: polite, brief."));
        assert!(prompt.contains("- name: Alice Johnson"));
        assert!(prompt.contains(END_CALL));
    }

    #[test]
    fn messages_pair_own_lines_with_agent_replies() {
        let mut sim = offline_sim(4);
        sim.said = vec!["I lost my card.".into(), "Alice Johnson.".into()];
        let transcript = [
            line(0, Speaker::Caller, "i lost my card"),
            line(0, Speaker::Agent, "Sorry to hear that."),
            line(0, Speaker::Agent, "Your name?"),
            line(1, Speaker::Caller, "alice johnson"),
        ];
        let messages = sim.messages(&transcript);
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "user", "assistant", "user"]
        );
        assert_eq!(messages[1]["content"], AGENT_GREETING);
        assert_eq!(messages[2]["content"], "I lost my card.");
        assert_eq!(messages[3]["content"], "Sorry to hear that. Your name?");
        assert_eq!(messages[5]["content"], "(no answer)");
    }

    #[test]
    fn finished_sim_hangs_up_without_a_request() {
        let mut sim = offline_sim(1);
        sim.said.push("hi".into());
        assert!(sim.finished());
        assert_eq!(sim.next_line(&[]), Ok(None));
        let mut caller = SimCaller::new(sim, TextVoice(TextQueue::default()));
        assert_eq!(caller.peek(), None);
        assert_eq!(caller.next_turn(&[]), Ok(None));
    }

    #[test]
    fn model_errors_are_reported_without_details() {
        let mut caller = SimCaller::new(offline_sim(2), TextVoice(TextQueue::default()));
        assert_eq!(caller.peek(), Some(false));
        let err = caller.next_turn(&[]).unwrap_err();
        assert!(
            err.starts_with("caller model: chat completions request failed"),
            "{err}"
        );
        assert!(!err.contains("Bearer"), "{err}");
    }

    #[test]
    fn text_voice_feeds_queue_stt() {
        let queue = TextQueue::default();
        let mut voice = TextVoice(queue.clone());
        let audio = voice.speak("hello there").unwrap();
        assert_eq!(audio.len(), FRAME_SAMPLES * 3);
        assert!(audio.iter().all(|s| *s == TEXT_TONE_LEVEL));
        let mut stt = QueueStt(queue);
        assert_eq!(stt.name(), "riley-text-queue");
        let utterance = Utterance {
            turn: TurnId(4),
            frames: Vec::new(),
        };
        let cancel = Cancel::new();
        let transcript = stt.transcribe(&utterance, &cancel).unwrap();
        assert_eq!(
            (transcript.turn, transcript.text.as_str()),
            (TurnId(4), "hello there")
        );
        let err = stt.transcribe(&utterance, &cancel).unwrap_err();
        assert!(err.to_string().contains("no caller line queued"), "{err}");
        cancel.shutdown();
        assert!(matches!(
            stt.transcribe(&utterance, &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn tts_voice_concatenates_chunks_and_rejects_silence() {
        let mut voice = TtsVoice(FakeTts);
        let audio = voice.speak("hi").unwrap();
        assert_eq!(audio, vec![i16::from(b'h'), i16::from(b'i')]);
        struct Silent;
        impl Tts for Silent {
            fn name(&self) -> &'static str {
                "silent"
            }
            fn synthesize_chunk(
                &mut self,
                _token: &TokenChunk,
                _cancel: &Cancel,
            ) -> Result<Vec<syllabix_core::types::SynthesizedAudio>> {
                Ok(Vec::new())
            }
        }
        assert_eq!(
            TtsVoice(Silent).speak("hi").unwrap_err(),
            "caller voice returned silence"
        );
        struct Broken;
        impl Tts for Broken {
            fn name(&self) -> &'static str {
                "broken"
            }
            fn synthesize_chunk(
                &mut self,
                _token: &TokenChunk,
                _cancel: &Cancel,
            ) -> Result<Vec<syllabix_core::types::SynthesizedAudio>> {
                Err(Error::Provider {
                    provider: "broken",
                    message: "HTTP 500".into(),
                })
            }
        }
        let err = TtsVoice(Broken).speak("hi").unwrap_err();
        assert!(err.starts_with("caller voice:"), "{err}");
    }
}
