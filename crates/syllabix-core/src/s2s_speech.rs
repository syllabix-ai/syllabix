//! Cascade STT/TTS adapters backed by a shared [`crate::s2s::S2sEngine`].

use std::sync::{Arc, Mutex};

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::s2s::{S2sEngine, LFM3_AUDIO_2B};
use crate::speech_text::{speak_text_for_tts, take_sentences, ThinkFilter};
use crate::stt::language::LANGUAGE_AUTO;
use crate::stt::{Stt, Transcript};
use crate::tts::Tts;
use crate::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId, Utterance};

type SharedS2s = Arc<Mutex<Box<dyn S2sEngine>>>;

/// LFM3-Audio ASR stage for [`crate::config::S2sReplyWith::Llm`].
pub struct S2sStt {
    engine: SharedS2s,
    language: Option<String>,
}

impl S2sStt {
    pub fn new(engine: SharedS2s, language: Option<String>) -> Self {
        Self { engine, language }
    }
}

impl Stt for S2sStt {
    fn name(&self) -> &'static str {
        LFM3_AUDIO_2B
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let lang_for_engine = self.language.as_deref();
        let transcript_language = self
            .language
            .as_deref()
            .unwrap_or(LANGUAGE_AUTO)
            .to_string();
        let text = {
            let mut engine = self.engine.lock().expect("s2s engine lock");
            engine.transcribe(utterance, lang_for_engine, cancel)?
        };
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: transcript_language,
        })
    }
}

#[derive(Default)]
struct ChunkState {
    think: ThinkFilter,
    buffer: String,
    turn: Option<TurnId>,
    generation: Option<GenerationId>,
    next_index: u32,
}

impl ChunkState {
    fn reset(&mut self) {
        self.think = ThinkFilter::default();
        self.buffer.clear();
        self.turn = None;
        self.generation = None;
        self.next_index = 0;
    }

    fn emit(&mut self, samples: Vec<i16>, token: &TokenChunk, is_last: bool) -> SynthesizedAudio {
        let index = self.next_index;
        self.next_index += 1;
        SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index,
            samples,
            is_last,
        }
    }
}

/// LFM3-Audio TTS stage for [`crate::config::S2sReplyWith::Llm`].
pub struct S2sTts {
    engine: SharedS2s,
    language: Option<String>,
    voice: Option<String>,
    state: ChunkState,
}

impl S2sTts {
    pub fn new(engine: SharedS2s, language: Option<String>, voice: Option<String>) -> Self {
        Self {
            engine,
            language,
            voice,
            state: ChunkState::default(),
        }
    }
}

impl Tts for S2sTts {
    fn name(&self) -> &'static str {
        LFM3_AUDIO_2B
    }

    fn model_id(&self) -> Option<&str> {
        Some(LFM3_AUDIO_2B)
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        let mut chunks = Vec::new();
        self.synthesize_chunk_into(token, cancel, &mut |audio| {
            chunks.push(audio);
            Ok(())
        })?;
        Ok(chunks)
    }

    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_stale(token.generation) {
            self.state.reset();
            return Err(Error::Cancelled);
        }
        if self.state.generation != Some(token.generation) || self.state.turn != Some(token.turn) {
            self.state.reset();
            self.state.generation = Some(token.generation);
            self.state.turn = Some(token.turn);
        }

        self.state
            .buffer
            .push_str(&self.state.think.push(&token.text, token.is_last));
        let sentences = take_sentences(&mut self.state.buffer, token.is_last);
        let spoken: Vec<String> = sentences
            .into_iter()
            .map(|sentence| speak_text_for_tts(&sentence))
            .filter(|sentence| !sentence.is_empty())
            .collect();
        let last_i = spoken.len().saturating_sub(1);
        let mut emitted_final = false;
        let lang = self.language.as_deref();
        let voice = self.voice.as_deref();
        for (i, sentence) in spoken.into_iter().enumerate() {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                self.state.reset();
                return Err(Error::Cancelled);
            }
            let is_last_sentence = token.is_last && i == last_i;
            let mut state = std::mem::take(&mut self.state);
            let mut index = state.next_index;
            let result = {
                let mut engine = self.engine.lock().expect("s2s engine lock");
                engine.speak(
                    token.turn,
                    token.generation,
                    &sentence,
                    lang,
                    voice,
                    cancel,
                    &mut |mut audio| {
                        if audio.is_last {
                            emitted_final |= is_last_sentence;
                        }
                        audio.is_last = is_last_sentence && audio.is_last;
                        audio.index = index;
                        index += 1;
                        on_audio(audio)
                    },
                )
            };
            state.next_index = index;
            self.state = state;
            if let Err(err) = result {
                self.state.reset();
                return Err(err);
            }
        }
        if token.is_last && !emitted_final {
            on_audio(self.state.emit(vec![0; 16], token, true))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{scripted_frames, FakeS2s, FakeS2sOp};

    fn shared_fake() -> SharedS2s {
        Arc::new(Mutex::new(Box::new(FakeS2s::new()) as Box<dyn S2sEngine>))
    }

    #[test]
    fn s2s_stt_transcribes_fake_utterance() {
        let mut stt = S2sStt::new(shared_fake(), Some("fr".into()));
        let utterance = Utterance {
            turn: TurnId(0),
            frames: scripted_frames(1, 2, 0),
        };
        let transcript = stt
            .transcribe(&utterance, &Cancel::new())
            .expect("transcribe");
        assert_eq!(transcript.language, "fr");
        assert!(!transcript.text.is_empty());
    }

    #[test]
    fn s2s_tts_sentence_buffer_emits_final_marker() {
        let fake = FakeS2s::new();
        let ops = fake.ops_log();
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let mut chunks = Vec::new();
        let tokens = [
            TokenChunk {
                turn: TurnId(1),
                generation,
                index: 0,
                text: "Hello there. ".into(),
                is_last: false,
            },
            TokenChunk {
                turn: TurnId(1),
                generation,
                index: 1,
                text: "How are you?".into(),
                is_last: true,
            },
        ];
        for token in tokens {
            tts.synthesize_chunk_into(&token, &cancel, &mut |audio| {
                chunks.push(audio);
                Ok(())
            })
            .expect("synthesize");
        }
        let log = ops.lock().expect("ops");
        let speak_ops: Vec<_> = log
            .iter()
            .filter(|op| matches!(op, FakeS2sOp::Speak { .. }))
            .collect();
        assert_eq!(speak_ops.len(), 2, "one speak call per sentence");
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
    }

    #[test]
    fn s2s_tts_stale_generation_cancels() {
        let mut tts = S2sTts::new(shared_fake(), None, None);
        let cancel = Cancel::new();
        let stale = cancel.generation();
        cancel.cancel_generation();
        let token = TokenChunk {
            turn: TurnId(0),
            generation: stale,
            index: 0,
            text: "Hi.".into(),
            is_last: true,
        };
        let err = tts
            .synthesize_chunk_into(&token, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn s2s_stt_shutdown_and_default_language() {
        let mut stt = S2sStt::new(shared_fake(), None);
        assert_eq!(stt.name(), LFM3_AUDIO_2B);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(0),
                    frames: scripted_frames(0, 1, 0),
                },
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        let cancel = Cancel::new();
        let transcript = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: scripted_frames(1, 1, 0),
                },
                &cancel,
            )
            .expect("transcribe");
        assert_eq!(transcript.language, LANGUAGE_AUTO);
    }

    #[test]
    fn s2s_tts_accessors_and_synthesize_chunk_wrapper() {
        let mut tts = S2sTts::new(shared_fake(), Some("en".into()), Some("male".into()));
        assert_eq!(tts.name(), LFM3_AUDIO_2B);
        assert_eq!(tts.model_id(), Some(LFM3_AUDIO_2B));
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let chunks = tts
            .synthesize_chunk(
                &TokenChunk {
                    turn: TurnId(2),
                    generation,
                    index: 0,
                    text: "Done.".into(),
                    is_last: true,
                },
                &cancel,
            )
            .expect("chunk");
        assert!(chunks.iter().any(|c| c.is_last));
    }

    #[test]
    fn s2s_tts_resets_on_new_turn_and_emits_silence_when_needed() {
        let mut tts = S2sTts::new(shared_fake(), None, None);
        let cancel = Cancel::new();
        let gen_a = cancel.generation();
        let token_a = TokenChunk {
            turn: TurnId(3),
            generation: gen_a,
            index: 0,
            text: "First.".into(),
            is_last: true,
        };
        tts.synthesize_chunk_into(&token_a, &cancel, &mut |_| Ok(()))
            .expect("first turn");
        let token_b = TokenChunk {
            turn: TurnId(4),
            generation: gen_a,
            index: 0,
            text: " ".into(),
            is_last: true,
        };
        let mut silence = false;
        tts.synthesize_chunk_into(&token_b, &cancel, &mut |audio| {
            if audio.is_last && audio.samples.iter().all(|&s| s == 0) {
                silence = true;
            }
            Ok(())
        })
        .expect("whitespace last");
        assert!(silence);
    }

    #[test]
    fn s2s_tts_cancel_mid_sentence_resets_state() {
        let engine = shared_fake();
        let mut tts = S2sTts::new(Arc::clone(&engine), None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(5),
            generation,
            index: 0,
            text: "One. Two. Three.".into(),
            is_last: true,
        };
        let err = tts
            .synthesize_chunk_into(&token, &cancel, &mut |_| {
                cancel.cancel_generation();
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }
}
