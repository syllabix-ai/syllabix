//! Cascade STT/TTS adapters backed by a shared [`crate::s2s::S2sEngine`].

use std::sync::{Arc, Mutex};

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::s2s::{S2sEngine, LFM3_AUDIO_2B};
use crate::speech_text::{speak_text_for_tts, ThinkFilter};
use crate::stt::language::LANGUAGE_AUTO;
use crate::stt::{Stt, Transcript};
use crate::tts::Tts;
use crate::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId, Utterance};

type SharedS2s = Arc<Mutex<Box<dyn S2sEngine>>>;

fn resolve_transcript_language(s2s_language: Option<&str>, config_language: &str) -> String {
    if let Some(lang) = s2s_language {
        return lang.to_string();
    }
    if config_language != LANGUAGE_AUTO {
        return config_language.to_string();
    }
    crate::stt::STT_LANGUAGE.to_string()
}

/// LFM3-Audio ASR stage for [`crate::config::S2sReplyWith::Llm`].
pub struct S2sStt {
    engine: SharedS2s,
    language: Option<String>,
    transcript_language: String,
}

impl S2sStt {
    pub fn new(
        engine: SharedS2s,
        language: Option<String>,
        config_language: impl AsRef<str>,
    ) -> Self {
        let transcript_language =
            resolve_transcript_language(language.as_deref(), config_language.as_ref());
        Self {
            engine,
            language,
            transcript_language,
        }
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
        let transcript_language = self.transcript_language.clone();
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
        if !token.is_last {
            return Ok(());
        }

        let spoken = speak_text_for_tts(&self.state.buffer);
        self.state.buffer.clear();

        if spoken.is_empty() {
            on_audio(self.state.emit(vec![0; 16], token, true))?;
            return Ok(());
        }

        let lang = self.language.as_deref();
        let voice = self.voice.as_deref();
        let mut emitted_final = false;
        let mut state = std::mem::take(&mut self.state);
        let mut index = state.next_index;
        let mut buffered = Vec::new();
        let result = {
            let mut engine = self.engine.lock().expect("s2s engine lock");
            engine.speak(
                token.turn,
                token.generation,
                &spoken,
                lang,
                voice,
                cancel,
                &mut |mut audio| {
                    if audio.is_last {
                        emitted_final = true;
                    }
                    audio.index = index;
                    index += 1;
                    buffered.push(audio);
                    Ok(())
                },
            )
        };
        state.next_index = index;
        self.state = state;
        if let Err(err) = result {
            self.state.reset();
            return Err(err);
        }
        for audio in buffered {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                self.state.reset();
                return Err(Error::Cancelled);
            }
            on_audio(audio)?;
        }
        if !emitted_final {
            on_audio(self.state.emit(vec![0; 16], token, true))?;
        }
        let capped = {
            let engine = self.engine.lock().expect("s2s engine lock");
            engine.last_output_capped()
        };
        if capped {
            tracing::debug!("s2s TTS hit the max-out-samples cap; ending the turn");
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
        let mut stt = S2sStt::new(shared_fake(), Some("fr".into()), "en");
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
    fn s2s_tts_buffers_until_last_token_then_one_speak() {
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
        for (i, token) in tokens.iter().enumerate() {
            tts.synthesize_chunk_into(token, &cancel, &mut |audio| {
                chunks.push(audio);
                Ok(())
            })
            .expect("synthesize");
            if i == 0 {
                let log = ops.lock().expect("ops");
                assert!(
                    !log.iter().any(|op| matches!(op, FakeS2sOp::Speak { .. })),
                    "must not speak before the last token"
                );
                assert!(chunks.is_empty(), "no audio before the last token");
            }
        }
        let log = ops.lock().expect("ops");
        let speak_ops: Vec<_> = log
            .iter()
            .filter(|op| matches!(op, FakeS2sOp::Speak { .. }))
            .collect();
        assert_eq!(speak_ops.len(), 1, "one speak for the full reply");
        if let FakeS2sOp::Speak { text, .. } = speak_ops[0] {
            assert!(text.contains("Hello there"));
            assert!(text.contains("How are you"));
        }
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
    }

    #[test]
    fn s2s_tts_max_out_samples_cap_ends_turn() {
        let fake = FakeS2s::new().with_max_out_samples(8);
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared.clone(), None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let mut chunks = Vec::new();
        tts.synthesize_chunk_into(
            &TokenChunk {
                turn: TurnId(6),
                generation,
                index: 0,
                text: "abcdefghij".into(),
                is_last: true,
            },
            &cancel,
            &mut |audio| {
                chunks.push(audio);
                Ok(())
            },
        )
        .expect("capped speak");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), 8);
        assert!(chunks[0].is_last);
        let engine = shared.lock().expect("s2s");
        assert!(engine.last_output_capped());
    }

    #[test]
    fn s2s_tts_under_cap_is_not_flagged() {
        let fake = FakeS2s::new().with_max_out_samples(8);
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared.clone(), None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        tts.synthesize_chunk_into(
            &TokenChunk {
                turn: TurnId(7),
                generation,
                index: 0,
                text: "hi".into(),
                is_last: true,
            },
            &cancel,
            &mut |_| Ok(()),
        )
        .expect("short speak");
        let engine = shared.lock().expect("s2s");
        assert!(!engine.last_output_capped());
    }

    #[test]
    fn s2s_tts_cancel_while_buffering_skips_speak() {
        let fake = FakeS2s::new();
        let ops = fake.ops_log();
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let stale = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(8),
            generation: stale,
            index: 0,
            text: "Hello ".into(),
            is_last: false,
        };
        tts.synthesize_chunk_into(&token, &cancel, &mut |_| Ok(()))
            .expect("buffer");
        cancel.cancel_generation();
        let err = tts
            .synthesize_chunk_into(
                &TokenChunk {
                    turn: TurnId(8),
                    generation: stale,
                    index: 1,
                    text: "world.".into(),
                    is_last: true,
                },
                &cancel,
                &mut |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        let log = ops.lock().expect("ops");
        assert!(
            !log.iter().any(|op| matches!(op, FakeS2sOp::Speak { .. })),
            "stale generation must not invoke speak"
        );
        let generation = cancel.generation();
        tts.synthesize_chunk_into(
            &TokenChunk {
                turn: TurnId(9),
                generation,
                index: 0,
                text: "Fresh.".into(),
                is_last: true,
            },
            &cancel,
            &mut |_| Ok(()),
        )
        .expect("new generation");
        let log = ops.lock().expect("ops");
        assert_eq!(
            log.iter()
                .filter(|op| matches!(op, FakeS2sOp::Speak { .. }))
                .count(),
            1
        );
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
        let mut stt = S2sStt::new(shared_fake(), None, LANGUAGE_AUTO);
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
        assert_eq!(
            transcript.language,
            crate::stt::STT_LANGUAGE,
            "unset s2s language must not stamp auto onto the LLM transcript"
        );
    }

    #[test]
    fn s2s_stt_transcript_language_uses_config_when_s2s_unset() {
        let mut stt = S2sStt::new(shared_fake(), None, "de");
        let transcript = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(2),
                    frames: scripted_frames(2, 1, 0),
                },
                &Cancel::new(),
            )
            .expect("transcribe");
        assert_eq!(transcript.language, "de");
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
