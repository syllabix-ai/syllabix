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
    /// Set when the last `speak` call reached the backend's sample cap, i.e.
    /// the cap was reached, so the reply may have been cut. A reply that
    /// legitimately ends exactly on the cap is also flagged, so treat this as
    /// a heuristic. The cap unit test asserts it; keep it on the state for
    /// diagnostics.
    #[cfg_attr(not(test), allow(dead_code))]
    truncated: bool,
}

impl ChunkState {
    fn reset(&mut self) {
        self.think = ThinkFilter::default();
        self.buffer.clear();
        self.turn = None;
        self.generation = None;
        self.next_index = 0;
        self.truncated = false;
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
        let mut emitted_final = false;
        if !spoken.is_empty() {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                self.state.reset();
                return Err(Error::Cancelled);
            }
            let lang = self.language.as_deref();
            let voice = self.voice.as_deref();
            let mut state = std::mem::take(&mut self.state);
            let mut index = state.next_index;
            let mut buffered = Vec::new();
            let mut emitted_samples = 0usize;
            let mut truncated = false;
            let result = {
                let mut engine = self.engine.lock().expect("s2s engine lock");
                let cap = engine.speak_sample_cap();
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
                        index = index.saturating_add(1);
                        emitted_samples = emitted_samples.saturating_add(audio.samples.len());
                        if let Some(cap) = cap {
                            if !truncated && emitted_samples >= cap {
                                truncated = true;
                                tracing::warn!(
                                    cap_samples = cap,
                                    emitted_samples,
                                    "s2s tts reply reached the sample cap; reply may have been cut and the turn ended"
                                );
                                eprintln!(
                                    "s2s tts: reply reached the {cap}-sample cap; it may have been cut and the turn ended"
                                );
                            }
                        }
                        buffered.push(audio);
                        Ok(())
                    },
                )
            };
            state.next_index = index;
            state.truncated = truncated;
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
        }
        if !emitted_final {
            on_audio(self.state.emit(vec![0; 16], token, true))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::fake::{scripted_frames, FakeS2s, FakeS2sOp};

    fn shared_fake() -> SharedS2s {
        Arc::new(Mutex::new(Box::new(FakeS2s::new()) as Box<dyn S2sEngine>))
    }

    /// Emits `chunks` PCM chunks from `speak`, optionally marking the last one.
    struct MultiChunkS2s {
        chunks: usize,
        mark_last: bool,
    }

    impl S2sEngine for MultiChunkS2s {
        fn name(&self) -> &'static str {
            "multi-chunk-s2s"
        }

        fn generate(
            &mut self,
            _utterance: &Utterance,
            _cancel: &Cancel,
            _on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
            _on_text: &mut dyn FnMut(&str, bool) -> Result<()>,
        ) -> Result<()> {
            Err(Error::Provider {
                provider: self.name(),
                message: "generate is not scripted".into(),
            })
        }

        fn speak(
            &mut self,
            turn: TurnId,
            generation: GenerationId,
            _text: &str,
            _language: Option<&str>,
            _voice: Option<&str>,
            _cancel: &Cancel,
            on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
        ) -> Result<()> {
            for index in 0..self.chunks {
                on_audio(SynthesizedAudio {
                    turn,
                    generation,
                    index: index as u32,
                    samples: vec![index as i16 + 1],
                    is_last: self.mark_last && index + 1 == self.chunks,
                })?;
            }
            Ok(())
        }
    }

    struct WarnCounter(Arc<AtomicUsize>);

    impl tracing::Subscriber for WarnCounter {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            *metadata.level() <= tracing::Level::WARN
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn enter(&self, _span: &tracing::span::Id) {}

        fn exit(&self, _span: &tracing::span::Id) {}
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
    fn s2s_tts_whole_reply_is_one_speak_call() {
        let fake = FakeS2s::new();
        let ops = fake.ops_log();
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let mut chunks = Vec::new();
        let first = TokenChunk {
            turn: TurnId(1),
            generation,
            index: 0,
            text: "Hello there. ".into(),
            is_last: false,
        };
        tts.synthesize_chunk_into(&first, &cancel, &mut |audio| {
            chunks.push(audio);
            Ok(())
        })
        .expect("buffer first token");
        assert!(
            chunks.is_empty(),
            "no audio before the final token of the reply"
        );
        let last = TokenChunk {
            turn: TurnId(1),
            generation,
            index: 1,
            text: "How are you?".into(),
            is_last: true,
        };
        tts.synthesize_chunk_into(&last, &cancel, &mut |audio| {
            chunks.push(audio);
            Ok(())
        })
        .expect("synthesize whole reply");
        let log = ops.lock().expect("ops");
        let speak_texts: Vec<&str> = log
            .iter()
            .filter_map(|op| match op {
                FakeS2sOp::Speak { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            speak_texts,
            vec!["Hello there. How are you?"],
            "the whole reply is one speak call"
        );
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
    }

    #[test]
    fn s2s_tts_engine_chunk_indices_increase_and_last_is_kept() {
        let engine = MultiChunkS2s {
            chunks: 3,
            mark_last: true,
        };
        let shared = Arc::new(Mutex::new(Box::new(engine) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(8),
            generation,
            index: 0,
            text: "One. Two. Three.".into(),
            is_last: true,
        };
        let chunks = tts
            .synthesize_chunk(&token, &cancel)
            .expect("multi-chunk reply");
        assert_eq!(
            chunks.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
        assert!(chunks[2].is_last);
    }

    #[test]
    fn s2s_tts_missing_engine_last_falls_back_to_silence_at_next_index() {
        let engine = MultiChunkS2s {
            chunks: 3,
            mark_last: false,
        };
        let shared = Arc::new(Mutex::new(Box::new(engine) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(9),
            generation,
            index: 0,
            text: "One. Two. Three.".into(),
            is_last: true,
        };
        let chunks = tts
            .synthesize_chunk(&token, &cancel)
            .expect("reply without an engine final chunk");
        assert_eq!(
            chunks.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
        assert!(chunks[3].is_last);
        assert_eq!(chunks[3].samples, vec![0; 16]);
    }

    #[test]
    fn s2s_tts_short_reply_under_cap_is_not_truncated() {
        let cap = 1024usize;
        let fake = FakeS2s::new().with_speak_sample_cap(cap);
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(10),
            generation,
            index: 0,
            text: "Hi.".into(),
            is_last: true,
        };
        let warns = Arc::new(AtomicUsize::new(0));
        let mut chunks = Vec::new();
        tracing::subscriber::with_default(WarnCounter(Arc::clone(&warns)), || {
            tts.synthesize_chunk_into(&token, &cancel, &mut |audio| {
                chunks.push(audio);
                Ok(())
            })
            .expect("short reply");
        });
        assert!(
            !tts.state.truncated,
            "a reply under the cap must not be marked truncated"
        );
        assert_eq!(warns.load(Ordering::SeqCst), 0, "no cap warning");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn s2s_tts_sample_cap_reports_truncation_and_closes_turn() {
        let cap = 8usize;
        let fake = FakeS2s::new().with_speak_sample_cap(cap);
        let ops = fake.ops_log();
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let mut chunks = Vec::new();
        let token = TokenChunk {
            turn: TurnId(6),
            generation,
            index: 0,
            text: "This reply is much longer than the tiny sample cap.".into(),
            is_last: true,
        };
        tts.synthesize_chunk_into(&token, &cancel, &mut |audio| {
            chunks.push(audio);
            Ok(())
        })
        .expect("a capped reply still closes the turn");
        let log = ops.lock().expect("ops");
        assert_eq!(
            log.iter()
                .filter(|op| matches!(op, FakeS2sOp::Speak { .. }))
                .count(),
            1
        );
        assert!(tts.state.truncated, "sample cap hit must be recorded");
        assert_eq!(chunks.iter().filter(|c| c.is_last).count(), 1);
        assert_eq!(
            chunks.iter().map(|c| c.samples.len()).sum::<usize>(),
            cap,
            "the fake truncates PCM to the reported cap"
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
    fn s2s_tts_cancel_during_whole_reply_speak_resets_state() {
        // A slow speak gives the canceller time to fire mid-call, like a barge-in.
        let fake = FakeS2s::with_delay(Duration::from_millis(1_000));
        let shared = Arc::new(Mutex::new(Box::new(fake) as Box<dyn S2sEngine>));
        let mut tts = S2sTts::new(shared, None, None);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let token = TokenChunk {
            turn: TurnId(5),
            generation,
            index: 0,
            text: "One. Two. Three.".into(),
            is_last: true,
        };
        let canceller = {
            let cancel = cancel.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                cancel.cancel_generation();
            })
        };
        let err = tts
            .synthesize_chunk_into(&token, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        canceller.join().expect("cancel thread");
        assert!(matches!(err, Error::Cancelled));
        assert!(tts.state.generation.is_none(), "cancel resets generation");
        assert!(
            tts.state.buffer.is_empty(),
            "cancel resets the reply buffer"
        );
    }
}
