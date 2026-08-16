//! In-process llama.cpp GGUF language model.

use std::os::raw::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

use syllabix_native::{ChatMessage, LlamaContext, LlamaError};

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::fake::LlmCall;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Llm;
use crate::types::{HistoryTurn, TokenChunk, Transcript};

/// Manifest id for the v0 Llama 3.2 1B instruct GGUF.
pub const LLAMA_1B_ASSET: &str = "llama-3.2-1b";

/// Short spoken-English system prompt. No Markdown.
pub const VOICE_SYSTEM_PROMPT: &str = "You are a voice assistant on a laptop. Reply in short spoken English, one or two sentences. Do not use markdown, lists, headings, or emoji.";

/// CPU context length for v0. Fits the 1B Q4_K_M GGUF on CI runners.
pub const LLAMA_N_CTX: i32 = 2048;

/// Greedy generation cap. Voice replies stay short.
pub const LLAMA_N_PREDICT: i32 = 96;

/// Oldest rolling turns kept in the prompt.
pub const LLAMA_MAX_HISTORY_TURNS: usize = 8;

/// Native cancel must surface as [`Error::Cancelled`] within this window.
pub const LLAMA_CANCEL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// In-process llama.cpp adapter. Loads the v0 Q4_K_M GGUF.
pub struct LlamaLlm {
    engine: Arc<Mutex<Box<dyn Engine>>>,
    calls: Arc<Mutex<Vec<LlmCall>>>,
}

impl Clone for LlamaLlm {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            calls: Arc::clone(&self.calls),
        }
    }
}

impl LlamaLlm {
    /// Load a GGUF from disk.
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self> {
        let engine = LlamaEngine::load(path.as_ref())?;
        Ok(Self {
            engine: Arc::new(Mutex::new(Box::new(engine))),
            calls: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Resolve `llama-3.2-1b` from the manifest cache, then load it.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        let asset = cache
            .manifest()
            .asset(LLAMA_1B_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the llama-3.2-1b asset".into(),
            })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_model_path(path)
    }

    /// Shared call log. Clone the `Arc` before moving the LLM into the pipeline.
    pub fn call_log(&self) -> Arc<Mutex<Vec<LlmCall>>> {
        Arc::clone(&self.calls)
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn Engine>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Llm for LlamaLlm {
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
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let generation = cancel.generation();
        {
            let mut calls = self.calls.lock().expect("llm call log");
            calls.push(LlmCall {
                history_len: history.len(),
                history_user_texts: history.iter().map(|t| t.user.text.clone()).collect(),
                user_text: user.text.clone(),
            });
        }

        let kept = if history.len() > LLAMA_MAX_HISTORY_TURNS {
            &history[history.len() - LLAMA_MAX_HISTORY_TURNS..]
        } else {
            history
        };
        let mut messages = Vec::with_capacity(2 + kept.len() * 2);
        messages.push(ChatMessage {
            role: "system".into(),
            content: VOICE_SYSTEM_PROMPT.into(),
        });
        for turn in kept {
            messages.push(ChatMessage {
                role: "user".into(),
                content: turn.user.text.clone(),
            });
            messages.push(ChatMessage {
                role: "assistant".into(),
                content: turn.assistant.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".into(),
            content: user.text.clone(),
        });

        let mut index = 0u32;
        self.engine.lock().expect("llama engine").generate(
            &messages,
            cancel,
            &mut |text, is_last| {
                if cancel.is_stale(generation) {
                    return Err(Error::Cancelled);
                }
                on_token(TokenChunk {
                    turn: user.turn,
                    generation,
                    index,
                    text: text.to_string(),
                    is_last,
                })?;
                index += 1;
                Ok(())
            },
        )?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

trait Engine: Send {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()>;
}

struct LlamaEngine {
    ctx: LlamaContext,
}

impl LlamaEngine {
    fn load(path: &Path) -> Result<Self> {
        let ctx = LlamaContext::load(path, LLAMA_N_CTX, thread_count()).map_err(|message| {
            Error::Provider {
                provider: "llama.cpp",
                message,
            }
        })?;
        Ok(Self { ctx })
    }
}

impl Engine for LlamaEngine {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort_user = cancel as *const Cancel as *mut c_void;
        match unsafe {
            self.ctx.generate(
                messages,
                LLAMA_N_PREDICT,
                thread_count(),
                Some(abort_on_shutdown),
                abort_user,
                &mut |text, is_last| match on_piece(text, is_last) {
                    Ok(()) => Ok(()),
                    Err(Error::Cancelled) => Err(LlamaError::Cancelled),
                    Err(err) => Err(LlamaError::Failed(err.to_string())),
                },
            )
        } {
            Ok(()) => {
                if cancel.is_shutdown() {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            }
            Err(LlamaError::Cancelled) => Err(Error::Cancelled),
            Err(LlamaError::Failed(_)) if cancel.is_shutdown() => Err(Error::Cancelled),
            Err(LlamaError::Failed(message)) => Err(Error::Provider {
                provider: "llama.cpp",
                message,
            }),
        }
    }
}

unsafe extern "C" fn abort_on_shutdown(user_data: *mut c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // Safety: `user_data` is `&Cancel` for the duration of `llama_decode`.
    unsafe { (*(user_data as *const Cancel)).is_shutdown() }
}

fn thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4) as i32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TurnId;
    use std::thread;
    use std::time::Duration;

    struct ScriptedEngine {
        pieces: Vec<String>,
        delay: Duration,
        last_messages: Arc<Mutex<Vec<ChatMessage>>>,
    }

    impl Engine for ScriptedEngine {
        fn generate(
            &mut self,
            messages: &[ChatMessage],
            cancel: &Cancel,
            on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
        ) -> Result<()> {
            *self.last_messages.lock().expect("messages") = messages.to_vec();
            if !self.delay.is_zero() {
                let start = std::time::Instant::now();
                while start.elapsed() < self.delay {
                    if cancel.is_shutdown() {
                        return Err(Error::Cancelled);
                    }
                    thread::sleep(Duration::from_millis(5));
                }
            }
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let last = self.pieces.len().saturating_sub(1);
            if self.pieces.is_empty() {
                on_piece("", true)?;
                return Ok(());
            }
            for (i, piece) in self.pieces.iter().enumerate() {
                on_piece(piece, i == last)?;
            }
            Ok(())
        }
    }

    fn user(turn: u64, text: &str) -> Transcript {
        Transcript {
            turn: TurnId(turn),
            text: text.into(),
        }
    }

    #[test]
    fn name_and_prompt_match_v0() {
        let llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["hi".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        assert_eq!(llm.name(), "llama.cpp");
        assert_eq!(LLAMA_1B_ASSET, BuiltinDefaults::v0().llm_model);
        assert!(VOICE_SYSTEM_PROMPT.contains("voice"));
        assert!(!VOICE_SYSTEM_PROMPT.contains("**"));
        assert_eq!(LLAMA_CANCEL_TIMEOUT, Duration::from_secs(2));
    }

    #[test]
    fn streams_tokens_with_last_flag_and_system_prompt() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["Hel".into(), "lo".into()],
            delay: Duration::ZERO,
            last_messages: Arc::clone(&messages),
        }));
        let mut chunks = Vec::new();
        llm.generate(&[], &user(3, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "Hel");
        assert!(!chunks[0].is_last);
        assert_eq!(chunks[1].text, "lo");
        assert!(chunks[1].is_last);
        assert_eq!(chunks[0].turn, TurnId(3));
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[1].index, 1);
        let prompt = messages.lock().unwrap();
        assert_eq!(prompt[0].role, "system");
        assert_eq!(prompt[0].content, VOICE_SYSTEM_PROMPT);
        assert_eq!(prompt[1].role, "user");
        assert_eq!(prompt[1].content, "hi");
    }

    #[test]
    fn rolling_history_is_passed_in_order() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::clone(&messages),
        }));
        let history = vec![HistoryTurn {
            user: user(0, "first"),
            assistant: "reply-one".into(),
        }];
        llm.generate(
            &history,
            &user(1, "second"),
            &Cancel::new(),
            &mut |_| Ok(()),
        )
        .unwrap();
        let log = llm.call_log();
        let calls = log.lock().unwrap();
        assert_eq!(calls[0].history_len, 1);
        assert_eq!(calls[0].history_user_texts, vec!["first".to_string()]);
        let prompt = messages.lock().unwrap();
        assert_eq!(prompt[1].role, "user");
        assert_eq!(prompt[1].content, "first");
        assert_eq!(prompt[2].role, "assistant");
        assert_eq!(prompt[2].content, "reply-one");
        assert_eq!(prompt[3].role, "user");
        assert_eq!(prompt[3].content, "second");
    }

    #[test]
    fn trims_history_to_the_last_eight_turns() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::clone(&messages),
        }));
        let history: Vec<HistoryTurn> = (0..10)
            .map(|i| HistoryTurn {
                user: user(i, &format!("u{i}")),
                assistant: format!("a{i}"),
            })
            .collect();
        llm.generate(&history, &user(10, "now"), &Cancel::new(), &mut |_| Ok(()))
            .unwrap();
        let prompt = messages.lock().unwrap();
        let users: Vec<&str> = prompt
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(users.first(), Some(&"u2"));
        assert_eq!(users.last(), Some(&"now"));
        assert_eq!(users.len(), 9);
    }

    #[test]
    fn shutdown_before_generate_is_cancelled() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["nope".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let cancel = Cancel::new();
        cancel.shutdown();
        assert!(matches!(
            llm.generate(&[], &user(0, "hi"), &cancel, &mut |_| Ok(())),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn cancel_during_generate_is_cancelled() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["late".into()],
            delay: Duration::from_secs(2),
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let cancel = Cancel::new();
        let cancel_thread = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            cancel_thread.shutdown();
        });
        let err = llm.generate(&[], &user(0, "hi"), &cancel, &mut |_| Ok(()));
        assert!(matches!(err, Err(Error::Cancelled)));
    }

    #[test]
    fn missing_model_file_is_a_provider_error() {
        let err = match LlamaLlm::from_model_path("/no/such/Llama-3.2-1B-Instruct-Q4_K_M.gguf") {
            Err(err) => err,
            Ok(_) => panic!("missing model path should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: "llama.cpp",
                ..
            }
        ));
    }

    #[test]
    fn from_cache_requires_llama_asset() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-llm-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(
            root,
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match LlamaLlm::from_cache(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, Error::ModelCache { .. }));
        assert!(err.to_string().contains("llama-3.2-1b"));
    }
}
