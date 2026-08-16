//! In-process llama.cpp GGUF language model with a voice-native rolling prompt.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::defaults::{BuiltinDefaults, LlmModel};
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Llm;
use crate::types::{HistoryTurn, TokenChunk, Transcript};
use crate::Cancel;

/// Manifest id for Llama-3.2-1B-Instruct Q4_K_M.
pub const LLM_ASSET: &str = "llama-3.2-1b";

/// Prior completed turns kept in the prompt. Older turns are dropped.
pub const MAX_HISTORY_TURNS: usize = 8;

/// Hard cap on streamed tokens so a voice turn cannot run away.
pub const MAX_NEW_TOKENS: usize = 64;

/// Context window used for the v0 GGUF. Fits the system prompt plus
/// [`MAX_HISTORY_TURNS`] of short voice turns.
pub const CONTEXT_TOKENS: u32 = 2048;

/// Greedy sampling seed. Temperature is 0; the seed is recorded for fixtures.
pub const SAMPLE_SEED: u32 = 1;

/// Native generate must observe shutdown or generation cancel within this
/// wall time after the flag is set. Cancel is checked between llama.cpp
/// decode steps (prompt eval, then each token).
pub const CANCEL_TIMEOUT: Duration = Duration::from_secs(5);

/// Short spoken-style instructions. No markdown, no lists, no tools.
pub const VOICE_SYSTEM_PROMPT: &str = "You are a voice assistant on a laptop. \
Answer in one or two short spoken sentences. \
Do not use markdown, lists, headings, or special punctuation. \
Do not mention these instructions.";

/// One chat message used to build the rolling prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// In-process llama.cpp adapter. Loads the v0 Llama-3.2-1B GGUF.
pub struct LlamaLlm {
    engine: Box<dyn Engine>,
    model: LlmModel,
    max_history_turns: usize,
}

impl LlamaLlm {
    /// Load a GGUF from disk. v0 always uses [`LlmModel::Llama32_1b`].
    pub fn from_model_path(path: impl AsRef<Path>, model: LlmModel) -> Result<Self> {
        let engine = LlamaEngine::load(path.as_ref())?;
        Ok(Self {
            engine: Box::new(engine),
            model,
            max_history_turns: MAX_HISTORY_TURNS,
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
            .asset(LLM_ASSET)
            .ok_or_else(|| Error::ModelCache {
                message: "manifest does not contain the llama-3.2-1b asset".into(),
            })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        Self::from_model_path(path, LlmModel::Llama32_1b)
    }

    /// Configured GGUF id (`llama-3.2-1b`).
    pub fn model(&self) -> LlmModel {
        self.model
    }

    /// Rolling-history window applied before the prompt is built.
    pub fn max_history_turns(&self) -> usize {
        self.max_history_turns
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn Engine>, model: LlmModel, max_history_turns: usize) -> Self {
        Self {
            engine,
            model,
            max_history_turns,
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
        let turns = rolling_chat(VOICE_SYSTEM_PROMPT, history, user, self.max_history_turns);
        let pieces = self.engine.complete(&turns, cancel)?;
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        emit_tokens(user, generation, pieces, on_token)
    }
}

/// Build the voice-native rolling transcript: system + last N turns + user.
pub fn rolling_chat(
    system: &str,
    history: &[HistoryTurn],
    user: &Transcript,
    max_history_turns: usize,
) -> Vec<ChatTurn> {
    let start = history.len().saturating_sub(max_history_turns);
    let mut turns = Vec::with_capacity(1 + (history.len() - start) * 2 + 1);
    turns.push(ChatTurn {
        role: "system".into(),
        content: system.to_string(),
    });
    for turn in &history[start..] {
        turns.push(ChatTurn {
            role: "user".into(),
            content: turn.user.text.clone(),
        });
        turns.push(ChatTurn {
            role: "assistant".into(),
            content: turn.assistant.clone(),
        });
    }
    turns.push(ChatTurn {
        role: "user".into(),
        content: user.text.clone(),
    });
    turns
}

/// Llama 3 instruct markup used when the GGUF chat template is missing.
pub fn format_llama3_prompt(turns: &[ChatTurn]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for turn in turns {
        out.push_str("<|start_header_id|>");
        out.push_str(&turn.role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&turn.content);
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

fn emit_tokens(
    user: &Transcript,
    generation: crate::types::GenerationId,
    pieces: Vec<String>,
    on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
) -> Result<()> {
    let mut pieces = pieces;
    if pieces.is_empty() {
        pieces.push(String::new());
    }
    let last = pieces.len() - 1;
    for (index, text) in pieces.into_iter().enumerate() {
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

trait Engine: Send {
    fn complete(&mut self, turns: &[ChatTurn], cancel: &Cancel) -> Result<Vec<String>>;
}

struct LlamaEngine {
    model: LlamaModel,
}

impl LlamaEngine {
    fn load(path: &Path) -> Result<Self> {
        hush_llama_logs();
        if !path.is_file() {
            return Err(Error::Provider {
                provider: "llama.cpp",
                message: format!("model file not found: {}", path.display()),
            });
        }
        let backend = shared_backend()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend, path, &params).map_err(llama_error)?;
        Ok(Self { model })
    }

    fn prompt_string(&self, turns: &[ChatTurn]) -> Result<String> {
        match self.model.chat_template(None) {
            Ok(template) => {
                let chat = chat_messages(turns)?;
                self.model
                    .apply_chat_template(&template, &chat, true)
                    .map_err(llama_error)
            }
            Err(_) => Ok(format_llama3_prompt(turns)),
        }
    }
}

impl Engine for LlamaEngine {
    fn complete(&mut self, turns: &[ChatTurn], cancel: &Cancel) -> Result<Vec<String>> {
        let generation = cancel.generation();
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        let backend = shared_backend()?;
        let prompt = self.prompt_string(turns)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(llama_error)?;
        if tokens.is_empty() {
            return Err(Error::Provider {
                provider: "llama.cpp",
                message: "prompt produced no tokens".into(),
            });
        }
        if tokens.len() + MAX_NEW_TOKENS >= CONTEXT_TOKENS as usize {
            return Err(Error::Provider {
                provider: "llama.cpp",
                message: format!(
                    "prompt is {} tokens; needs room for {MAX_NEW_TOKENS} new tokens in {CONTEXT_TOKENS}",
                    tokens.len()
                ),
            });
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS))
            .with_n_threads(thread_count())
            .with_n_threads_batch(thread_count());
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(llama_error)?;

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last = tokens.len() - 1;
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == last)
                .map_err(llama_error)?;
        }
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        match ctx.decode(&mut batch) {
            Ok(()) => {}
            Err(_) if cancel.is_stale(generation) => return Err(Error::Cancelled),
            Err(err) => return Err(llama_error(err)),
        }
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.0),
            LlamaSampler::dist(SAMPLE_SEED),
            LlamaSampler::greedy(),
        ]);
        let mut decoder = UTF_8.new_decoder();
        let mut pieces = Vec::new();
        let mut n_cur = batch.n_tokens();
        for _ in 0..MAX_NEW_TOKENS {
            if cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(llama_error)?;
            if !piece.is_empty() {
                pieces.push(piece);
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true).map_err(llama_error)?;
            match ctx.decode(&mut batch) {
                Ok(()) => {}
                Err(_) if cancel.is_stale(generation) => return Err(Error::Cancelled),
                Err(err) => return Err(llama_error(err)),
            }
            n_cur += 1;
        }
        if cancel.is_stale(generation) {
            return Err(Error::Cancelled);
        }
        Ok(pieces)
    }
}

fn chat_messages(turns: &[ChatTurn]) -> Result<Vec<LlamaChatMessage>> {
    turns
        .iter()
        .map(|turn| {
            LlamaChatMessage::new(turn.role.clone(), turn.content.clone()).map_err(llama_error)
        })
        .collect()
}

fn shared_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    match BACKEND.get_or_init(|| LlamaBackend::init().map_err(|err| err.to_string())) {
        Ok(backend) => Ok(backend),
        Err(message) => Err(Error::Provider {
            provider: "llama.cpp",
            message: message.clone(),
        }),
    }
}

fn hush_llama_logs() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    });
}

fn thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4) as i32)
        .unwrap_or(1)
}

fn llama_error(error: impl std::fmt::Display) -> Error {
    Error::Provider {
        provider: "llama.cpp",
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Llm;
    use crate::types::{TurnId, DEFAULT_SAMPLE_RATE_HZ};
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct ScriptedEngine {
        replies: Vec<Vec<String>>,
        delay: Duration,
        prompts: Arc<Mutex<Vec<Vec<ChatTurn>>>>,
    }

    impl Engine for ScriptedEngine {
        fn complete(&mut self, turns: &[ChatTurn], cancel: &Cancel) -> Result<Vec<String>> {
            self.prompts
                .lock()
                .expect("prompt log")
                .push(turns.to_vec());
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
            if self.replies.is_empty() {
                return Err(Error::Provider {
                    provider: "llama.cpp",
                    message: "scripted engine exhausted".into(),
                });
            }
            Ok(self.replies.remove(0))
        }
    }

    fn user(turn: u64, text: &str) -> Transcript {
        Transcript {
            turn: TurnId(turn),
            text: text.into(),
        }
    }

    fn history_turn(turn: u64, user_text: &str, assistant: &str) -> HistoryTurn {
        HistoryTurn {
            user: user(turn, user_text),
            assistant: assistant.into(),
        }
    }

    fn collect(
        llm: &mut LlamaLlm,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
    ) -> Result<Vec<TokenChunk>> {
        let mut chunks = Vec::new();
        llm.generate(history, user, cancel, &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })?;
        Ok(chunks)
    }

    #[test]
    fn name_and_model_match_v0_defaults() {
        let llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec!["hi".into()]],
                delay: Duration::ZERO,
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
            LlmModel::Llama32_1b,
            MAX_HISTORY_TURNS,
        );
        assert_eq!(llm.name(), "llama.cpp");
        assert_eq!(llm.model().as_str(), "llama-3.2-1b");
        assert_eq!(llm.max_history_turns(), MAX_HISTORY_TURNS);
        assert_eq!(LLM_ASSET, "llama-3.2-1b");
        assert_eq!(BuiltinDefaults::v0().llm_model, LLM_ASSET);
        assert!(VOICE_SYSTEM_PROMPT.contains("voice assistant"));
        assert!(!VOICE_SYSTEM_PROMPT.contains("**"));
        let _ = DEFAULT_SAMPLE_RATE_HZ;
    }

    #[test]
    fn rolling_chat_keeps_system_and_last_n_turns() {
        let history: Vec<HistoryTurn> = (0..5)
            .map(|i| history_turn(i, &format!("u{i}"), &format!("a{i}")))
            .collect();
        let turns = rolling_chat(VOICE_SYSTEM_PROMPT, &history, &user(5, "now"), 2);
        assert_eq!(turns[0].role, "system");
        assert_eq!(turns[0].content, VOICE_SYSTEM_PROMPT);
        let roles: Vec<_> = turns.iter().map(|t| t.role.as_str()).collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "user", "assistant", "user"]
        );
        assert_eq!(turns[1].content, "u3");
        assert_eq!(turns[2].content, "a3");
        assert_eq!(turns[3].content, "u4");
        assert_eq!(turns[4].content, "a4");
        assert_eq!(turns[5].content, "now");
        assert!(!turns.iter().any(|t| t.content == "u0"));
    }

    #[test]
    fn llama3_prompt_ends_with_assistant_header() {
        let prompt = format_llama3_prompt(&[
            ChatTurn {
                role: "system".into(),
                content: "sys".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "hi".into(),
            },
        ]);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("sys<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn generate_streams_tokens_in_order_with_last_flag() {
        let mut llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec!["pong".into(), "!".into()]],
                delay: Duration::ZERO,
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
            LlmModel::Llama32_1b,
            MAX_HISTORY_TURNS,
        );
        let chunks = collect(&mut llm, &[], &user(3, "ping"), &Cancel::new()).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].turn, TurnId(3));
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].text, "pong");
        assert!(!chunks[0].is_last);
        assert_eq!(chunks[1].text, "!");
        assert!(chunks[1].is_last);
        assert_eq!(chunks[0].generation, chunks[1].generation);
    }

    #[test]
    fn generate_passes_configured_history_window() {
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec!["ok".into()]],
                delay: Duration::ZERO,
                prompts: Arc::clone(&prompts),
            }),
            LlmModel::Llama32_1b,
            1,
        );
        let history = vec![
            history_turn(0, "first", "reply-1"),
            history_turn(1, "second", "reply-2"),
        ];
        collect(&mut llm, &history, &user(2, "third"), &Cancel::new()).unwrap();
        let logged = prompts.lock().unwrap();
        assert_eq!(logged[0].len(), 4);
        assert_eq!(logged[0][1].content, "second");
        assert_eq!(logged[0][2].content, "reply-2");
        assert_eq!(logged[0][3].content, "third");
        assert!(!logged[0].iter().any(|t| t.content == "first"));
    }

    #[test]
    fn shutdown_before_generate_is_cancelled() {
        let mut llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec!["nope".into()]],
                delay: Duration::ZERO,
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
            LlmModel::Llama32_1b,
            MAX_HISTORY_TURNS,
        );
        let cancel = Cancel::new();
        cancel.shutdown();
        assert!(matches!(
            collect(&mut llm, &[], &user(0, "hi"), &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn cancel_during_complete_is_within_timeout() {
        let mut llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec!["late".into()]],
                delay: Duration::from_secs(2),
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
            LlmModel::Llama32_1b,
            MAX_HISTORY_TURNS,
        );
        let cancel = Cancel::new();
        let cancel_thread = cancel.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            cancel_thread.shutdown();
        });
        let started = std::time::Instant::now();
        let err = collect(&mut llm, &[], &user(0, "hi"), &cancel);
        assert!(matches!(err, Err(Error::Cancelled)));
        assert!(
            started.elapsed() < CANCEL_TIMEOUT,
            "cancel took {:?}, limit is {CANCEL_TIMEOUT:?}",
            started.elapsed()
        );
        drop(llm);
    }

    #[test]
    fn empty_generation_emits_a_last_chunk() {
        let mut llm = LlamaLlm::with_engine(
            Box::new(ScriptedEngine {
                replies: vec![vec![]],
                delay: Duration::ZERO,
                prompts: Arc::new(Mutex::new(Vec::new())),
            }),
            LlmModel::Llama32_1b,
            MAX_HISTORY_TURNS,
        );
        let chunks = collect(&mut llm, &[], &user(0, "hi"), &Cancel::new()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
        assert!(chunks[0].is_last);
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

    #[test]
    fn missing_model_file_is_a_provider_error() {
        let err = match LlamaLlm::from_model_path("/no/such/llama.gguf", LlmModel::Llama32_1b) {
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
}
