//! In-process llama.cpp GGUF language model.

use std::os::raw::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use syllabix_native::{ChatMessage, LlamaContext, LlamaError, LlamaGenerate};

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::fake::LlmCall;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Llm;
use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, Transcript};

/// Manifest id for the default Qwen3.5 0.8B instruct GGUF.
pub const QWEN35_08B_ASSET: &str = "qwen3.5-0.8b";

/// Manifest id for the yaml-only Qwen3.5 2B instruct GGUF.
pub const QWEN35_2B_ASSET: &str = "qwen3.5-2b";

/// Manifest id for the yaml-only Llama 3.2 1B instruct GGUF.
pub const LLAMA_32_1B_ASSET: &str = "llama-3.2-1b";

/// `{language}` in a yaml `system_prompt` is replaced with the STT language's
/// English name (`French`, `German`, …) at generate time. Unknown or `auto`
/// codes (which never reach the LLM in a real run) stay `English`.
pub const LANGUAGE_PLACEHOLDER: &str = "{language}";

/// Default system prompt template. Written by `init`; used when yaml omits
/// `pipeline.llm.system_prompt`. Shared by `local` GGUF and `online`
/// chat/completions: the reply is still spoken, so the prompt is voice-native
/// rather than naming where the weights run.
pub const VOICE_SYSTEM_PROMPT_TEMPLATE: &str = "You are a smart assistant. This is a spoken conversation. Reply in spoken {language}, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji.";

/// English rendering of [`VOICE_SYSTEM_PROMPT_TEMPLATE`]. Zero-config `run`
/// with STT `en` sends this byte-for-byte.
pub const VOICE_SYSTEM_PROMPT: &str = "You are a smart assistant. This is a spoken conversation. Reply in spoken English, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji.";

/// Render a system-prompt template for the turn's STT language.
///
/// `{language}` is substituted when present. Custom yaml that omits the
/// placeholder still gets a spoken-language pin for a known non-English
/// code, so STT `language:` / `auto` keep working. Unknown/`auto` codes
/// leave a placeholder-free template unchanged.
pub fn render_system_prompt(template: &str, language: &str) -> String {
    let name = crate::language::language_name(language).unwrap_or("English");
    if template.contains(LANGUAGE_PLACEHOLDER) {
        template.replace(LANGUAGE_PLACEHOLDER, name)
    } else if language != crate::stt::STT_LANGUAGE
        && crate::language::language_name(language).is_some()
    {
        format!("{template} Reply in spoken {name}.")
    } else {
        template.to_string()
    }
}

/// System prompt for the turn's STT language using the launch default template.
pub fn system_prompt_for(language: &str) -> String {
    render_system_prompt(VOICE_SYSTEM_PROMPT_TEMPLATE, language)
}

/// Rolling history kept in the prompt.
pub const LLAMA_MAX_HISTORY_TURNS: usize = 8;

/// Native cancel must surface as [`Error::Cancelled`] within this window.
/// Full GGUF `n_ctx` makes one `llama_decode` heavier than the old 2048-slot cap.
pub const LLAMA_CANCEL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// End a stalled local generation promptly, rather than leaving the TUI on an
/// unfinished turn. Each emitted token resets this timer.
pub const LLAMA_TOKEN_STALL_TIMEOUT: Duration = Duration::from_secs(2);

/// True when `id` is a v0 llama.cpp GGUF.
pub fn is_v0_llm_model(id: &str) -> bool {
    id == QWEN35_08B_ASSET || id == QWEN35_2B_ASSET || id == LLAMA_32_1B_ASSET
}

/// In-process llama.cpp adapter. Loads a v0 Q4_K_M GGUF.
pub struct LlamaLlm {
    engine: Arc<Mutex<Box<dyn Engine>>>,
    calls: Arc<Mutex<Vec<LlmCall>>>,
    thinking: bool,
    model_id: String,
    system_prompt: String,
}

impl Clone for LlamaLlm {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            calls: Arc::clone(&self.calls),
            thinking: self.thinking,
            model_id: self.model_id.clone(),
            system_prompt: self.system_prompt.clone(),
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
            thinking: false,
            model_id: path
                .as_ref()
                .file_stem()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            system_prompt: VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
        })
    }

    /// Resolve the default `llama-3.2-1b` GGUF from the manifest cache, then load it.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        Self::from_cached_model(
            cache,
            fetcher,
            progress,
            cancel,
            BuiltinDefaults::v0().llm_model,
            BuiltinDefaults::v0().llm_thinking,
        )
    }

    /// Resolve one v0 GGUF id and load it. Does not fetch the other size.
    pub fn from_cached_model(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        model_id: &str,
        thinking: bool,
    ) -> Result<Self> {
        if !is_v0_llm_model(model_id) {
            return Err(Error::Config {
                field: "pipeline.llm.model".into(),
                message: format!(
                    "unsupported value {model_id:?} (allowed: llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b)"
                ),
            });
        }
        let asset = cache
            .manifest()
            .asset(model_id)
            .ok_or_else(|| Error::ModelCache {
                message: format!("manifest does not contain the {model_id} asset"),
            })?;
        let path = cache.resolve(asset, fetcher, progress, cancel)?;
        let mut llm = Self::from_model_path(path)?;
        llm.thinking = thinking;
        llm.model_id = model_id.to_string();
        Ok(llm)
    }

    /// Shared call log. Clone the `Arc` before moving the LLM into the pipeline.
    pub fn call_log(&self) -> Arc<Mutex<Vec<LlmCall>>> {
        Arc::clone(&self.calls)
    }

    /// Thinking flag used for this load.
    pub fn thinking(&self) -> bool {
        self.thinking
    }

    /// Yaml `pipeline.llm.system_prompt` (or the launch default template).
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// `(n_ctx, n_ctx_train)` after a real GGUF load.
    pub fn context_window(&self) -> Option<(i32, i32)> {
        self.engine.lock().expect("llama engine").context_window()
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn Engine>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            calls: Arc::new(Mutex::new(Vec::new())),
            thinking: false,
            model_id: BuiltinDefaults::v0().llm_model.to_string(),
            system_prompt: VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
        }
    }

    #[cfg(test)]
    fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }
}

impl Llm for LlamaLlm {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().llm.as_str()
    }

    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        Some(LlmDebugMeta {
            provider: BuiltinDefaults::v0().llm.as_str().into(),
            model: self.model_id.clone(),
            endpoint: String::new(),
            request_id: String::new(),
        })
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
            content: render_system_prompt(&self.system_prompt, &user.language),
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
        let append_thinking_off_suffix =
            !self.thinking && is_thinking_tag_supported_model(&self.model_id);
        self.engine.lock().expect("llama engine").generate(
            &messages,
            append_thinking_off_suffix,
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
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()>;

    fn context_window(&self) -> Option<(i32, i32)> {
        None
    }
}

struct LlamaEngine {
    ctx: LlamaContext,
}

impl LlamaEngine {
    fn load(path: &Path) -> Result<Self> {
        let ctx = LlamaContext::load(path, thread_count()).map_err(|message| Error::Provider {
            provider: crate::defaults::BuiltinDefaults::v0().llm.as_str(),
            message,
        })?;
        Ok(Self { ctx })
    }
}

impl Engine for LlamaEngine {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort = GenerationAbort::new(cancel);
        let abort_user = (&abort as *const GenerationAbort).cast_mut().cast();
        match unsafe {
            self.ctx.generate(
                messages,
                LlamaGenerate {
                    append_thinking_off_suffix,
                    n_threads: thread_count(),
                },
                Some(abort_on_stall),
                abort_user,
                &mut |text, is_last| {
                    abort.note_token();
                    match on_piece(text, is_last) {
                        Ok(()) => Ok(()),
                        Err(Error::Cancelled) => Err(LlamaError::Cancelled),
                        Err(err) => Err(LlamaError::Failed(err.to_string())),
                    }
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
            Err(LlamaError::Cancelled) if abort.timed_out() => {
                // A terminal empty chunk lets TTS emit its tiny completion
                // chunk, so the pipeline records timings and returns to
                // listening instead of leaving the TUI mid-turn.
                on_piece("", true)
            }
            Err(LlamaError::Cancelled) => Err(Error::Cancelled),
            Err(LlamaError::Failed(_)) if cancel.is_shutdown() => Err(Error::Cancelled),
            Err(LlamaError::Failed(message)) => Err(Error::Provider {
                provider: crate::defaults::BuiltinDefaults::v0().llm.as_str(),
                message,
            }),
        }
    }

    fn context_window(&self) -> Option<(i32, i32)> {
        Some((self.ctx.n_ctx(), self.ctx.n_ctx_train()))
    }
}

struct GenerationAbort<'a> {
    cancel: &'a Cancel,
    last_token: Mutex<Instant>,
    timed_out: AtomicBool,
}

impl<'a> GenerationAbort<'a> {
    fn new(cancel: &'a Cancel) -> Self {
        Self {
            cancel,
            last_token: Mutex::new(Instant::now()),
            timed_out: AtomicBool::new(false),
        }
    }

    fn note_token(&self) {
        *self.last_token.lock().expect("llama token timer") = Instant::now();
    }

    fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::SeqCst)
    }
}

unsafe extern "C" fn abort_on_stall(user_data: *mut c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    // Safety: `user_data` is a `GenerationAbort` for the duration of this
    // llama.cpp call.
    let state = unsafe { &*(user_data as *const GenerationAbort<'_>) };
    if state.cancel.is_shutdown() {
        return true;
    }
    if state
        .last_token
        .lock()
        .expect("llama token timer")
        .elapsed()
        >= LLAMA_TOKEN_STALL_TIMEOUT
    {
        state.timed_out.store(true, Ordering::SeqCst);
        return true;
    }
    false
}

fn thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4) as i32)
        .unwrap_or(1)
}

fn is_thinking_tag_supported_model(model_id: &str) -> bool {
    matches!(model_id, QWEN35_08B_ASSET | QWEN35_2B_ASSET)
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
            _thinking: bool,
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
            language: crate::stt::STT_LANGUAGE.to_string(),
        }
    }

    fn user_in(turn: u64, text: &str, language: &str) -> Transcript {
        Transcript {
            turn: TurnId(turn),
            text: text.into(),
            language: language.into(),
        }
    }

    #[test]
    fn system_prompt_pins_the_reply_language() {
        assert_eq!(
            render_system_prompt(VOICE_SYSTEM_PROMPT_TEMPLATE, "en"),
            VOICE_SYSTEM_PROMPT
        );
        assert_eq!(system_prompt_for("en"), VOICE_SYSTEM_PROMPT);
        assert_eq!(
            system_prompt_for("fr"),
            "You are a smart assistant. This is a spoken conversation. Reply in spoken French, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji."
        );
        assert_eq!(system_prompt_for("de"), system_prompt_for("de"));
        assert!(system_prompt_for("de").contains("German"));
        assert!(system_prompt_for("yue").contains("Cantonese"));
        // Unknown codes (and `auto`, which a real run never forwards) fall
        // back to the English prompt instead of speaking a broken sentence.
        assert_eq!(system_prompt_for("auto"), VOICE_SYSTEM_PROMPT);
        assert_eq!(system_prompt_for("klingon"), VOICE_SYSTEM_PROMPT);
        assert!(!system_prompt_for("fr").contains("**"));
        assert!(!VOICE_SYSTEM_PROMPT.contains(LANGUAGE_PLACEHOLDER));
        assert!(VOICE_SYSTEM_PROMPT_TEMPLATE.contains(LANGUAGE_PLACEHOLDER));
    }

    #[test]
    fn yaml_system_prompt_keeps_persona_and_still_pins_language() {
        let custom = "You are a cooking coach. Keep answers short.";
        assert_eq!(
            render_system_prompt(custom, "en"),
            custom,
            "English (and unknown codes) leave a placeholder-free template alone"
        );
        assert_eq!(
            render_system_prompt(custom, "fr"),
            "You are a cooking coach. Keep answers short. Reply in spoken French."
        );
        assert_eq!(
            render_system_prompt("Speak in {language} only.", "ja"),
            "Speak in Japanese only."
        );
        assert_eq!(render_system_prompt(custom, "auto"), custom);
    }

    #[test]
    fn name_and_prompt_match_v0() {
        let llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["hi".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        assert_eq!(llm.name(), "local");
        assert_eq!(LLAMA_32_1B_ASSET, BuiltinDefaults::v0().llm_model);
        assert_eq!(QWEN35_08B_ASSET, "qwen3.5-0.8b");
        assert_eq!(QWEN35_2B_ASSET, "qwen3.5-2b");
        assert_eq!(LLAMA_32_1B_ASSET, "llama-3.2-1b");
        assert!(is_v0_llm_model(QWEN35_08B_ASSET));
        assert!(is_v0_llm_model(QWEN35_2B_ASSET));
        assert!(is_v0_llm_model(LLAMA_32_1B_ASSET));
        assert!(!is_v0_llm_model("mistral"));
        assert!(VOICE_SYSTEM_PROMPT.contains("smart assistant"));
        assert!(VOICE_SYSTEM_PROMPT.contains("spoken"));
        assert!(!VOICE_SYSTEM_PROMPT.contains("**"));
        assert_eq!(LLAMA_CANCEL_TIMEOUT, Duration::from_secs(5));
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
    fn configured_system_prompt_is_sent_to_the_engine() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::clone(&messages),
        }))
        .with_system_prompt("You are a cooking coach. Reply in spoken {language}.");
        llm.generate(
            &[],
            &user_in(1, "hi", "es"),
            &Cancel::new(),
            &mut |_| Ok(()),
        )
        .unwrap();
        let prompt = messages.lock().unwrap();
        assert_eq!(
            prompt[0].content,
            "You are a cooking coach. Reply in spoken Spanish."
        );
    }

    #[test]
    fn detected_language_pins_into_the_system_prompt() {
        let messages = Arc::new(Mutex::new(Vec::new()));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["Bonjour".into()],
            delay: Duration::ZERO,
            last_messages: Arc::clone(&messages),
        }));
        llm.generate(
            &[],
            &user_in(7, "bonjour", "fr"),
            &Cancel::new(),
            &mut |_| Ok(()),
        )
        .unwrap();
        let prompt = messages.lock().unwrap();
        assert_eq!(prompt[0].role, "system");
        assert_eq!(
            prompt[0].content,
            system_prompt_for("fr"),
            "the LLM must be told to reply in the transcript's language"
        );
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
        let err = match LlamaLlm::from_model_path("/no/such/Qwen3.5-2B-Q4_K_M.gguf") {
            Err(err) => err,
            Ok(_) => panic!("missing model path should fail"),
        };
        assert!(matches!(
            err,
            Error::Provider {
                provider: "local",
                ..
            }
        ));
    }

    #[test]
    fn empty_engine_still_emits_a_last_token() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec![],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
        assert!(chunks[0].text.is_empty());
    }

    #[test]
    fn stale_generation_during_callback_cancels() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["a".into(), "b".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let cancel = Cancel::new();
        let err = llm.generate(&[], &user(0, "hi"), &cancel, &mut |chunk| {
            if chunk.index == 0 {
                cancel.cancel_generation();
            }
            Ok(())
        });
        assert!(matches!(err, Err(Error::Cancelled)));
    }

    #[test]
    fn clone_shares_call_log() {
        let llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let log = llm.call_log();
        let mut cloned = llm.clone();
        cloned
            .generate(&[], &user(0, "ping"), &Cancel::new(), &mut |_| Ok(()))
            .unwrap();
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(log.lock().unwrap()[0].user_text, "ping");
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
    fn thinking_flag_does_not_cap_generation() {
        let off = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: (0..200).map(|i| format!("t{i}")).collect(),
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let on = off.clone().with_thinking(true);
        assert!(!off.thinking());
        assert!(on.thinking());
        assert!(off.context_window().is_none());
        let mut chunks = Vec::new();
        let mut llm = off;
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
        assert_eq!(chunks.len(), 200);
        assert!(chunks.last().unwrap().is_last);
    }

    #[test]
    fn only_thinking_tag_models_get_the_thinking_off_prompt_suffix() {
        assert!(is_thinking_tag_supported_model(QWEN35_08B_ASSET));
        assert!(is_thinking_tag_supported_model(QWEN35_2B_ASSET));
        assert!(!is_thinking_tag_supported_model(LLAMA_32_1B_ASSET));
    }

    #[test]
    fn stalled_abort_is_reported() {
        let cancel = Cancel::new();
        let state = GenerationAbort::new(&cancel);
        *state.last_token.lock().unwrap() = Instant::now() - LLAMA_TOKEN_STALL_TIMEOUT;
        assert!(unsafe { abort_on_stall((&state as *const GenerationAbort).cast_mut().cast()) });
        assert!(state.timed_out());
    }

    #[test]
    fn unknown_cached_model_id_is_config() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-llm-unknown-{}-{}",
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
        let err = match LlamaLlm::from_cached_model(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            "mistral",
            true,
        ) {
            Err(err) => err,
            Ok(_) => panic!("unknown model should fail"),
        };
        assert!(matches!(err, Error::Config { .. }));
    }
}
