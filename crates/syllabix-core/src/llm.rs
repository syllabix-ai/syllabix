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
use crate::types::{
    HistoryTurn, LlmDebugMeta, TokenChunk, ToolCall, ToolResult, ToolTurnEvent, Transcript,
};

/// Manifest id for the default Qwen3.5 0.8B instruct GGUF.
pub const QWEN35_08B_ASSET: &str = "qwen3.5-0.8b";

/// Manifest id for the yaml-only Qwen3.5 2B instruct GGUF.
pub const QWEN35_2B_ASSET: &str = "qwen3.5-2b";

/// Manifest id for the yaml-only Llama 3.2 1B instruct GGUF.
pub const LLAMA_32_1B_ASSET: &str = "llama-3.2-1b";

/// Manifest id for the LiquidAI LFM2.5-2.6B QAD Q4_0 GGUF.
/// It is the default local `pipeline.llm.model`; the other three ids are yaml opt-ins.
pub const LFM25_26B_ASSET: &str = "lfm2.5-2.6b";

/// Upper bound on collected text for one LFM tool turn (issue-89 lesson).
/// LFM always thinks, so a turn that never emits a complete
/// `<|tool_call_start|>` block must fail closed into the spoken fallback
/// instead of generating until cancel. The bound lives in the adapter, not
/// in yaml (EOS-generate contract stands).
pub const LFM_TOOL_TURN_MAX_CHARS: usize = 8_000;

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
/// unfinished turn. Each emitted token resets this timer, re-armed after the
/// downstream consumer returns so slow TTS backpressure is never mistaken
/// for a model stall.
pub const LLAMA_TOKEN_STALL_TIMEOUT: Duration = Duration::from_secs(4);

/// True when `id` is a v0 llama.cpp GGUF.
pub fn is_v0_llm_model(id: &str) -> bool {
    id == QWEN35_08B_ASSET
        || id == QWEN35_2B_ASSET
        || id == LLAMA_32_1B_ASSET
        || id == LFM25_26B_ASSET
}

/// In-process llama.cpp adapter. Loads a v0 Q4_K_M GGUF.
pub struct LlamaLlm {
    engine: Arc<Mutex<Box<dyn Engine>>>,
    calls: Arc<Mutex<Vec<LlmCall>>>,
    thinking: bool,
    model_id: String,
    system_prompt: String,
    tool_events: Arc<Mutex<Vec<ToolTurnEvent>>>,
}

impl Clone for LlamaLlm {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            calls: Arc::clone(&self.calls),
            thinking: self.thinking,
            model_id: self.model_id.clone(),
            system_prompt: self.system_prompt.clone(),
            tool_events: Arc::clone(&self.tool_events),
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
            tool_events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Resolve the default `lfm2.5-2.6b` GGUF from the manifest cache, then load it.
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
                    "unsupported value {model_id:?} (allowed: lfm2.5-2.6b, llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b)"
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

    /// Exact prompt-token count for the configured chat template. This is
    /// contributor-benchmark evidence, not a runtime token budget.
    pub fn benchmark_prompt_tokens(
        &self,
        history: &[HistoryTurn],
        user: &Transcript,
    ) -> Result<usize> {
        let messages = self.messages(history, user);
        self.engine
            .lock()
            .expect("llama engine")
            .prompt_token_count(
                &messages,
                !self.thinking && is_thinking_tag_supported_model(&self.model_id),
            )
    }

    fn messages(&self, history: &[HistoryTurn], user: &Transcript) -> Vec<ChatMessage> {
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
        messages
    }

    #[cfg(test)]
    fn with_engine(engine: Box<dyn Engine>) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            calls: Arc::new(Mutex::new(Vec::new())),
            thinking: false,
            model_id: BuiltinDefaults::v0().llm_model.to_string(),
            system_prompt: VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
            tool_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    #[cfg(test)]
    fn with_thinking(mut self, thinking: bool) -> Self {
        self.thinking = thinking;
        self
    }

    fn note_tool_event(
        &self,
        kind: &str,
        name: &str,
        call_id: &str,
        arguments: &str,
        content: &str,
    ) {
        self.tool_events
            .lock()
            .expect("llama tool-events mutex")
            .push(ToolTurnEvent {
                kind: kind.into(),
                name: name.into(),
                call_id: call_id.into(),
                arguments: arguments.into(),
                content: content.into(),
            });
    }

    /// Prompt messages for a tools-aware turn: the normal history plus one
    /// `tool`-role message per executed result, in order. The Qwen renderer
    /// groups them into native `<tool_response>` turns; the online adapter's
    /// `tool`-role result messages carry the same facts, so the pipeline
    /// cannot tell which backend ran.
    fn tool_messages(
        &self,
        history: &[HistoryTurn],
        user: &Transcript,
        results: &[ToolResult],
    ) -> Vec<ChatMessage> {
        let mut messages = self.messages(history, user);
        for result in results {
            messages.push(ChatMessage {
                role: "tool".into(),
                content: result.content.clone(),
            });
        }
        messages
    }

    /// One tools-aware turn against the local Qwen template (issue 87).
    ///
    /// Adapter half of the shared tool-call contract: renders the `<tools>`
    /// preamble through the shim's tools-aware entry, collects the turn text,
    /// then parses and normalizes Qwen `<tool_call>` blocks into [`ToolCall`]s
    /// with `call` / `rejected` events for [`Llm::take_tool_events`].
    /// Executing the calls stays with the Phase-4 loop (host-owned executor);
    /// this never runs a tool. Not on the default `run` path: [`Llm::generate`]
    /// stays tool-free and byte-identical.
    pub fn generate_tool_turn(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<Vec<ToolCall>> {
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
        let messages = self.tool_messages(history, user, &[]);
        let tools = local_tool_definitions_json();
        let append_thinking_off_suffix =
            !self.thinking && is_thinking_tag_supported_model(&self.model_id);
        let mut text = String::new();
        self.engine
            .lock()
            .expect("llama engine")
            .generate_with_tools(
                &messages,
                &tools,
                append_thinking_off_suffix,
                cancel,
                &mut |piece, _| {
                    if cancel.is_stale(generation) {
                        return Err(Error::Cancelled);
                    }
                    text.push_str(piece);
                    Ok(())
                },
            )?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        on_token(TokenChunk {
            turn: user.turn,
            generation,
            index: 0,
            text: text.clone(),
            is_last: true,
        })?;
        let parsed = match parse_qwen_tool_calls(&text) {
            Ok(parsed) => parsed,
            Err(message) => {
                self.note_tool_event("rejected", "", "", "", &message);
                return Ok(Vec::new());
            }
        };
        let mut calls = Vec::with_capacity(parsed.len());
        for (index, parsed) in parsed.into_iter().enumerate() {
            match normalize_local_tool_call(index, parsed) {
                Ok(call) => {
                    self.note_tool_event(
                        "call",
                        &call.name,
                        &call.id,
                        &call.arguments.to_string(),
                        "",
                    );
                    calls.push(call);
                }
                Err(message) => {
                    self.note_tool_event("rejected", "", "", "", &message);
                    return Ok(Vec::new());
                }
            }
        }
        Ok(calls)
    }

    /// One tools-aware turn against the local LFM template (Phase 4).
    ///
    /// Adapter half of the shared tool-call contract for the second dialect:
    /// renders the `List of tools:` preamble through the shim's LFM-aware
    /// entry, collects the turn text, then parses and normalizes LFM
    /// `<|tool_call_start|>[name(k="v")]<|tool_call_end|>` blocks into
    /// [`ToolCall`]s with `call` / `rejected` events for
    /// [`Llm::take_tool_events`]. Executing the calls stays with the Phase-4
    /// loop (host-owned executor); this never runs a tool. Not on the
    /// default `run` path: [`Llm::generate`] stays tool-free and
    /// byte-identical.
    ///
    /// Bound (issue-89 lesson): collection stops at
    /// [`LFM_TOOL_TURN_MAX_CHARS`]; an over-budget turn records a
    /// `rejected` event and returns no calls (spoken fallback) instead of
    /// generating until cancel. A real shutdown still surfaces as
    /// [`Error::Cancelled`].
    pub fn generate_lfm_tool_turn(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<Vec<ToolCall>> {
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
        let messages = self.tool_messages(history, user, &[]);
        let tools = local_tool_definitions_json();
        let mut text = String::new();
        let mut over_budget = false;
        let outcome = self
            .engine
            .lock()
            .expect("llama engine")
            .generate_with_lfm_tools(&messages, &tools, cancel, &mut |piece, _| {
                if cancel.is_stale(generation) {
                    return Err(Error::Cancelled);
                }
                text.push_str(piece);
                if text.len() > LFM_TOOL_TURN_MAX_CHARS {
                    over_budget = true;
                    return Err(Error::Cancelled);
                }
                Ok(())
            });
        match outcome {
            Err(Error::Cancelled) if over_budget && !cancel.is_shutdown() => {
                self.note_tool_event(
                    "rejected",
                    "",
                    "",
                    "",
                    "lfm tool turn exceeded the thinking bound",
                );
                return Ok(Vec::new());
            }
            Err(err) => return Err(err),
            Ok(()) => {}
        }
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        on_token(TokenChunk {
            turn: user.turn,
            generation,
            index: 0,
            text: text.clone(),
            is_last: true,
        })?;
        let parsed = match parse_lfm_tool_calls(&text) {
            Ok(parsed) => parsed,
            Err(message) => {
                self.note_tool_event("rejected", "", "", "", &message);
                return Ok(Vec::new());
            }
        };
        let mut calls = Vec::with_capacity(parsed.len());
        for (index, parsed) in parsed.into_iter().enumerate() {
            match normalize_local_tool_call(index, parsed) {
                Ok(call) => {
                    self.note_tool_event(
                        "call",
                        &call.name,
                        &call.id,
                        &call.arguments.to_string(),
                        "",
                    );
                    calls.push(call);
                }
                Err(message) => {
                    self.note_tool_event("rejected", "", "", "", &message);
                    return Ok(Vec::new());
                }
            }
        }
        Ok(calls)
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

    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        std::mem::take(&mut *self.tool_events.lock().expect("llama tool-events mutex"))
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

        let messages = self.messages(history, user);

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
    fn prompt_token_count(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
    ) -> Result<usize>;

    fn generate(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()>;

    /// Tools-aware generation (issue 87). The default runs the plain path so
    /// scripted/test engines stay tool-free; the native engine renders the
    /// Qwen `<tools>` preamble through the shim alongside the plain path.
    fn generate_with_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        let _ = tools_json;
        self.generate(messages, append_thinking_off_suffix, cancel, on_piece)
    }

    /// LFM tools-aware generation (Phase 4). The default runs the plain
    /// path so scripted/test engines stay tool-free; the native engine
    /// renders the `List of tools:` preamble through the LFM shim entry.
    /// Kept separate from [`Engine::generate_with_tools`] (second-dialect
    /// exception).
    fn generate_with_lfm_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        let _ = tools_json;
        // LFM has no think-off suffix convention; the plain path default
        // (`false`) keeps tool-free framing unchanged.
        self.generate(messages, false, cancel, on_piece)
    }

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
    fn prompt_token_count(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
    ) -> Result<usize> {
        self.ctx
            .prompt_token_count(messages, append_thinking_off_suffix)
            .map_err(|err| Error::Provider {
                provider: crate::defaults::BuiltinDefaults::v0().llm.as_str(),
                message: format!("{err:?}"),
            })
    }

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
        let outcome = unsafe {
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
                        Ok(()) => {
                            // Re-arm after the downstream consumer returns:
                            // a slow token queue / TTS worker is backpressure,
                            // not a model stall.
                            abort.note_token();
                            Ok(())
                        }
                        Err(Error::Cancelled) => Err(LlamaError::Cancelled),
                        Err(err) => Err(LlamaError::Failed(err.to_string())),
                    }
                },
            )
        };
        finish_native(outcome, &abort, cancel, on_piece)
    }

    fn generate_with_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort = GenerationAbort::new(cancel);
        let abort_user = (&abort as *const GenerationAbort).cast_mut().cast();
        let c_tools = tools_json.to_string();
        let outcome = unsafe {
            self.ctx.generate_with_tools(
                messages,
                &c_tools,
                LlamaGenerate {
                    append_thinking_off_suffix,
                    n_threads: thread_count(),
                },
                Some(abort_on_stall),
                abort_user,
                &mut |text, is_last| {
                    abort.note_token();
                    match on_piece(text, is_last) {
                        Ok(()) => {
                            // Re-arm after the downstream consumer returns:
                            // a slow token queue / TTS worker is backpressure,
                            // not a model stall.
                            abort.note_token();
                            Ok(())
                        }
                        Err(Error::Cancelled) => Err(LlamaError::Cancelled),
                        Err(err) => Err(LlamaError::Failed(err.to_string())),
                    }
                },
            )
        };
        finish_native(outcome, &abort, cancel, on_piece)
    }

    fn generate_with_lfm_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let abort = GenerationAbort::new(cancel);
        let abort_user = (&abort as *const GenerationAbort).cast_mut().cast();
        let c_tools = tools_json.to_string();
        let outcome = unsafe {
            self.ctx.generate_with_lfm_tools(
                messages,
                &c_tools,
                LlamaGenerate {
                    append_thinking_off_suffix: false,
                    n_threads: thread_count(),
                },
                Some(abort_on_stall),
                abort_user,
                &mut |text, is_last| {
                    abort.note_token();
                    match on_piece(text, is_last) {
                        Ok(()) => {
                            // Re-arm after the downstream consumer returns:
                            // a slow token queue / TTS worker is backpressure,
                            // not a model stall.
                            abort.note_token();
                            Ok(())
                        }
                        Err(Error::Cancelled) => Err(LlamaError::Cancelled),
                        Err(err) => Err(LlamaError::Failed(err.to_string())),
                    }
                },
            )
        };
        finish_native(outcome, &abort, cancel, on_piece)
    }

    fn context_window(&self) -> Option<(i32, i32)> {
        Some((self.ctx.n_ctx(), self.ctx.n_ctx_train()))
    }
}

fn finish_native(
    outcome: std::result::Result<(), LlamaError>,
    abort: &GenerationAbort<'_>,
    cancel: &Cancel,
    on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
) -> Result<()> {
    match outcome {
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
    matches!(model_id, QWEN35_2B_ASSET)
}

/// Tool schemas for the local Qwen tools-aware prompt (issue 87). The same
/// two host-owned primitives the online adapter sends (`web_fetch`, `shell`);
/// serialized as the `<tools>` JSON array the Qwen template consumes. One
/// shared contract, translated at each adapter's edge.
pub fn local_tool_definitions_json() -> String {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch and read the content of one public HTTP or HTTPS URL.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The absolute public HTTP or HTTPS URL to fetch."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a read-only command via direct argv in the workspace.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["argv"],
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command and argument array."
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional relative path within the workspace."
                        }
                    }
                }
            }
        }
    ])
    .to_string()
}

/// One parsed Qwen `<tool_call>` block: function name plus its parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLocalToolCall {
    /// Function name as written in `<function=name>`.
    pub name: String,
    /// Parameter map built from the `<parameter>` blocks.
    pub arguments: serde_json::Value,
}

/// Parse Qwen-native tool calls from generated text (issue 87).
///
/// Each `<tool_call><function=name><parameter=k>v</parameter>…</function></tool_call>`
/// block yields one call, in order. Prose around the blocks (optional
/// reasoning before the call) is ignored. Malformed blocks fail closed with
/// a message; the caller records a `rejected` tool event and speaks the
/// fallback instead of executing anything.
pub fn parse_qwen_tool_calls(text: &str) -> std::result::Result<Vec<ParsedLocalToolCall>, String> {
    let mut calls = Vec::new();
    let mut rest = text;
    let mut searched = 0;
    while let Some(open) = rest.find("<tool_call>") {
        searched += open;
        let after_open = &rest[open + "<tool_call>".len()..];
        let close = after_open
            .find("</tool_call>")
            .ok_or_else(|| "qwen tool call is missing </tool_call>".to_string())?;
        // Reject nested blocks rather than guessing a pairing.
        if after_open[..close].contains("<tool_call>") {
            return Err("qwen tool call contains a nested <tool_call>".to_string());
        }
        calls.push(parse_qwen_tool_call_block(&after_open[..close])?);
        rest = &after_open[close + "</tool_call>".len()..];
        searched += "<tool_call>".len() + close + "</tool_call>".len();
    }
    if calls.is_empty() {
        return Err("no qwen <tool_call> block found".to_string());
    }
    let _ = searched;
    Ok(calls)
}

fn parse_qwen_tool_call_block(block: &str) -> std::result::Result<ParsedLocalToolCall, String> {
    let fn_open = block
        .find("<function=")
        .ok_or_else(|| "qwen tool call is missing <function=name>".to_string())?;
    let name_start = fn_open + "<function=".len();
    let name_end = block[name_start..]
        .find('>')
        .ok_or_else(|| "qwen tool call has a malformed <function=>".to_string())?;
    let name = block[name_start..name_start + name_end].trim().to_string();
    if name.is_empty() {
        return Err("qwen tool call has an empty function name".to_string());
    }
    let fn_body = &block[name_start + name_end + 1..];
    let fn_close = fn_body
        .find("</function>")
        .ok_or_else(|| "qwen tool call is missing </function>".to_string())?;
    if fn_body[..fn_close].contains("<function=") {
        return Err("qwen tool call contains a nested <function=".to_string());
    }
    let params = &fn_body[..fn_close];
    let mut map = serde_json::Map::new();
    let mut rest = params;
    while let Some(p_open) = rest.find("<parameter=") {
        let key_start = p_open + "<parameter=".len();
        let key_end = rest[key_start..]
            .find('>')
            .ok_or_else(|| "qwen tool call has a malformed <parameter=>".to_string())?;
        let key = rest[key_start..key_start + key_end].trim().to_string();
        if key.is_empty() {
            return Err("qwen tool call has an empty parameter name".to_string());
        }
        let value_start = key_start + key_end + 1;
        let value_end = rest[value_start..]
            .find("</parameter>")
            .ok_or_else(|| format!("qwen tool call parameter {key:?} is missing </parameter>"))?;
        let value = rest[value_start..value_start + value_end]
            .trim()
            .to_string();
        if map.contains_key(&key) {
            return Err(format!("qwen tool call repeats parameter {key:?}"));
        }
        map.insert(key, serde_json::Value::String(value));
        rest = &rest[value_start + value_end + "</parameter>".len()..];
    }
    Ok(ParsedLocalToolCall {
        name,
        arguments: serde_json::Value::Object(map),
    })
}

/// Normalize one parsed local call into the shared [`ToolCall`] contract.
/// Local generations carry no provider ids, so the id is synthesized
/// (`local-call-{index}`); the matching [`ToolResult::tool_call_id`] uses the
/// same value, and the pipeline correlates them exactly like online ids.
/// Unknown names and non-object arguments fail closed.
pub fn normalize_local_tool_call(
    index: usize,
    parsed: ParsedLocalToolCall,
) -> std::result::Result<ToolCall, String> {
    if !matches!(parsed.name.as_str(), "web_fetch" | "shell") {
        return Err("tool call has an unknown name".to_string());
    }
    if !parsed.arguments.is_object() {
        return Err("tool call arguments must be a JSON object".to_string());
    }
    Ok(ToolCall {
        id: format!("local-call-{index}"),
        name: parsed.name,
        arguments: parsed.arguments,
    })
}

/// Parse LFM-native tool calls from generated text (Phase 4).
///
/// Each `<|tool_call_start|>[name(k="v", …)]<|tool_call_end|>` block yields
/// one or more calls, in order; multiple blocks are concatenated in document
/// order. Prose around the blocks (reasoning, `always-thinks` chatter) is
/// ignored. Bracket lists may hold several comma-separated calls.
/// Malformed blocks fail closed with a message; the caller records a
/// `rejected` tool event and speaks the fallback instead of executing
/// anything. Separate from [`parse_qwen_tool_calls`] (second-dialect
/// exception); values are stored as strings like the Qwen path (lists are
/// space-joined) so both dialects feed the same shared contract.
pub fn parse_lfm_tool_calls(text: &str) -> std::result::Result<Vec<ParsedLocalToolCall>, String> {
    const OPEN: &str = "<|tool_call_start|>";
    const CLOSE: &str = "<|tool_call_end|>";
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(OPEN) {
        let after_open = &rest[open + OPEN.len()..];
        let close = after_open
            .find(CLOSE)
            .ok_or_else(|| "lfm tool call is missing <|tool_call_end|>".to_string())?;
        let block = &after_open[..close];
        if block.contains(OPEN) {
            return Err("lfm tool call contains a nested <|tool_call_start|>".to_string());
        }
        calls.extend(parse_lfm_tool_call_block(block)?);
        rest = &after_open[close + CLOSE.len()..];
    }
    if calls.is_empty() {
        return Err("no lfm <|tool_call_start|> block found".to_string());
    }
    Ok(calls)
}

fn parse_lfm_tool_call_block(block: &str) -> std::result::Result<Vec<ParsedLocalToolCall>, String> {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return Err("lfm tool call block is empty".to_string());
    }
    let inner = if trimmed.starts_with('[') {
        if !trimmed.ends_with(']') {
            return Err("lfm tool call block has unbalanced brackets".to_string());
        }
        trimmed[1..trimmed.len() - 1].trim()
    } else {
        trimmed
    };
    if inner.is_empty() {
        return Err("lfm tool call block is empty".to_string());
    }
    let mut calls = Vec::new();
    for source in split_top_level(inner, ',') {
        calls.push(parse_lfm_pythonic_call(&source)?);
    }
    Ok(calls)
}

fn parse_lfm_pythonic_call(source: &str) -> std::result::Result<ParsedLocalToolCall, String> {
    let source = source.trim();
    let paren = source
        .find('(')
        .ok_or_else(|| "lfm tool call is missing (args)".to_string())?;
    let name = source[..paren].trim();
    if name.is_empty() || !is_lfm_identifier(name) {
        return Err("lfm tool call has an invalid function name".to_string());
    }
    if !source.ends_with(')') {
        return Err("lfm tool call is missing the closing )".to_string());
    }
    let args = &source[paren + 1..source.len() - 1];
    let mut map = serde_json::Map::new();
    if !args.trim().is_empty() {
        for part in split_top_level(args, ',') {
            let (key, value) = parse_lfm_kwarg(&part)?;
            if map.contains_key(&key) {
                return Err(format!("lfm tool call repeats parameter {key:?}"));
            }
            map.insert(key, value);
        }
    }
    Ok(ParsedLocalToolCall {
        name: name.to_string(),
        arguments: serde_json::Value::Object(map),
    })
}

fn parse_lfm_kwarg(part: &str) -> std::result::Result<(String, serde_json::Value), String> {
    let part = part.trim();
    let eq = part
        .find('=')
        .ok_or_else(|| "lfm tool call parameter is missing =".to_string())?;
    // Reject `==` (comparison, not a kwarg) rather than splitting inside it.
    if part[eq + 1..].starts_with('=') {
        return Err("lfm tool call parameter is missing =".to_string());
    }
    let key = part[..eq].trim();
    if key.is_empty() || !is_lfm_identifier(key) {
        return Err("lfm tool call has an invalid parameter name".to_string());
    }
    let raw = part[eq + 1..].trim();
    if raw.is_empty() {
        return Err(format!("lfm tool call parameter {key:?} is empty"));
    }
    Ok((key.to_string(), parse_lfm_value(raw)?))
}

fn parse_lfm_value(raw: &str) -> std::result::Result<serde_json::Value, String> {
    let raw = raw.trim();
    if raw.starts_with('[') {
        if !raw.ends_with(']') {
            return Err("lfm tool call list value has unbalanced brackets".to_string());
        }
        let inner = raw[1..raw.len() - 1].trim();
        if inner.is_empty() {
            return Ok(serde_json::Value::String(String::new()));
        }
        let mut items = Vec::new();
        for item in split_top_level(inner, ',') {
            items.push(parse_lfm_scalar(&item)?);
        }
        // Qwen-path parity: every parameter is a string, so a Pythonic
        // list (e.g. `argv=["df", "-h", "."]`) joins into one string.
        return Ok(serde_json::Value::String(items.join(" ")));
    }
    parse_lfm_scalar(raw).map(serde_json::Value::String)
}

fn parse_lfm_scalar(raw: &str) -> std::result::Result<String, String> {
    let raw = raw.trim();
    if raw.len() >= 2
        && ((raw.starts_with('"') && raw.ends_with('"'))
            || (raw.starts_with('\'') && raw.ends_with('\'')))
    {
        return Ok(unescape_lfm_string(&raw[1..raw.len() - 1]));
    }
    if raw.contains(['"', '\'', '[', ']', '(', ')']) {
        return Err("lfm tool call value is malformed".to_string());
    }
    Ok(raw.to_string())
}

fn unescape_lfm_string(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some(other) => out.push(other),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn is_lfm_identifier(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Split on `sep` at the top level only: separators inside quotes, `()`,
/// or `[]` do not split.
fn split_top_level(source: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth_paren = 0usize;
    let mut depth_bracket = 0usize;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut start = 0usize;
    for (i, c) in source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if let Some(q) = quote {
            if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => quote = Some(c),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            _ => {
                if c == sep && depth_paren == 0 && depth_bracket == 0 {
                    parts.push(source[start..i].trim().to_string());
                    start = i + c.len_utf8();
                }
            }
        }
    }
    parts.push(source[start..].trim().to_string());
    parts
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
        fn prompt_token_count(
            &mut self,
            messages: &[ChatMessage],
            _append_thinking_off_suffix: bool,
        ) -> Result<usize> {
            Ok(messages
                .iter()
                .map(|message| message.content.split_whitespace().count())
                .sum())
        }

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
        assert_eq!(LFM25_26B_ASSET, BuiltinDefaults::v0().llm_model);
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
    fn from_cache_requires_default_lfm_asset() {
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
        assert!(err.to_string().contains("lfm2.5-2.6b"));
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
    fn only_the_2b_model_gets_the_thinking_off_prompt_suffix() {
        assert!(is_thinking_tag_supported_model(QWEN35_2B_ASSET));
        assert!(!is_thinking_tag_supported_model(QWEN35_08B_ASSET));
        assert!(!is_thinking_tag_supported_model(LLAMA_32_1B_ASSET));
    }

    #[test]
    fn local_tool_definitions_cover_the_shared_contract() {
        let value: serde_json::Value =
            serde_json::from_str(&local_tool_definitions_json()).expect("valid tools JSON");
        let names: Vec<&str> = value
            .as_array()
            .expect("tools array")
            .iter()
            .map(|tool| {
                tool.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .expect("function name")
            })
            .collect();
        assert_eq!(names, vec!["web_fetch", "shell"]);
    }

    #[test]
    fn qwen_tool_call_with_two_parameters_parses_in_order() {
        let text = "Let me check that.\n\n<tool_call>\n<function=shell>\n<parameter=argv>\ndf -h .\n</parameter>\n<parameter=cwd>\n.\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_qwen_tool_calls(text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("argv").and_then(|v| v.as_str()),
            Some("df -h .")
        );
        let normalized = normalize_local_tool_call(0, calls[0].clone()).expect("normalize");
        assert_eq!(normalized.id, "local-call-0");
        assert_eq!(normalized.name, "shell");
        assert!(normalized.arguments.is_object());
    }

    #[test]
    fn qwen_multiple_tool_calls_keep_document_order() {
        let text = "<tool_call>\n<function=web_fetch>\n<parameter=url>\nhttps://example.com/a\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=shell>\n<parameter=argv>\ndate\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_qwen_tool_calls(text).expect("parse");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_fetch");
        assert_eq!(calls[1].name, "shell");
        assert_eq!(
            normalize_local_tool_call(1, calls[1].clone())
                .expect("normalize")
                .id,
            "local-call-1"
        );
    }

    #[test]
    fn qwen_malformed_blocks_fail_closed() {
        assert!(parse_qwen_tool_calls("just prose, no call").is_err());
        assert!(parse_qwen_tool_calls("<tool_call>\n<function=shell>").is_err());
        assert!(parse_qwen_tool_calls(
            "<tool_call>\n<function=shell>\n<parameter=argv>\ndate\n</function>\n</tool_call>"
        )
        .is_err());
        assert!(parse_qwen_tool_calls(
            "<tool_call>\n<function=>\n<parameter=argv>\ndate\n</parameter>\n</function>\n</tool_call>"
        )
        .is_err());
        assert!(parse_qwen_tool_calls(
            "<tool_call>\n<tool_call>\n<function=shell>\n</function>\n</tool_call>\n</tool_call>"
        )
        .is_err());
    }

    #[test]
    fn local_unknown_tool_name_is_rejected_like_online() {
        let parsed = ParsedLocalToolCall {
            name: "curl".into(),
            arguments: serde_json::json!({"url": "https://example.com"}),
        };
        assert_eq!(
            normalize_local_tool_call(0, parsed).expect_err("unknown name"),
            "tool call has an unknown name"
        );
        let parsed = ParsedLocalToolCall {
            name: "shell".into(),
            arguments: serde_json::json!(["date"]),
        };
        assert_eq!(
            normalize_local_tool_call(0, parsed).expect_err("non-object"),
            "tool call arguments must be a JSON object"
        );
    }

    #[test]
    fn tool_results_become_tool_role_messages_in_order() {
        let llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let results = vec![
            ToolResult {
                tool_call_id: "local-call-0".into(),
                content: "first result".into(),
                ok: true,
            },
            ToolResult {
                tool_call_id: "local-call-1".into(),
                content: "second result".into(),
                ok: true,
            },
        ];
        let messages = llm.tool_messages(&[], &user(0, "hi"), &results);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].content, "first result");
        assert_eq!(messages[3].role, "tool");
        assert_eq!(messages[3].content, "second result");
    }

    #[test]
    fn local_tool_events_drain_like_the_online_adapter() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        assert!(Llm::take_tool_events(&mut llm).is_empty());
        llm.note_tool_event("call", "shell", "local-call-0", "{\"argv\":\"date\"}", "");
        llm.note_tool_event(
            "result",
            "shell",
            "local-call-0",
            "{\"argv\":\"date\"}",
            "ok",
        );
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, "call");
        assert_eq!(events[1].kind, "result");
        assert!(Llm::take_tool_events(&mut llm).is_empty());
    }

    #[test]
    fn tool_turn_parses_scripted_calls_and_records_events() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec![
                "<tool_call>\n<function=shell>\n<parameter=argv>\ndate\n</parameter>\n</function>\n</tool_call>"
                    .into(),
            ],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_tool_turn(
                &[],
                &user(0, "what time is it"),
                &Cancel::new(),
                &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                },
            )
            .expect("tool turn");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "local-call-0");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "call");
        assert_eq!(events[0].call_id, "local-call-0");
    }

    #[test]
    fn tool_turn_without_a_call_records_rejection() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["just prose, no call".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_tool_turn(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("tool turn");
        assert!(calls.is_empty());
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "rejected");
    }

    #[test]
    fn lfm_asset_is_manifest_pinned_and_selectable() {
        assert_eq!(LFM25_26B_ASSET, "lfm2.5-2.6b");
        assert!(is_v0_llm_model(LFM25_26B_ASSET));
        assert_eq!(LFM_TOOL_TURN_MAX_CHARS, 8_000);
    }

    #[test]
    fn lfm_pythonic_call_with_kwargs_parses_in_order() {
        let text = "Checking now.\n<|tool_call_start|>[shell(argv=\"df -h .\")]<|tool_call_end|>";
        let calls = parse_lfm_tool_calls(text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("argv").and_then(|v| v.as_str()),
            Some("df -h .")
        );
        let normalized = normalize_local_tool_call(0, calls[0].clone()).expect("normalize");
        assert_eq!(normalized.id, "local-call-0");
        assert_eq!(normalized.name, "shell");
        assert!(normalized.arguments.is_object());
    }

    #[test]
    fn lfm_list_values_join_like_the_qwen_path() {
        let text = "<|tool_call_start|>[shell(argv=[\"df\", \"-h\", \".\"])]<|tool_call_end|>";
        let calls = parse_lfm_tool_calls(text).expect("parse");
        assert_eq!(
            calls[0].arguments.get("argv").and_then(|v| v.as_str()),
            Some("df -h .")
        );
    }

    #[test]
    fn lfm_multiple_blocks_and_calls_keep_document_order() {
        let text = "<|tool_call_start|>[web_fetch(url=\"https://example.com/a\")]<|tool_call_end|> \
            <|tool_call_start|>[shell(argv=\"date\"), web_fetch(url='https://example.com/b')]<|tool_call_end|>";
        let calls = parse_lfm_tool_calls(text).expect("parse");
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "web_fetch");
        assert_eq!(calls[1].name, "shell");
        assert_eq!(calls[2].name, "web_fetch");
        assert_eq!(
            normalize_local_tool_call(2, calls[2].clone())
                .expect("normalize")
                .id,
            "local-call-2"
        );
    }

    #[test]
    fn lfm_malformed_blocks_fail_closed() {
        assert!(parse_lfm_tool_calls("just prose, no call").is_err());
        assert!(parse_lfm_tool_calls("<|tool_call_start|>[shell(argv=\"date\")]").is_err());
        assert!(parse_lfm_tool_calls(
            "<|tool_call_start|>shell(argv=\"date\")<|tool_call_start|>[shell(argv=\"date\")]<|tool_call_end|>"
        )
        .is_err());
        assert!(parse_lfm_tool_calls("<|tool_call_start|>[]<|tool_call_end|>").is_err());
        assert!(
            parse_lfm_tool_calls("<|tool_call_start|>[shell(argv=\"date\"]<|tool_call_end|>")
                .is_err()
        );
        assert!(parse_lfm_tool_calls(
            "<|tool_call_start|>[unknown_tool(x=\"1\")]<|tool_call_end|>"
        )
        .map(|calls| normalize_local_tool_call(0, calls[0].clone()))
        .expect("parses")
        .is_err());
    }

    #[test]
    fn lfm_tool_turn_parses_scripted_calls_and_records_events() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec![
                "Let me check.\n<|tool_call_start|>[shell(argv=\"date\")]<|tool_call_end|>".into(),
            ],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_lfm_tool_turn(
                &[],
                &user(0, "what time is it"),
                &Cancel::new(),
                &mut |chunk| {
                    chunks.push(chunk);
                    Ok(())
                },
            )
            .expect("lfm tool turn");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "local-call-0");
        assert_eq!(calls[0].name, "shell");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "call");
        assert_eq!(events[0].call_id, "local-call-0");
    }

    #[test]
    fn lfm_tool_turn_without_a_call_records_rejection() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["just prose, no call".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_lfm_tool_turn(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("lfm tool turn");
        assert!(calls.is_empty());
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "rejected");
    }

    #[test]
    fn lfm_tool_turn_over_budget_fails_closed_instead_of_hanging() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["x".repeat(LFM_TOOL_TURN_MAX_CHARS + 1)],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_lfm_tool_turn(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("over-budget turn ends in fallback");
        assert!(calls.is_empty());
        assert!(chunks.is_empty());
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "rejected");
        assert!(events[0].content.contains("thinking bound"));
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
