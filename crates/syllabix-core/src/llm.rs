//! In-process llama.cpp GGUF language model.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
#[cfg(all(not(coverage), feature = "ggml"))]
use std::time::Duration;

/// Chat turn passed to the llama.cpp engine. Kept local so scripted tests
/// compile without `syllabix-native` (no CMake).
#[derive(Clone, Debug, PartialEq, Eq)]
struct ChatMessage {
    role: String,
    content: String,
}

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::executor;
use crate::fake::LlmCall;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::policy::{DeveloperPermissions, NetworkMode};
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

/// Manifest id for the default LiquidAI LFM3-1.7B Q8_0 GGUF.
pub const LFM3_1_7B_ASSET: &str = "lfm3-1.7b";

/// Manifest id for the yaml-only LiquidAI LFM3-1.7B BF16 GGUF.
pub const LFM3_1_7B_BF16_ASSET: &str = "lfm3-1.7b-bf16";

/// Manifest id for the LiquidAI LFM2.5-2.6B QAD Q4_0 GGUF (yaml selectable).
/// Named `2_6B` (not `26B`) so the constant cannot be read as twenty-six billion.
pub const LFM25_2_6B_ASSET: &str = "lfm2.5-2.6b";

/// Manifest id for the yaml-only LiquidAI LFM2.5-350M QAD Q4_0 GGUF.
pub const LFM25_350M_ASSET: &str = "lfm2.5-350m";

/// Manifest id for the yaml-only LiquidAI LFM2.5-230M QAD Q4_0 GGUF.
pub const LFM25_230M_ASSET: &str = "lfm2.5-230m";

/// Upper bound on collected text for one local common-chat tool turn (issue-89
/// lesson). Models that think without emitting a complete tool call must fail
/// closed into the spoken fallback instead of generating until cancel. The
/// bound lives in the adapter, not in YAML; generation still stops on EOS,
/// cancellation, or context exhaustion.
pub const LOCAL_TOOL_TURN_MAX_CHARS: usize = 8_000;

/// Backward-compatible alias for integration tests and yaml comments.
pub const LFM_TOOL_TURN_MAX_CHARS: usize = LOCAL_TOOL_TURN_MAX_CHARS;

/// Spoken fallback for malformed local tool syntax. The host never executes
/// an unnormalized call and never forwards its dialect text to TTS. Eval
/// scoring treats this exactly like the other harness apologies: it is a
/// safe outcome, never a task answer.
pub const LOCAL_TOOL_FALLBACK_TEXT: &str = "Sorry, I could not complete that tool request safely.";

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

/// Append the labelled, read-only local skill catalog to an agent-harness
/// system prompt. An empty catalog preserves the existing prompt byte-for-byte.
pub fn render_system_prompt_with_skills(
    template: &str,
    language: &str,
    skill_context: &str,
) -> String {
    let prompt = render_system_prompt(template, language);
    if skill_context.trim().is_empty() {
        prompt
    } else {
        format!("{prompt}\n\n{skill_context}")
    }
}

/// Render the built-in system prompt for the turn's STT language.
pub fn system_prompt_for(language: &str) -> String {
    render_system_prompt(VOICE_SYSTEM_PROMPT_TEMPLATE, language)
}

/// Rolling history kept in the prompt.
pub const LLAMA_MAX_HISTORY_TURNS: usize = 8;

/// Native cancel must surface as [`Error::Cancelled`] within this window.
/// Full GGUF `n_ctx` makes one `llama_decode` heavier than the old 2048-slot cap.
pub const LLAMA_CANCEL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Every yaml-selectable local GGUF id, default first.
pub const V0_LLM_MODELS: [&str; 8] = [
    LFM3_1_7B_ASSET,
    LFM3_1_7B_BF16_ASSET,
    LFM25_2_6B_ASSET,
    LFM25_350M_ASSET,
    LFM25_230M_ASSET,
    LLAMA_32_1B_ASSET,
    QWEN35_08B_ASSET,
    QWEN35_2B_ASSET,
];

/// Return whether `id` names a supported local llama.cpp model.
pub fn is_v0_llm_model(id: &str) -> bool {
    V0_LLM_MODELS.contains(&id)
}

/// In-process llama.cpp adapter for supported GGUF models.
pub struct LlamaLlm {
    engine: Arc<Mutex<Box<dyn Engine>>>,
    calls: Arc<Mutex<Vec<LlmCall>>>,
    thinking: bool,
    model_id: String,
    system_prompt: String,
    skill_context: String,
    tool_events: Arc<Mutex<Vec<ToolTurnEvent>>>,
    agent_harness: bool,
    developer_permissions: DeveloperPermissions,
    workspace: PathBuf,
}

impl Clone for LlamaLlm {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            calls: Arc::clone(&self.calls),
            thinking: self.thinking,
            model_id: self.model_id.clone(),
            system_prompt: self.system_prompt.clone(),
            skill_context: self.skill_context.clone(),
            tool_events: Arc::clone(&self.tool_events),
            agent_harness: self.agent_harness,
            developer_permissions: self.developer_permissions.clone(),
            workspace: self.workspace.clone(),
        }
    }
}

impl LlamaLlm {
    /// Load a GGUF from disk.
    #[cfg(all(not(coverage), feature = "ggml"))]
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self> {
        let engine = native::LlamaEngine::load(path.as_ref())?;
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
            skill_context: String::new(),
            tool_events: Arc::new(Mutex::new(Vec::new())),
            agent_harness: false,
            developer_permissions: DeveloperPermissions::default_session(),
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        })
    }

    /// Coverage never loads native GGUF weights; without `ggml` there is no
    /// ggml frontend. Keep the constructor as a callable seam for tests.
    #[cfg(any(coverage, not(feature = "ggml")))]
    pub fn from_model_path(path: impl AsRef<Path>) -> Result<Self> {
        let _ = path;
        Err(Error::Provider {
            provider: crate::defaults::LlmProvider::Local.as_str(),
            message: if cfg!(coverage) {
                "llama.cpp inference is not loaded in coverage tests".into()
            } else {
                "llama.cpp inference requires the `ggml` feature".into()
            },
        })
    }

    /// Resolve the default `lfm3-1.7b` GGUF from the manifest cache, then load it.
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

    /// Resolve and load one GGUF id without fetching unselected models.
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
                    "unsupported value {model_id:?} (allowed: lfm3-1.7b, lfm3-1.7b-bf16, lfm2.5-2.6b, lfm2.5-350m, lfm2.5-230m, llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b)"
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

    /// YAML `pipeline.llm.system_prompt`, or the built-in template.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the host-discovered skill catalog for the opt-in harness only.
    pub fn with_skill_context(mut self, skill_context: impl Into<String>) -> Self {
        self.skill_context = skill_context.into();
        self
    }

    /// Enable the bounded local agent harness (sandboxed shell tools).
    pub fn with_agent_harness(mut self, enabled: bool) -> Self {
        self.agent_harness = enabled;
        self
    }

    /// Set the immutable capability ceiling used by the local agent harness.
    pub fn with_developer_permissions(mut self, permissions: DeveloperPermissions) -> Self {
        self.developer_permissions = permissions;
        self
    }

    /// Set the host-owned workspace used by agent-harness shell calls.
    /// Admission tests use this to keep writes out of the source checkout.
    pub fn with_workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = workspace.into();
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
            content: render_system_prompt_with_skills(
                &self.system_prompt,
                &user.language,
                &self.skill_context,
            ),
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
            skill_context: String::new(),
            tool_events: Arc::new(Mutex::new(Vec::new())),
            agent_harness: false,
            developer_permissions: DeveloperPermissions::default_session(),
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
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

    /// OpenAI-shaped message JSON for the local common-chat harness: rolling
    /// history plus assistant `tool_calls` and matching `tool` results.
    fn local_harness_messages_json(
        &self,
        history: &[HistoryTurn],
        user: &Transcript,
        exchanges: &[(Vec<ToolCall>, Vec<ToolResult>)],
    ) -> String {
        let mut messages: Vec<serde_json::Value> = self
            .messages(history, user)
            .iter()
            .map(|message| {
                serde_json::json!({
                    "role": message.role,
                    "content": message.content,
                })
            })
            .collect();
        for (calls, results) in exchanges {
            let wire_calls: Vec<serde_json::Value> = calls
                .iter()
                .map(|call| {
                    serde_json::json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments.to_string(),
                        }
                    })
                })
                .collect();
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": wire_calls,
            }));
            for result in results {
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }));
            }
        }
        serde_json::to_string(&messages).expect("harness messages JSON")
    }

    fn collect_common_chat_turn(
        &mut self,
        messages_json: &str,
        cancel: &Cancel,
        generation: crate::types::GenerationId,
    ) -> Result<(String, ParsedChatOutput), CommonChatCollectError> {
        let tools = local_tool_definitions_json_for(self.developer_permissions.network);
        let append_thinking_off_suffix =
            !self.thinking && is_thinking_tag_supported_model(&self.model_id);
        let mut text = String::new();
        let mut over_budget = false;
        let outcome = self
            .engine
            .lock()
            .expect("llama engine")
            .generate_with_common_chat(
                messages_json,
                &tools,
                self.thinking,
                append_thinking_off_suffix,
                cancel,
                &mut |piece, _| {
                    if cancel.is_stale(generation) {
                        return Err(Error::Cancelled);
                    }
                    text.push_str(piece);
                    if text.len() > LOCAL_TOOL_TURN_MAX_CHARS {
                        over_budget = true;
                        return Err(Error::Cancelled);
                    }
                    Ok(())
                },
            );
        match outcome {
            Err(Error::Cancelled) if over_budget && !cancel.is_shutdown() => {
                return Err(CommonChatCollectError::OverBudget);
            }
            Err(err) => return Err(CommonChatCollectError::Failed(err)),
            Ok(()) => {}
        }
        if cancel.is_shutdown() || cancel.is_stale(generation) {
            return Err(CommonChatCollectError::Failed(Error::Cancelled));
        }
        let output = match self
            .engine
            .lock()
            .expect("llama engine")
            .chat_parse_output(&text, false)
        {
            Ok(output) => output,
            Err(message) => {
                let recovered = recover_local_tool_calls(&text);
                if recovered.is_empty() {
                    return Err(CommonChatCollectError::Parse(format!(
                        "{message}; raw={}",
                        raw_snippet(&text)
                    )));
                }
                ParsedChatOutput {
                    content: String::new(),
                    tool_calls: recovered,
                }
            }
        };
        let output = if output.tool_calls.is_empty() {
            let recovered = recover_local_tool_calls(&text);
            if recovered.is_empty() {
                output
            } else {
                ParsedChatOutput {
                    content: String::new(),
                    tool_calls: recovered,
                }
            }
        } else {
            output
        };
        Ok((text, output))
    }

    fn normalize_chat_tool_calls(
        &self,
        output: ParsedChatOutput,
        call_index: usize,
    ) -> std::result::Result<Vec<ToolCall>, String> {
        if output.tool_calls.is_empty() {
            return Err("no tool calls in model output".to_string());
        }
        let parsed = parsed_chat_tool_calls(output)?;
        let mut calls = Vec::with_capacity(parsed.len());
        for (offset, parsed) in parsed.into_iter().enumerate() {
            calls.push(normalize_local_tool_call(call_index + offset, parsed)?);
        }
        Ok(calls)
    }

    fn generate_local_common_tool_turn(
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
        let messages_json = simple_messages_to_openai_json(&self.tool_messages(history, user, &[]));
        let (text, output) = match self.collect_common_chat_turn(&messages_json, cancel, generation)
        {
            Ok(value) => value,
            Err(CommonChatCollectError::OverBudget) => {
                self.note_tool_event(
                    "rejected",
                    "",
                    "",
                    "",
                    "local tool turn exceeded the thinking bound",
                );
                return Ok(Vec::new());
            }
            Err(CommonChatCollectError::Failed(err)) => return Err(err),
            Err(CommonChatCollectError::Parse(message)) => {
                self.note_tool_event("rejected", "", "", "", &message);
                return Ok(Vec::new());
            }
        };
        on_token(TokenChunk {
            turn: user.turn,
            generation,
            index: 0,
            text: text.clone(),
            is_last: true,
        })?;
        let calls = match self.normalize_chat_tool_calls(output, 0) {
            Ok(calls) => calls,
            Err(message) => {
                self.note_tool_event("rejected", "", "", "", &message);
                return Ok(Vec::new());
            }
        };
        for call in &calls {
            self.note_tool_event(
                "call",
                &call.name,
                &call.id,
                &call.arguments.to_string(),
                "",
            );
        }
        Ok(calls)
    }

    /// Run one tools-aware turn using llama.cpp common chat.
    pub fn generate_tool_turn(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<Vec<ToolCall>> {
        self.generate_local_common_tool_turn(history, user, cancel, on_token)
    }

    /// Run one tools-aware turn using llama.cpp common chat.
    pub fn generate_lfm_tool_turn(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<Vec<ToolCall>> {
        self.generate_local_common_tool_turn(history, user, cancel, on_token)
    }

    /// Complete the local Phase-5 harness loop via llama.cpp common chat.
    /// Tool dialect text stays inside the adapter; only the answer after the
    /// final re-prompt reaches the pipeline token stream and therefore TTS.
    fn generate_local_harness(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        let generation = cancel.generation();
        self.calls.lock().expect("llm call log").push(LlmCall {
            history_len: history.len(),
            history_user_texts: history.iter().map(|turn| turn.user.text.clone()).collect(),
            user_text: user.text.clone(),
        });
        let mut exchanges: Vec<(Vec<ToolCall>, Vec<ToolResult>)> = Vec::new();
        let mut call_count = 0usize;
        let mut prior_outputs: Vec<(String, serde_json::Value, String)> = Vec::new();
        loop {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            let messages_json = self.local_harness_messages_json(history, user, &exchanges);
            let (_text, output) =
                match self.collect_common_chat_turn(&messages_json, cancel, generation) {
                    Ok(value) => value,
                    Err(CommonChatCollectError::OverBudget) => {
                        self.note_tool_event(
                            "rejected",
                            "",
                            "",
                            "",
                            "local tool turn exceeded the thinking bound",
                        );
                        return emit_local_harness_reply(
                            user.turn,
                            generation,
                            LOCAL_TOOL_FALLBACK_TEXT,
                            cancel,
                            on_token,
                        );
                    }
                    Err(CommonChatCollectError::Failed(err)) => return Err(err),
                    Err(CommonChatCollectError::Parse(message)) => {
                        eprintln!("local-harness-fallback reason=parse {message}");
                        self.note_tool_event("rejected", "", "", "", &message);
                        return emit_local_harness_reply(
                            user.turn,
                            generation,
                            LOCAL_TOOL_FALLBACK_TEXT,
                            cancel,
                            on_token,
                        );
                    }
                };
            if output.tool_calls.is_empty() {
                let spoken = sanitize_harness_spoken_content(&output.content, &_text);
                if crate::speech_text::speak_text_for_tts(&spoken)
                    .trim()
                    .is_empty()
                {
                    eprintln!(
                        "local-harness-fallback reason=empty-spoken raw={}",
                        raw_snippet(&_text)
                    );
                    self.note_tool_event(
                        "rejected",
                        "",
                        "",
                        "",
                        &format!("empty spoken content; raw={}", raw_snippet(&_text)),
                    );
                }
                return emit_local_harness_reply(user.turn, generation, &spoken, cancel, on_token);
            }
            let parsed = match parsed_or_recovered_tool_calls(output, &_text) {
                Ok(parsed) if !parsed.is_empty() => parsed,
                Ok(_) | Err(_) => {
                    self.note_tool_event(
                        "rejected",
                        "",
                        "",
                        "",
                        &format!("no tool calls in model output; raw={}", raw_snippet(&_text)),
                    );
                    return emit_local_harness_reply(
                        user.turn,
                        generation,
                        LOCAL_TOOL_FALLBACK_TEXT,
                        cancel,
                        on_token,
                    );
                }
            };
            let mut executed = Vec::with_capacity(parsed.len());
            let mut results = Vec::with_capacity(parsed.len());
            for parsed in parsed {
                let index = call_count + executed.len();
                let call = match normalize_local_tool_call(index, parsed.clone()) {
                    Ok(call) => call,
                    Err(message) => {
                        let call = ToolCall {
                            id: format!("local-call-{index}"),
                            name: parsed.name.clone(),
                            arguments: parsed.arguments.clone(),
                        };
                        let result = crate::types::ToolResult {
                            tool_call_id: call.id.clone(),
                            ok: false,
                            content: format!(
                                "{message}. Available tools: {}. Do not retry this name; explain the outcome.",
                                available_tool_names(self.developer_permissions.network)
                            ),
                        };
                        eprintln!(
                            "local-harness-tool-error name={} message={message} raw={}",
                            parsed.name,
                            raw_snippet(&_text)
                        );
                        self.note_tool_event(
                            "call",
                            &call.name,
                            &call.id,
                            &call.arguments.to_string(),
                            "",
                        );
                        self.note_tool_event(
                            "result",
                            &call.name,
                            &call.id,
                            &call.arguments.to_string(),
                            &result.content,
                        );
                        eprintln!(
                            "local-harness-step id={} name={} ok={} args={} result={}",
                            call.id,
                            call.name,
                            result.ok,
                            raw_snippet(&call.arguments.to_string()),
                            raw_snippet(&result.content),
                        );
                        executed.push(call);
                        results.push(result);
                        continue;
                    }
                };
                self.note_tool_event(
                    "call",
                    &call.name,
                    &call.id,
                    &call.arguments.to_string(),
                    "",
                );
                let outcome = executor::execute_with_permissions(
                    &call,
                    &self.workspace,
                    &self.developer_permissions,
                    cancel,
                );
                if cancel.is_shutdown() || cancel.is_stale(generation) {
                    return Err(Error::Cancelled);
                }
                let original = outcome.content.clone();
                let result = note_duplicate_same_output(
                    &call.name,
                    &call.arguments,
                    outcome,
                    &prior_outputs,
                );
                prior_outputs.push((call.name.clone(), call.arguments.clone(), original));
                self.note_tool_event(
                    "result",
                    &call.name,
                    &call.id,
                    &call.arguments.to_string(),
                    &result.content,
                );
                eprintln!(
                    "local-harness-step id={} name={} ok={} args={} result={}",
                    call.id,
                    call.name,
                    result.ok,
                    raw_snippet(&call.arguments.to_string()),
                    raw_snippet(&result.content),
                );
                executed.push(call);
                results.push(result);
            }
            call_count += executed.len();
            exchanges.push((executed, results));
        }
    }
}

enum CommonChatCollectError {
    OverBudget,
    Parse(String),
    Failed(Error),
}

/// Drop DeepSeek-style reasoning blocks from text we send to TTS. When
/// generation stops before `</think>`, the parser can leave
/// chain-of-thought in `content`; never speak that.
fn strip_unfinished_reasoning_from_spoken_content(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = text.to_string();
    loop {
        let Some(start) = out.find(OPEN) else {
            break;
        };
        if let Some(rel) = out[start..].find(CLOSE) {
            let end = start + rel + CLOSE.len();
            out.replace_range(start..end, "");
        } else {
            out.truncate(start);
            break;
        }
    }
    out.trim().to_string()
}

fn sanitize_harness_spoken_content(parsed_content: &str, raw_generation: &str) -> String {
    let mut spoken = strip_unfinished_reasoning_from_spoken_content(parsed_content);
    if raw_generation.contains("<think>") && !raw_generation.contains("</think>") {
        spoken.clear();
    }
    spoken
}

fn emit_local_harness_reply(
    turn: crate::types::TurnId,
    generation: crate::types::GenerationId,
    text: &str,
    cancel: &Cancel,
    on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
) -> Result<()> {
    if cancel.is_shutdown() || cancel.is_stale(generation) {
        return Err(Error::Cancelled);
    }
    // A tool result that yields no speakable reply is a silent turn: the
    // microphone reopens with nothing said. Fail closed with the spoken
    // fallback instead of emitting an empty reply TTS cannot voice.
    let scrubbed = strip_unfinished_reasoning_from_spoken_content(text);
    let reply = if crate::speech_text::speak_text_for_tts(&scrubbed)
        .trim()
        .is_empty()
    {
        LOCAL_TOOL_FALLBACK_TEXT.to_string()
    } else {
        scrubbed
    };
    on_token(TokenChunk {
        turn,
        generation,
        index: 0,
        text: reply,
        is_last: true,
    })
}

impl Llm for LlamaLlm {
    fn name(&self) -> &'static str {
        crate::defaults::LlmProvider::Local.as_str()
    }

    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        Some(LlmDebugMeta {
            provider: crate::defaults::LlmProvider::Local.as_str().into(),
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
        if self.agent_harness
            && self
                .engine
                .lock()
                .expect("llama engine")
                .chat_tools_supported()?
        {
            return self.generate_local_harness(history, user, cancel, on_token);
        }
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

    fn chat_tools_supported(&mut self) -> Result<bool> {
        Ok(true)
    }

    fn generate_with_common_chat(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        enable_thinking: bool,
        append_thinking_off_suffix: bool,
        cancel: &Cancel,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
    ) -> Result<()> {
        let _ = (tools_json, enable_thinking);
        let messages = openai_messages_json_to_chat(messages_json)?;
        self.generate(&messages, append_thinking_off_suffix, cancel, on_piece)
    }

    fn chat_parse_output(
        &mut self,
        text: &str,
        is_partial: bool,
    ) -> std::result::Result<ParsedChatOutput, String> {
        let _ = is_partial;
        Ok(ParsedChatOutput {
            content: text.to_string(),
            tool_calls: Vec::new(),
        })
    }

    fn context_window(&self) -> Option<(i32, i32)> {
        None
    }
}

#[cfg(all(not(coverage), feature = "ggml"))]
mod native {
    use super::*;
    use std::os::raw::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    use syllabix_native::{
        LlamaContext, LlamaError, LlamaGenerate, ParsedChatOutput as NativeParsedChatOutput,
    };

    fn ffi_messages(messages: &[ChatMessage]) -> Vec<syllabix_native::ChatMessage> {
        messages
            .iter()
            .map(|message| syllabix_native::ChatMessage {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect()
    }

    /// End a stalled local generation promptly, rather than leaving the TUI
    /// on an unfinished turn. This belongs with the native callback that
    /// consumes it; coverage never compiles that FFI boundary.
    pub(super) const LLAMA_TOKEN_STALL_TIMEOUT: Duration = Duration::from_secs(10);

    /// Prompt prefill emits no tokens, and llama.cpp polls the abort callback
    /// inside each prefill `llama_decode`. A tools-aware prompt (~500 tokens
    /// for the three host tools, more after tool results) can take well over
    /// [`LLAMA_TOKEN_STALL_TIMEOUT`] on CPU for a 4B-class GGUF, so the bound
    /// before the first streamed token is separate and generous.
    pub(super) const LLAMA_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(300);

    pub(super) struct LlamaEngine {
        ctx: LlamaContext,
    }

    impl LlamaEngine {
        pub(super) fn load(path: &Path) -> Result<Self> {
            let ctx =
                LlamaContext::load(path, thread_count()).map_err(|message| Error::Provider {
                    provider: crate::defaults::LlmProvider::Local.as_str(),
                    message,
                })?;
            Ok(Self { ctx })
        }

        fn parse_chat_output(
            &mut self,
            text: &str,
            is_partial: bool,
        ) -> std::result::Result<ParsedChatOutput, String> {
            self.ctx
                .chat_parse_output(text, is_partial)
                .map(map_native_parsed_output)
                .map_err(|err| format!("{err:?}"))
        }
    }

    fn map_native_parsed_output(output: NativeParsedChatOutput) -> ParsedChatOutput {
        ParsedChatOutput {
            content: output.content,
            tool_calls: output
                .tool_calls
                .into_iter()
                .map(|call| ParsedChatToolCall {
                    name: call.name,
                    arguments: call.arguments,
                    id: call.id,
                })
                .collect(),
        }
    }

    impl Engine for LlamaEngine {
        fn prompt_token_count(
            &mut self,
            messages: &[ChatMessage],
            append_thinking_off_suffix: bool,
        ) -> Result<usize> {
            self.ctx
                .prompt_token_count(&ffi_messages(messages), append_thinking_off_suffix)
                .map_err(|err| Error::Provider {
                    provider: crate::defaults::LlmProvider::Local.as_str(),
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
            let messages = ffi_messages(messages);
            let outcome = unsafe {
                self.ctx.generate(
                    &messages,
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

        fn chat_tools_supported(&mut self) -> Result<bool> {
            self.ctx
                .chat_tools_supported()
                .map_err(|err| Error::Provider {
                    provider: crate::defaults::LlmProvider::Local.as_str(),
                    message: format!("{err:?}"),
                })
        }

        fn generate_with_common_chat(
            &mut self,
            messages_json: &str,
            tools_json: &str,
            enable_thinking: bool,
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
                self.ctx.chat_generate(
                    messages_json,
                    &c_tools,
                    enable_thinking,
                    append_thinking_off_suffix,
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

        fn chat_parse_output(
            &mut self,
            text: &str,
            is_partial: bool,
        ) -> std::result::Result<ParsedChatOutput, String> {
            self.parse_chat_output(text, is_partial)
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
                provider: crate::defaults::LlmProvider::Local.as_str(),
                message,
            }),
        }
    }

    pub(super) struct GenerationAbort<'a> {
        pub(super) cancel: &'a Cancel,
        pub(super) last_token: Mutex<Instant>,
        pub(super) timed_out: AtomicBool,
        pub(super) first_token_seen: AtomicBool,
    }

    impl<'a> GenerationAbort<'a> {
        pub(super) fn new(cancel: &'a Cancel) -> Self {
            Self {
                cancel,
                last_token: Mutex::new(Instant::now()),
                timed_out: AtomicBool::new(false),
                first_token_seen: AtomicBool::new(false),
            }
        }

        pub(super) fn note_token(&self) {
            *self.last_token.lock().expect("llama token timer") = Instant::now();
            self.first_token_seen.store(true, Ordering::SeqCst);
        }

        /// Inter-token bound once streaming started; prefill bound before.
        pub(super) fn stall_timeout(&self) -> Duration {
            if self.first_token_seen.load(Ordering::SeqCst) {
                LLAMA_TOKEN_STALL_TIMEOUT
            } else {
                LLAMA_FIRST_TOKEN_TIMEOUT
            }
        }

        pub(super) fn timed_out(&self) -> bool {
            self.timed_out.load(Ordering::SeqCst)
        }
    }

    pub(super) unsafe extern "C" fn abort_on_stall(user_data: *mut c_void) -> bool {
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
            >= state.stall_timeout()
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

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Instant;

        #[test]
        fn stalled_abort_is_reported() {
            let cancel = Cancel::new();
            let state = GenerationAbort::new(&cancel);
            state.note_token();
            *state.last_token.lock().unwrap() = Instant::now() - LLAMA_TOKEN_STALL_TIMEOUT;
            assert!(unsafe {
                abort_on_stall((&state as *const GenerationAbort).cast_mut().cast())
            });
            assert!(state.timed_out());
        }

        #[test]
        fn prefill_gets_the_first_token_bound() {
            let cancel = Cancel::new();
            let state = GenerationAbort::new(&cancel);
            *state.last_token.lock().unwrap() = Instant::now() - LLAMA_TOKEN_STALL_TIMEOUT;
            assert!(!unsafe {
                abort_on_stall((&state as *const GenerationAbort).cast_mut().cast())
            });
            assert!(!state.timed_out());
            *state.last_token.lock().unwrap() = Instant::now() - LLAMA_FIRST_TOKEN_TIMEOUT;
            assert!(unsafe {
                abort_on_stall((&state as *const GenerationAbort).cast_mut().cast())
            });
            assert!(state.timed_out());
        }
    }
}

fn is_thinking_tag_supported_model(model_id: &str) -> bool {
    matches!(
        model_id,
        QWEN35_2B_ASSET | LFM3_1_7B_ASSET | LFM3_1_7B_BF16_ASSET
    )
}

/// Tool schemas for the local tools-aware prompt. The same
/// three host-owned primitives the online adapter sends (`web_fetch`,
/// `web_search`, `shell`); serialized as the `<tools>` JSON array the Qwen
/// template consumes. One shared representation, translated at each adapter
/// boundary. Network tools are omitted when the session network mode is none.
pub fn local_tool_definitions_json() -> String {
    local_tool_definitions_json_for(NetworkMode::Allow)
}

/// Tool schemas for the configured session network mode.
pub fn local_tool_definitions_json_for(network: NetworkMode) -> String {
    let shell = serde_json::json!({
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command in the workspace sandbox. Optional permission: read-only, workspace-write, or danger-full-access; the host never exceeds the session ceiling.",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Optional relative path inside the workspace."
                    },
                    "permission": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "description": "Optional filesystem permission request. Pass workspace-write to create or edit files; if the session is read-only the host denies it and you should explain that limit instead of retrying."
                    }
                }
            }
        }
    });
    let mut tools = Vec::new();
    if network == NetworkMode::Allow {
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch one public HTTP or HTTPS URL. Not for workspace files.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute public HTTP or HTTPS URL."
                        }
                    }
                }
            }
        }));
        tools.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web without an API key. Plain keywords only.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (1-256 chars)."
                        },
                        "count": {
                            "type": "number",
                            "description": "How many results to return (1-8, default 5)."
                        }
                    }
                }
            }
        }));
    }
    tools.push(shell);
    serde_json::Value::Array(tools).to_string()
}

/// One tool call from llama.cpp `common_chat_parse`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedChatToolCall {
    name: String,
    arguments: String,
    id: String,
}

/// Assistant turn from llama.cpp `common_chat_parse`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(crate) struct ParsedChatOutput {
    content: String,
    tool_calls: Vec<ParsedChatToolCall>,
}

/// One parsed Qwen `<tool_call>` block: function name plus its parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedLocalToolCall {
    /// Function name as written in `<function=name>`.
    pub name: String,
    /// Parameter map built from the `<parameter>` blocks.
    pub arguments: serde_json::Value,
}

/// Map a native common-chat parse into [`ParsedLocalToolCall`] for normalization.
pub(crate) fn parsed_chat_tool_calls(
    output: ParsedChatOutput,
) -> std::result::Result<Vec<ParsedLocalToolCall>, String> {
    let mut calls = Vec::with_capacity(output.tool_calls.len());
    for tool_call in output.tool_calls {
        let arguments: serde_json::Value = serde_json::from_str(&tool_call.arguments)
            .map_err(|_| "tool call arguments are not valid JSON".to_string())?;
        calls.push(ParsedLocalToolCall {
            name: tool_call.name,
            arguments,
        });
    }
    Ok(calls)
}

fn simple_messages_to_openai_json(messages: &[ChatMessage]) -> String {
    let value: Vec<serde_json::Value> = messages
        .iter()
        .map(|message| {
            serde_json::json!({
                "role": message.role,
                "content": message.content,
            })
        })
        .collect();
    serde_json::to_string(&value).expect("messages JSON")
}

fn openai_messages_json_to_chat(messages_json: &str) -> Result<Vec<ChatMessage>> {
    let value: Vec<serde_json::Value> =
        serde_json::from_str(messages_json).map_err(|err| Error::Provider {
            provider: crate::defaults::LlmProvider::Local.as_str(),
            message: format!("invalid harness messages JSON: {err}"),
        })?;
    value
        .into_iter()
        .map(|message| {
            let role = message
                .get("role")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let content = message
                .get("content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ChatMessage { role, content })
        })
        .collect()
}

pub fn normalize_local_tool_call(
    index: usize,
    mut parsed: ParsedLocalToolCall,
) -> std::result::Result<ToolCall, String> {
    if !matches!(parsed.name.as_str(), "web_fetch" | "web_search" | "shell") {
        return Err("tool call has an unknown name".to_string());
    }
    if !parsed.arguments.is_object() {
        return Err("tool call arguments must be a JSON object".to_string());
    }
    if parsed.name == "shell" {
        if let Some(argv) = parsed
            .arguments
            .get("argv")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
        {
            let command = parsed.arguments.as_object_mut().expect("object");
            command.insert("command".into(), serde_json::Value::String(argv));
            command.remove("argv");
        }
        if let Some(cwd) = parsed
            .arguments
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
        {
            let command = parsed.arguments.as_object_mut().expect("object");
            command.insert("workdir".into(), serde_json::Value::String(cwd));
            command.remove("cwd");
        }
    }
    if parsed.name == "web_search" {
        if let Some(object) = parsed.arguments.as_object_mut() {
            if let Some(count) = object.get("count").cloned() {
                if let Some(text) = count.as_str() {
                    if let Ok(number) = text.trim().parse::<u64>() {
                        object.insert(
                            "count".to_string(),
                            serde_json::Value::Number(number.into()),
                        );
                    }
                }
            }
        }
    }
    Ok(ToolCall {
        id: format!("local-call-{index}"),
        name: parsed.name,
        arguments: parsed.arguments,
    })
}

fn raw_snippet(text: &str) -> String {
    const MAX: usize = 1500;
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= MAX {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

/// Appended when this turn already ran the same name+arguments and got the
/// same real output. The call still executes; `ok` stays the real status.
pub(crate) const DUPLICATE_SAME_OUTPUT_HINT: &str =
    "You already ran this exact call and got the same result. Use it and answer the user.";

/// If `result` matches an earlier name+arguments+output in this turn, append
/// [`DUPLICATE_SAME_OUTPUT_HINT`]. Always keep the real `ok` and output.
pub(crate) fn note_duplicate_same_output(
    name: &str,
    arguments: &serde_json::Value,
    mut result: ToolResult,
    prior: &[(String, serde_json::Value, String)],
) -> ToolResult {
    let duplicate = prior
        .iter()
        .any(|(previous_name, previous_arguments, previous)| {
            previous_name == name && previous_arguments == arguments && previous == &result.content
        });
    if duplicate {
        if !result.content.is_empty() && !result.content.ends_with('\n') {
            result.content.push('\n');
        }
        result.content.push_str(DUPLICATE_SAME_OUTPUT_HINT);
    }
    result
}

fn available_tool_names(network: NetworkMode) -> &'static str {
    match network {
        NetworkMode::Allow => "web_fetch, web_search, shell",
        NetworkMode::None => "shell",
    }
}

fn parsed_or_recovered_tool_calls(
    output: ParsedChatOutput,
    raw: &str,
) -> std::result::Result<Vec<ParsedLocalToolCall>, String> {
    let mut calls = Vec::new();
    for tool_call in output.tool_calls {
        let arguments =
            serde_json::from_str(&tool_call.arguments).unwrap_or(serde_json::Value::Null);
        calls.push(ParsedLocalToolCall {
            name: tool_call.name,
            arguments,
        });
    }
    if calls.is_empty() {
        for tool_call in recover_local_tool_calls(raw) {
            let arguments = serde_json::from_str(&tool_call.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            calls.push(ParsedLocalToolCall {
                name: tool_call.name,
                arguments,
            });
        }
    }
    Ok(calls)
}

/// Recover LFM3/Qwen XML or LFM2 pythonic calls when common-chat parse fails
/// or swallows them as content. Unknown names stay on the calls so the
/// harness can return an error result instead of the spoken fallback.
pub(crate) fn recover_local_tool_calls(text: &str) -> Vec<ParsedChatToolCall> {
    let xml = recover_xml_tool_calls(text);
    if !xml.is_empty() {
        return xml;
    }
    recover_pythonic_tool_calls(text)
}

fn recover_xml_tool_calls(text: &str) -> Vec<ParsedChatToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<function=") {
        let after = &rest[start + "<function=".len()..];
        let Some(name_end) = after.find('>') else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        let body = &after[name_end + 1..];
        let close = body.find("</function>").unwrap_or(body.len());
        let inner = &body[..close];
        let mut object = serde_json::Map::new();
        let mut params = inner;
        while let Some(param_at) = params.find("<parameter=") {
            let param_after = &params[param_at + "<parameter=".len()..];
            let Some(param_name_end) = param_after.find('>') else {
                break;
            };
            let key = param_after[..param_name_end].trim().to_string();
            let value_src = &param_after[param_name_end + 1..];
            let (value, consumed) = if let Some(end) = value_src.find("</parameter>") {
                (
                    value_src[..end].trim().to_string(),
                    end + "</parameter>".len(),
                )
            } else {
                (value_src.trim().to_string(), value_src.len())
            };
            if !key.is_empty() {
                object.insert(key, serde_json::Value::String(value));
            }
            params = &param_after[param_name_end + 1 + consumed..];
        }
        if !name.is_empty() {
            calls.push(ParsedChatToolCall {
                name,
                arguments: serde_json::Value::Object(object).to_string(),
                id: String::new(),
            });
        }
        rest = if close < body.len() {
            &body[close + "</function>".len()..]
        } else {
            ""
        };
    }
    calls
}

fn recover_pythonic_tool_calls(text: &str) -> Vec<ParsedChatToolCall> {
    const START: &str = "<|tool_call_start|>";
    const END: &str = "<|tool_call_end|>";
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(START) {
        let after = &rest[start + START.len()..];
        let end = after.find(END).unwrap_or(after.len());
        let inner = after[..end].trim();
        let inner = inner.strip_prefix('[').unwrap_or(inner);
        let inner = inner.strip_suffix(']').unwrap_or(inner);
        if let Some(paren) = inner.find('(') {
            let name = inner[..paren].trim().to_string();
            let args_src = inner[paren + 1..].trim().trim_end_matches(')');
            if !name.is_empty() {
                calls.push(ParsedChatToolCall {
                    name,
                    arguments: pythonic_args_to_json(args_src).to_string(),
                    id: String::new(),
                });
            }
        }
        rest = if end < after.len() {
            &after[end + END.len()..]
        } else {
            ""
        };
    }
    calls
}

fn pythonic_args_to_json(src: &str) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for part in src.split(',') {
        let part = part.trim();
        let Some((key, value)) = part.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
        if !key.is_empty() {
            object.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
    }
    serde_json::Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TurnId;
    use std::collections::VecDeque;
    use std::thread;
    use std::time::Duration;

    struct ScriptedEngine {
        pieces: Vec<String>,
        delay: Duration,
        last_messages: Arc<Mutex<Vec<ChatMessage>>>,
        parse_outputs: VecDeque<ParsedChatOutput>,
        chat_tools: bool,
        parse_error: Option<String>,
        last_enable_thinking: Arc<Mutex<Option<bool>>>,
        piece_per_turn: bool,
    }

    impl Default for ScriptedEngine {
        fn default() -> Self {
            Self {
                pieces: Vec::new(),
                delay: Duration::ZERO,
                last_messages: Arc::new(Mutex::new(Vec::new())),
                parse_outputs: VecDeque::new(),
                chat_tools: true,
                parse_error: None,
                last_enable_thinking: Arc::new(Mutex::new(None)),
                piece_per_turn: false,
            }
        }
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
            if self.piece_per_turn {
                if self.pieces.is_empty() {
                    on_piece("", true)?;
                    return Ok(());
                }
                let piece = self.pieces.remove(0);
                on_piece(&piece, true)?;
                return Ok(());
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

        fn chat_tools_supported(&mut self) -> Result<bool> {
            Ok(self.chat_tools)
        }

        fn generate_with_common_chat(
            &mut self,
            messages_json: &str,
            _tools_json: &str,
            enable_thinking: bool,
            append_thinking_off_suffix: bool,
            cancel: &Cancel,
            on_piece: &mut dyn FnMut(&str, bool) -> Result<()>,
        ) -> Result<()> {
            *self.last_enable_thinking.lock().expect("thinking flag") = Some(enable_thinking);
            let messages = openai_messages_json_to_chat(messages_json)?;
            self.generate(&messages, append_thinking_off_suffix, cancel, on_piece)
        }

        fn chat_parse_output(
            &mut self,
            text: &str,
            is_partial: bool,
        ) -> std::result::Result<ParsedChatOutput, String> {
            if let Some(message) = self.parse_error.take() {
                return Err(message);
            }
            if let Some(output) = self.parse_outputs.pop_front() {
                return Ok(output);
            }
            let _ = is_partial;
            Ok(ParsedChatOutput {
                content: text.to_string(),
                tool_calls: Vec::new(),
            })
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
    fn skill_context_is_labelled_and_empty_context_preserves_prompt() {
        let context = "--- skill release-check | source: custom ---\nRead the release notes.\n";
        let rendered = render_system_prompt_with_skills("Speak plainly.", "en", context);
        assert!(rendered.contains("source: custom"));
        assert!(!rendered.contains("trust:"));
        assert!(rendered.contains("Read the release notes."));
        assert_eq!(
            render_system_prompt_with_skills("Speak plainly.", "en", ""),
            "Speak plainly."
        );
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
        }));
        assert_eq!(llm.name(), "local");
        assert_eq!(LFM3_1_7B_ASSET, BuiltinDefaults::v0().llm_model);
        assert_eq!(QWEN35_08B_ASSET, "qwen3.5-0.8b");
        assert_eq!(QWEN35_2B_ASSET, "qwen3.5-2b");
        assert_eq!(LLAMA_32_1B_ASSET, "llama-3.2-1b");
        assert_eq!(
            V0_LLM_MODELS,
            [
                LFM3_1_7B_ASSET,
                LFM3_1_7B_BF16_ASSET,
                LFM25_2_6B_ASSET,
                LFM25_350M_ASSET,
                LFM25_230M_ASSET,
                LLAMA_32_1B_ASSET,
                QWEN35_08B_ASSET,
                QWEN35_2B_ASSET
            ]
        );
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
    #[cfg(not(coverage))]
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
        assert!(err.to_string().contains("lfm3-1.7b"));
    }

    #[test]
    fn thinking_flag_does_not_cap_generation() {
        let off = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: (0..200).map(|i| format!("t{i}")).collect(),
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
    fn thinking_off_suffix_models_include_qwen_2b_and_lfm3() {
        assert!(is_thinking_tag_supported_model(QWEN35_2B_ASSET));
        assert!(is_thinking_tag_supported_model(LFM3_1_7B_ASSET));
        assert!(is_thinking_tag_supported_model(LFM3_1_7B_BF16_ASSET));
        assert!(!is_thinking_tag_supported_model(QWEN35_08B_ASSET));
        assert!(!is_thinking_tag_supported_model(LLAMA_32_1B_ASSET));
        assert!(!is_thinking_tag_supported_model(LFM25_2_6B_ASSET));
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
        assert_eq!(names, vec!["web_fetch", "web_search", "shell"]);
        let search_description = value[1]["function"]["description"]
            .as_str()
            .expect("search description");
        assert!(search_description.contains("without an API key"));
        let fetch_description = value[0]["function"]["description"]
            .as_str()
            .expect("fetch description");
        assert!(fetch_description.contains("Not for workspace files"));
        let shell = &value[2]["function"];
        assert!(shell["description"]
            .as_str()
            .expect("shell description")
            .contains("workspace sandbox"));
        assert!(shell["parameters"]["properties"]["workdir"]["description"]
            .as_str()
            .expect("workdir description")
            .contains("workspace"));
    }

    #[test]
    fn network_none_omits_web_tools_from_the_prompt() {
        let value: serde_json::Value =
            serde_json::from_str(&local_tool_definitions_json_for(NetworkMode::None))
                .expect("valid tools JSON");
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
        assert_eq!(names, vec!["shell"]);
    }

    #[test]
    fn recover_xml_and_pythonic_tool_calls() {
        let xml = recover_local_tool_calls(
            "<think></think>\n<tool_call>\n<function=shell>\n<parameter=command>\nfind . -name '*.rs'\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(xml.len(), 1);
        assert_eq!(xml[0].name, "shell");
        let args: serde_json::Value = serde_json::from_str(&xml[0].arguments).unwrap();
        assert_eq!(args["command"], "find . -name '*.rs'");

        let unclosed =
            recover_local_tool_calls("<function=shell><parameter=command>echo hi</parameter>");
        assert_eq!(unclosed.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&unclosed[0].arguments).unwrap();
        assert_eq!(args["command"], "echo hi");

        let unclosed_param = recover_local_tool_calls("<function=shell><parameter=command>pwd");
        assert_eq!(unclosed_param.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&unclosed_param[0].arguments).unwrap();
        assert_eq!(args["command"], "pwd");

        let skipped_empty =
            recover_local_tool_calls("<function=><parameter=command>x</parameter></function>");
        assert!(skipped_empty.is_empty());

        let python = recover_local_tool_calls(
            "<|tool_call_start|>[shell(command=\"cat README.md\")]<|tool_call_end|>",
        );
        assert_eq!(python.len(), 1);
        assert_eq!(python[0].name, "shell");
        let args: serde_json::Value = serde_json::from_str(&python[0].arguments).unwrap();
        assert_eq!(args["command"], "cat README.md");

        let python_unclosed = recover_local_tool_calls("<|tool_call_start|>shell(command=\"ls\")");
        assert_eq!(python_unclosed.len(), 1);
        assert_eq!(python_unclosed[0].name, "shell");

        let python_no_call =
            recover_local_tool_calls("<|tool_call_start|>not-a-call<|tool_call_end|>");
        assert!(python_no_call.is_empty());

        let prefers_xml = recover_local_tool_calls(
            "<function=shell><parameter=command>pwd</parameter></function><|tool_call_start|>[shell(command=\"ls\")]<|tool_call_end|>",
        );
        assert_eq!(prefers_xml.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&prefers_xml[0].arguments).unwrap();
        assert_eq!(args["command"], "pwd");
    }

    fn temp_harness_workspace(label: &str) -> std::path::PathBuf {
        let workspace = std::env::temp_dir().join(format!(
            "syllabix-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&workspace).unwrap();
        workspace
    }

    #[test]
    fn agent_harness_recovers_xml_when_common_chat_parse_fails() {
        let workspace = temp_harness_workspace("xml-parse");
        let xml = "<function=shell><parameter=command>echo coverage-xml</parameter></function>";
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec![xml.into(), "Recovered the shell call.".into()],
            parse_error: Some("unrecognized dialect".into()),
            parse_outputs: VecDeque::from([ParsedChatOutput {
                content: "Recovered the shell call.".into(),
                tool_calls: Vec::new(),
            }]),
            piece_per_turn: true,
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true)
        .with_workspace(&workspace);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "run echo"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("recovered xml after parse error");
        assert_eq!(chunks[0].text, "Recovered the shell call.");
        let events = Llm::take_tool_events(&mut llm);
        assert!(events
            .iter()
            .any(|event| event.kind == "call" && event.name == "shell"));
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[test]
    fn agent_harness_recovers_xml_swallowed_as_content() {
        let workspace = temp_harness_workspace("xml-content");
        let xml = "<function=shell><parameter=command>echo coverage-content</parameter></function>";
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec![xml.into(), "The swallowed call ran.".into()],
            parse_outputs: VecDeque::from([
                ParsedChatOutput {
                    content: xml.into(),
                    tool_calls: Vec::new(),
                },
                ParsedChatOutput {
                    content: "The swallowed call ran.".into(),
                    tool_calls: Vec::new(),
                },
            ]),
            piece_per_turn: true,
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true)
        .with_workspace(&workspace);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "run echo"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("recovered xml swallowed as content");
        assert_eq!(chunks[0].text, "The swallowed call ran.");
        let events = Llm::take_tool_events(&mut llm);
        assert!(events
            .iter()
            .any(|event| event.kind == "call" && event.name == "shell"));
        assert!(events.iter().any(|event| {
            event.kind == "result"
                && (event.content.contains("coverage-content")
                    || event.content.contains("SANDBOX_UNAVAILABLE"))
        }));
        let _ = std::fs::remove_dir_all(&workspace);
    }

    fn sample_result(ok: bool, content: &str) -> crate::types::ToolResult {
        crate::types::ToolResult {
            tool_call_id: "local-call-0".into(),
            ok,
            content: content.into(),
        }
    }

    #[test]
    fn duplicate_same_output_appends_a_hint_and_keeps_ok() {
        let args = serde_json::json!({"command": "echo same"});
        let first = sample_result(true, "exit: 0\nstdout:\nsame\nstderr:\n");
        let prior = vec![("shell".into(), args.clone(), first.content.clone())];
        let annotated = note_duplicate_same_output("shell", &args, first.clone(), &prior);
        assert!(annotated.ok);
        assert!(annotated.content.starts_with(&first.content));
        assert!(annotated.content.contains(DUPLICATE_SAME_OUTPUT_HINT));
    }

    #[test]
    fn duplicate_name_with_different_output_gets_no_hint() {
        let args = serde_json::json!({"command": "date"});
        let prior = vec![(
            "shell".into(),
            args.clone(),
            "exit: 0\nstdout:\nnoon\nstderr:\n".into(),
        )];
        let second = sample_result(true, "exit: 0\nstdout:\nmidnight\nstderr:\n");
        let annotated = note_duplicate_same_output("shell", &args, second.clone(), &prior);
        assert_eq!(annotated.ok, second.ok);
        assert_eq!(annotated.content, second.content);
        assert!(!annotated.content.contains(DUPLICATE_SAME_OUTPUT_HINT));
    }

    #[test]
    fn different_command_gets_no_hint_even_with_the_same_output() {
        let prior = vec![(
            "shell".into(),
            serde_json::json!({"command": "echo a"}),
            "exit: 0\nstdout:\nok\nstderr:\n".into(),
        )];
        let other = sample_result(true, "exit: 0\nstdout:\nok\nstderr:\n");
        let annotated = note_duplicate_same_output(
            "shell",
            &serde_json::json!({"command": "echo b"}),
            other.clone(),
            &prior,
        );
        assert_eq!(annotated.content, other.content);
        assert!(!annotated.content.contains(DUPLICATE_SAME_OUTPUT_HINT));
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
    fn web_search_string_count_normalizes_to_number() {
        let parsed = ParsedLocalToolCall {
            name: "web_search".into(),
            arguments: serde_json::json!({"query": "Tamil Nadu Chief Minister", "count": "5"}),
        };
        let normalized = normalize_local_tool_call(0, parsed).expect("normalize");
        assert_eq!(normalized.arguments["count"], serde_json::json!(5));
        let validated =
            crate::executor::validate_call(&normalized, &std::env::current_dir().expect("cwd"))
                .expect("string count validates after normalization");
        assert!(matches!(
            validated,
            crate::executor::ValidatedCall::WebSearch { count: 5, .. }
        ));
    }

    #[test]
    fn tool_results_become_tool_role_messages_in_order() {
        let llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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

    fn parsed_shell_call(command: &str) -> ParsedChatOutput {
        ParsedChatOutput {
            content: String::new(),
            tool_calls: vec![ParsedChatToolCall {
                name: "shell".into(),
                arguments: format!(r#"{{"command":"{command}"}}"#),
                id: String::new(),
            }],
        }
    }

    #[test]
    fn tool_turn_parses_scripted_calls_and_records_events() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["model raw".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
            parse_outputs: VecDeque::from([parsed_shell_call("date")]),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
        assert_eq!(LFM3_1_7B_ASSET, "lfm3-1.7b");
        assert_eq!(LFM3_1_7B_BF16_ASSET, "lfm3-1.7b-bf16");
        assert_eq!(LFM25_2_6B_ASSET, "lfm2.5-2.6b");
        assert_eq!(LFM25_350M_ASSET, "lfm2.5-350m");
        assert_eq!(LFM25_230M_ASSET, "lfm2.5-230m");
        assert!(is_v0_llm_model(LFM3_1_7B_ASSET));
        assert!(is_v0_llm_model(LFM3_1_7B_BF16_ASSET));
        assert!(is_v0_llm_model(LFM25_2_6B_ASSET));
        assert!(is_v0_llm_model(LFM25_350M_ASSET));
        assert!(is_v0_llm_model(LFM25_230M_ASSET));
        assert_eq!(LFM_TOOL_TURN_MAX_CHARS, LOCAL_TOOL_TURN_MAX_CHARS);
    }

    #[test]
    fn lfm_tool_turn_parses_scripted_calls_and_records_events() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["model raw".into()],
            delay: Duration::ZERO,
            last_messages: Arc::new(Mutex::new(Vec::new())),
            parse_outputs: VecDeque::from([parsed_shell_call("date")]),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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
            parse_outputs: VecDeque::new(),
            ..ScriptedEngine::default()
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

    #[test]
    fn parsed_chat_tool_calls_rejects_non_json_arguments() {
        let output = ParsedChatOutput {
            content: String::new(),
            tool_calls: vec![ParsedChatToolCall {
                name: "shell".into(),
                arguments: "not-json".into(),
                id: String::new(),
            }],
        };
        assert_eq!(
            parsed_chat_tool_calls(output).expect_err("bad json"),
            "tool call arguments are not valid JSON"
        );
    }

    #[test]
    fn common_chat_tool_turn_over_budget_records_rejection() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["x".repeat(LOCAL_TOOL_TURN_MAX_CHARS + 1)],
            ..ScriptedEngine::default()
        }));
        let mut chunks = Vec::new();
        let calls = llm
            .generate_tool_turn(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            })
            .expect("over-budget turn");
        assert!(calls.is_empty());
        assert!(chunks.is_empty());
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "rejected");
        assert!(events[0].content.contains("thinking bound"));
    }

    #[test]
    fn common_chat_passes_enable_thinking_to_the_engine() {
        let thinking_flag = Arc::new(Mutex::new(None));
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["ok".into()],
            parse_outputs: VecDeque::from([parsed_shell_call("date")]),
            last_enable_thinking: Arc::clone(&thinking_flag),
            ..ScriptedEngine::default()
        }))
        .with_thinking(true);
        llm.generate_tool_turn(&[], &user(0, "hi"), &Cancel::new(), &mut |_| Ok(()))
            .expect("tool turn");
        assert_eq!(*thinking_flag.lock().unwrap(), Some(true));
    }

    #[test]
    fn agent_harness_without_chat_tools_falls_back_to_plain_chat() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["plain reply".into()],
            chat_tools: false,
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("plain chat fallback");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "plain reply");
        assert!(Llm::take_tool_events(&mut llm).is_empty());
    }

    #[test]
    fn agent_harness_replies_directly_when_model_returns_content_only() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["spoken answer".into()],
            parse_outputs: VecDeque::from([ParsedChatOutput {
                content: "spoken answer".into(),
                tool_calls: Vec::new(),
            }]),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("harness reply");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "spoken answer");
    }

    #[test]
    fn agent_harness_executes_a_tool_then_speaks_the_follow_up() {
        let workspace = std::env::temp_dir().join(format!(
            "syllabix-harness-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&workspace).unwrap();
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["call".into(), "done".into()],
            parse_outputs: VecDeque::from([
                parsed_shell_call("echo harness-ok"),
                ParsedChatOutput {
                    content: "The command succeeded.".into(),
                    tool_calls: Vec::new(),
                },
            ]),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true)
        .with_workspace(&workspace);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "run echo"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("harness loop");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "The command succeeded.");
        let events = Llm::take_tool_events(&mut llm);
        assert!(events.iter().any(|event| event.kind == "call"));
        assert!(events.iter().any(|event| event.kind == "result"));
    }

    #[test]
    fn agent_harness_identical_echo_still_runs_and_hints_on_the_same_output() {
        let workspace = std::env::temp_dir().join(format!(
            "syllabix-harness-dup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&workspace).unwrap();
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["call".into(), "again".into(), "done".into()],
            parse_outputs: VecDeque::from([
                parsed_shell_call("echo harness-ok"),
                parsed_shell_call("echo harness-ok"),
                ParsedChatOutput {
                    content: "I used the same result.".into(),
                    tool_calls: Vec::new(),
                },
            ]),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true)
        .with_workspace(&workspace);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "run echo"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("harness continues after a repeated call");
        assert_eq!(chunks[0].text, "I used the same result.");
        let events = Llm::take_tool_events(&mut llm);
        let results: Vec<_> = events
            .iter()
            .filter(|event| event.kind == "result")
            .collect();
        assert_eq!(results.len(), 2);
        let ran = results[0].content.contains("harness-ok")
            || results[0].content.contains("SANDBOX_UNAVAILABLE");
        assert!(ran, "{}", results[0].content);
        assert!(!results[0].content.contains(DUPLICATE_SAME_OUTPUT_HINT));
        assert!(
            results[1].content.contains("harness-ok")
                || results[1].content.contains("SANDBOX_UNAVAILABLE"),
            "{}",
            results[1].content
        );
        assert!(results[1].content.contains(DUPLICATE_SAME_OUTPUT_HINT));
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[test]
    fn agent_harness_unknown_tool_returns_an_error_result() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["call".into(), "done".into()],
            parse_outputs: VecDeque::from([
                ParsedChatOutput {
                    content: String::new(),
                    tool_calls: vec![ParsedChatToolCall {
                        name: "read_file".into(),
                        arguments: r#"{"path":"README.md"}"#.into(),
                        id: String::new(),
                    }],
                },
                ParsedChatOutput {
                    content: "That tool is not available in this session.".into(),
                    tool_calls: Vec::new(),
                },
            ]),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true);
        let mut chunks = Vec::new();
        llm.generate(
            &[],
            &user(0, "read the readme"),
            &Cancel::new(),
            &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            },
        )
        .expect("harness continues after unknown tool");
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].text,
            "That tool is not available in this session."
        );
        let events = Llm::take_tool_events(&mut llm);
        assert!(events
            .iter()
            .any(|event| event.kind == "call" && event.name == "read_file"));
        assert!(events
            .iter()
            .any(|event| { event.kind == "result" && event.content.contains("unknown name") }));
    }

    #[test]
    fn agent_harness_parse_error_emits_spoken_fallback() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["bad".into()],
            parse_error: Some("parse failed".into()),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("harness fallback");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, LOCAL_TOOL_FALLBACK_TEXT);
    }

    #[test]
    fn strip_unfinished_reasoning_removes_open_blocks() {
        assert_eq!(
            strip_unfinished_reasoning_from_spoken_content(
                "<think>\nplanning steps\n</think>\nHello"
            ),
            "Hello"
        );
        assert!(strip_unfinished_reasoning_from_spoken_content(
            "<think>\nThe user wants me to write outside\n"
        )
        .is_empty());
    }

    #[test]
    fn agent_harness_does_not_speak_unclosed_reasoning() {
        let mut llm = LlamaLlm::with_engine(Box::new(ScriptedEngine {
            pieces: vec!["<think>\nsecret plan\n".into()],
            parse_outputs: VecDeque::from([ParsedChatOutput {
                content: "secret plan leaked".into(),
                tool_calls: Vec::new(),
            }]),
            ..ScriptedEngine::default()
        }))
        .with_agent_harness(true);
        let mut chunks = Vec::new();
        llm.generate(&[], &user(0, "hi"), &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("harness reply");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, LOCAL_TOOL_FALLBACK_TEXT);
    }
}
