//! Eval-only LLM stage for Riley voice sessions (#329 phase A2).
//!
//! Implements [`syllabix_core::Llm`] against an OpenAI-compatible
//! chat-completions endpoint with the Riley prompt and the five card tools.
//! The production `OpenAiLlm` advertises a fixed tool menu (`web_fetch`,
//! `web_search`, `shell`) and Goose attaches only `syllabix mcp`, so this
//! adapter keeps the default harness unchanged. Tool calls run in-process
//! through [`crate::tools::execute`] and are reported with
//! [`Llm::take_tool_events`], so the pipeline emits them as
//! `LoopEvent::Tool` exactly like the built-in harness.
//!
//! The key comes from `SYLLABIX_LLM_API_KEY` via
//! [`syllabix_core::openai::resolve_api_key`] and is never printed.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use syllabix_core::types::{
    GenerationId, HistoryTurn, LlmDebugMeta, TokenChunk, ToolCall, ToolResult, ToolTurnEvent,
    Transcript,
};
use syllabix_core::{Cancel, Error, Llm, Result};
use zeroize::Zeroizing;

use crate::db::CardDb;
use crate::{tools, AGENT_PROMPT};

/// Config / diagnostics name of this stage.
pub const PROVIDER_NAME: &str = "riley-openai";

/// Model rounds allowed per user turn before the turn fails.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 8;

/// Budget for one chat-completions request.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// How often a blocked request checks for barge-in or shutdown.
const POLL_TICK: Duration = Duration::from_millis(20);

/// Endpoint and model for one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RileyLlmSettings {
    /// Absolute chat-completions URL.
    pub endpoint: String,
    /// Model id sent verbatim.
    pub model: String,
    /// See [`DEFAULT_MAX_TOOL_ROUNDS`].
    pub max_tool_rounds: usize,
    /// See [`DEFAULT_REQUEST_TIMEOUT`].
    pub request_timeout: Duration,
}

impl RileyLlmSettings {
    /// `{base_url}/chat/completions` with the default limits.
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            endpoint: syllabix_core::openai::join_endpoint(base_url),
            model: model.to_string(),
            max_tool_rounds: DEFAULT_MAX_TOOL_ROUNDS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

/// Runs one tool call. Production sessions use [`RileyLlm::with_card_db`].
pub type ToolExecutor = Box<dyn FnMut(&ToolCall) -> ToolResult + Send>;

/// Riley agent stage: prompt + tools + per-session message log.
pub struct RileyLlm {
    settings: RileyLlmSettings,
    api_key: Zeroizing<String>,
    agent: ureq::Agent,
    execute: ToolExecutor,
    /// Committed messages after the system prompt, including tool rounds, so
    /// later turns remember earlier lookups.
    messages: Vec<Value>,
    events: Vec<ToolTurnEvent>,
}

impl RileyLlm {
    /// Stage with a custom tool executor.
    pub fn new(
        settings: RileyLlmSettings,
        api_key: Zeroizing<String>,
        execute: ToolExecutor,
    ) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(settings.request_timeout)
            .build();
        Self {
            settings,
            api_key,
            agent,
            execute,
            messages: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Stage whose tools run against `db` (one Postgres schema).
    pub fn with_card_db(
        settings: RileyLlmSettings,
        api_key: Zeroizing<String>,
        mut db: CardDb,
    ) -> Self {
        Self::new(
            settings,
            api_key,
            Box::new(move |call| tools::execute(&mut db, call)),
        )
    }

    /// Settings used by this stage.
    pub fn settings(&self) -> &RileyLlmSettings {
        &self.settings
    }

    fn request_body(&self, pending: &[Value]) -> Value {
        let mut messages = vec![json!({ "role": "system", "content": AGENT_PROMPT })];
        messages.extend(self.messages.iter().cloned());
        messages.extend(pending.iter().cloned());
        json!({
            "model": self.settings.model,
            "messages": messages,
            "tools": tools::openai_tool_definitions(),
            "tool_choice": "auto",
            "stream": false,
        })
    }

    fn note(&mut self, kind: &str, name: &str, call_id: &str, arguments: &str, content: &str) {
        self.events.push(ToolTurnEvent {
            kind: kind.into(),
            name: name.into(),
            call_id: call_id.into(),
            arguments: arguments.into(),
            content: content.into(),
        });
    }

    /// POST on a worker thread so barge-in and shutdown stay responsive.
    fn post(&self, body: String, cancel: &Cancel, generation: GenerationId) -> Result<Value> {
        let (tx, rx) = mpsc::channel();
        let request = self
            .agent
            .post(&self.settings.endpoint)
            .set(
                "Authorization",
                &format!("Bearer {}", self.api_key.as_str()),
            )
            .set("Content-Type", "application/json");
        std::thread::Builder::new()
            .name("riley-llm".into())
            .spawn(move || {
                let outcome = match request.send_string(&body) {
                    Ok(response) => response
                        .into_string()
                        .map_err(|err| format!("could not read the response: {err}")),
                    Err(ureq::Error::Status(code, _)) => {
                        Err(format!("chat completions returned HTTP {code}"))
                    }
                    Err(ureq::Error::Transport(err)) => {
                        Err(format!("chat completions request failed: {}", err.kind()))
                    }
                };
                let _ = tx.send(outcome);
            })
            .map_err(|err| provider(format!("failed to start request worker: {err}")))?;
        let started = Instant::now();
        loop {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            match rx.recv_timeout(POLL_TICK) {
                Ok(Ok(text)) => {
                    return serde_json::from_str(&text)
                        .map_err(|_| provider("chat completions returned invalid JSON".into()))
                }
                Ok(Err(message)) => return Err(provider(message)),
                Err(RecvTimeoutError::Timeout)
                    if started.elapsed() >= self.settings.request_timeout =>
                {
                    return Err(provider("chat completions request timed out".into()))
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(provider("request worker exited without a reply".into()))
                }
            }
        }
    }
}

fn provider(message: String) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message,
    }
}

/// Parse one `tool_calls[]` entry. Unknown names are left to
/// [`tools::execute`], which answers `{"error": "unknown tool: …"}`.
pub fn parse_tool_call(raw: &Value) -> std::result::Result<ToolCall, String> {
    let id = raw
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| "tool call is missing id".to_string())?;
    let function = raw.get("function").unwrap_or(&Value::Null);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "tool call is missing a function name".to_string())?;
    let text = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let arguments: Value = serde_json::from_str(if text.trim().is_empty() { "{}" } else { text })
        .map_err(|_| "tool call arguments are not valid JSON".to_string())?;
    if !arguments.is_object() {
        return Err("tool call arguments must be a JSON object".into());
    }
    Ok(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    })
}

impl Llm for RileyLlm {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        Some(LlmDebugMeta {
            provider: PROVIDER_NAME.into(),
            model: self.settings.model.clone(),
            endpoint: self.settings.endpoint.clone(),
            request_id: String::new(),
        })
    }

    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        std::mem::take(&mut self.events)
    }

    /// One user turn: call the model, run every tool call it asks for, and
    /// repeat until it answers with text. The reply is one final token, like
    /// the built-in API harness. `_history` is ignored: this stage keeps its
    /// own log so tool rounds survive across turns.
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
        let generation = cancel.generation();
        let mut pending = vec![json!({ "role": "user", "content": user.text })];
        let outcome = self.run_turn(&mut pending, cancel, generation);
        // Keep completed rounds even when the turn fails: their side effects
        // already happened in the database.
        self.messages.append(&mut pending);
        let text = outcome?;
        on_token(TokenChunk {
            turn: user.turn,
            generation,
            index: 0,
            text,
            is_last: true,
        })
    }
}

impl RileyLlm {
    fn run_turn(
        &mut self,
        pending: &mut Vec<Value>,
        cancel: &Cancel,
        generation: GenerationId,
    ) -> Result<String> {
        for _ in 0..self.settings.max_tool_rounds {
            let body = self.request_body(pending).to_string();
            let response = self.post(body, cancel, generation)?;
            let message = response
                .pointer("/choices/0/message")
                .filter(|message| message.is_object())
                .cloned()
                .ok_or_else(|| provider("response has no choices[0].message".into()))?;
            let calls = message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if calls.is_empty() {
                let text = message
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if text.is_empty() {
                    return Err(provider("model returned an empty reply".into()));
                }
                pending.push(json!({ "role": "assistant", "content": text }));
                return Ok(text);
            }
            pending.push(json!({
                "role": "assistant",
                "content": message.get("content").cloned().unwrap_or(Value::Null),
                "tool_calls": calls,
            }));
            for (index, raw) in calls.iter().enumerate() {
                let result = match parse_tool_call(raw) {
                    Ok(call) => {
                        let arguments = call.arguments.to_string();
                        self.note("call", &call.name, &call.id, &arguments, "");
                        let result = (self.execute)(&call);
                        self.note("result", &call.name, &call.id, &arguments, &result.content);
                        result
                    }
                    Err(message) => {
                        let id = raw
                            .get("id")
                            .and_then(Value::as_str)
                            .map_or_else(|| format!("riley-invalid-{index}"), str::to_string);
                        let content = json!({ "error": message }).to_string();
                        self.note("rejected", "", &id, "", &content);
                        ToolResult {
                            tool_call_id: id,
                            content,
                            ok: false,
                        }
                    }
                };
                pending.push(json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }));
            }
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
        }
        self.note("limit", "", "", "", "tool round limit reached");
        Err(provider(format!(
            "no reply after {} tool rounds",
            self.settings.max_tool_rounds
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_join_the_chat_completions_path() {
        let settings = RileyLlmSettings::new("http://127.0.0.1:9/v1/", "m");
        assert_eq!(settings.endpoint, "http://127.0.0.1:9/v1/chat/completions");
        assert_eq!(settings.model, "m");
        assert_eq!(settings.max_tool_rounds, DEFAULT_MAX_TOOL_ROUNDS);
        assert_eq!(settings.request_timeout, DEFAULT_REQUEST_TIMEOUT);
    }

    #[test]
    fn parse_tool_call_accepts_openai_shape_and_rejects_bad_input() {
        let call = parse_tool_call(&json!({
            "id": "c1",
            "type": "function",
            "function": {"name": "display_card_info_by_last4", "arguments": "{\"last4\":\"1111\"}"}
        }))
        .unwrap();
        assert_eq!(call.id, "c1");
        assert_eq!(call.name, "display_card_info_by_last4");
        assert_eq!(call.arguments, json!({"last4": "1111"}));
        let empty =
            parse_tool_call(&json!({"id": "c2", "function": {"name": "x", "arguments": ""}}));
        assert_eq!(empty.unwrap().arguments, json!({}));
        assert_eq!(
            parse_tool_call(&json!({"function": {"name": "x"}})).unwrap_err(),
            "tool call is missing id"
        );
        assert_eq!(
            parse_tool_call(&json!({"id": "c3", "function": {}})).unwrap_err(),
            "tool call is missing a function name"
        );
        assert_eq!(
            parse_tool_call(&json!({"id": "c4", "function": {"name": "x", "arguments": "{"}}))
                .unwrap_err(),
            "tool call arguments are not valid JSON"
        );
        assert_eq!(
            parse_tool_call(&json!({"id": "c5", "function": {"name": "x", "arguments": "[1]"}}))
                .unwrap_err(),
            "tool call arguments must be a JSON object"
        );
    }

    #[test]
    fn request_body_carries_the_riley_prompt_and_only_the_five_tools() {
        let llm = RileyLlm::new(
            RileyLlmSettings::new("http://127.0.0.1:9/v1", "m"),
            Zeroizing::new("k".into()),
            Box::new(|call| ToolResult {
                tool_call_id: call.id.clone(),
                content: "{}".into(),
                ok: true,
            }),
        );
        let body = llm.request_body(&[json!({"role": "user", "content": "hi"})]);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], AGENT_PROMPT);
        assert_eq!(body["messages"][1]["content"], "hi");
        assert_eq!(body["stream"], false);
        let names: Vec<&str> = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, tools::TOOL_NAMES);
        assert_eq!(llm.name(), PROVIDER_NAME);
        assert_eq!(llm.settings().model, "m");
        let meta = llm.debug_meta().unwrap();
        assert_eq!(meta.endpoint, "http://127.0.0.1:9/v1/chat/completions");
    }
}
