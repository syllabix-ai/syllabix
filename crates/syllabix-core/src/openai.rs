//! OpenAI-compatible streaming `chat/completions` adapter using a user-supplied key.
//!
//! VAD, AEC, STT, and TTS stay on-device; only transcript text reaches the
//! endpoint the user chose. The API key comes from the `SYLLABIX_LLM_API_KEY`
//! environment variable **only** — never yaml, never a `.env` file — and is
//! held in zeroizing memory for the life of the run. A failed turn speaks a
//! short fallback instead of hanging; there is no auto-retry.

use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ureq::Agent;
use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::{Error, Result};
use crate::executor;
use crate::llm::LLAMA_MAX_HISTORY_TURNS;
use crate::providers::Llm;
use crate::types::{
    HistoryTurn, LlmDebugMeta, TokenChunk, ToolCall, ToolResult, ToolTurnEvent, Transcript,
};

/// The exploration is deliberately bounded before any executor exists.
pub const MAX_TOOL_CALLS_PER_TURN: usize = 5;
/// Tool result text is bounded before it can re-enter a model context.
pub const MAX_TOOL_RESULT_BYTES: usize = 8 * 1024;

/// Environment variable carrying the BYO key. The only supported source.
pub const API_KEY_ENV: &str = "SYLLABIX_LLM_API_KEY";

/// Endpoint used when `pipeline.llm.base_url` is not set.
pub const DEFAULT_LLM_BASE_URL: &str = "https://api.openai.com/v1";

/// Config / log name.
/// Config / log name for the remote execution model.
pub const PROVIDER_NAME: &str = "online";

/// Response header echoed into diagnostics sidecars when present.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Hard connect timeout. A dead endpoint fails the turn instead of hanging.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Socket read granularity inside the stream worker. Also bounds how long a
/// cancelled connection lingers before the worker observes the abandon flag.
pub const READ_POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// No bytes for this long aborts the turn as a provider failure.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(20);

/// How often the generate loop wakes to check cancel and idle deadlines.
const POLL_TICK: Duration = Duration::from_millis(100);

/// Spoken when a cloud turn fails. Short, plain, no Markdown.
pub const CLOUD_FALLBACK_TEXT: &str = "Sorry, I could not reach the language model.";
pub const TOOL_LIMIT_TEXT: &str = "Sorry, I reached the tool-call limit for this turn.";

/// Reject or accept a `pipeline.llm.base_url` value.
///
/// Absolute http(s) URL with a host and no embedded credentials (keys belong
/// in [`API_KEY_ENV`], not in shared config files).
pub fn validate_base_url(raw: &str) -> std::result::Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("must be an absolute http(s) URL, e.g. https://api.groq.com/openai/v1".into());
    }
    let url = url::Url::parse(trimmed)
        .map_err(|err| format!("{trimmed:?} is not a valid URL ({err}); use https://host/v1"))?;
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "scheme {other:?} is not supported; use http or https"
            ))
        }
    }
    if url.host_str().is_none() {
        return Err("URL needs a host, e.g. https://api.openai.com/v1".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(
            "credentials in the URL are not supported; keep the key in SYLLABIX_LLM_API_KEY".into(),
        );
    }
    Ok(())
}

/// Resolve the BYO key from the environment. Missing or empty fails fast with
/// an actionable message so `run` never reaches audio setup.
pub fn resolve_api_key(getenv: impl Fn(&str) -> Option<String>) -> Result<Zeroizing<String>> {
    match getenv(API_KEY_ENV) {
        Some(key) if !key.trim().is_empty() => Ok(Zeroizing::new(key)),
        Some(_) => Err(Error::Config {
            field: API_KEY_ENV.into(),
            message: "is set but empty. Export your key, e.g. export SYLLABIX_LLM_API_KEY=sk-…, \
                      and start syllabix again."
                .into(),
        }),
        None => Err(Error::Config {
            field: API_KEY_ENV.into(),
            message: format!(
                "is required when pipeline.llm.provider is {PROVIDER_NAME}. \
                 Supply it for one run (SYLLABIX_LLM_API_KEY=sk-… syllabix run) or export it \
                 (export SYLLABIX_LLM_API_KEY=sk-…) and start syllabix again. \
                 Keys are read from the environment only — never from syllabix.yaml \
                 or a .env file."
            ),
        }),
    }
}

/// Endpoint + model resolved from [`AgentConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiSettings {
    /// Absolute chat-completions URL (`{base_url}/chat/completions`).
    pub endpoint: String,
    /// Model id sent verbatim; the endpoint decides what it serves.
    pub model: String,
    /// System prompt template (`pipeline.llm.system_prompt`). `{language}` is
    /// replaced at generate time, same as the local engine.
    pub system_prompt: String,
    /// Whether this explicit developer-only run advertises the two developer
    /// harness schemas. Default runs omit the API `tools` member entirely.
    pub developer_harness: bool,
}

impl OpenAiSettings {
    /// Build from a validated config (default endpoint when `base_url` is unset).
    pub fn from_config(config: &AgentConfig) -> Self {
        let base = config
            .llm_base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_LLM_BASE_URL.to_string());
        Self {
            endpoint: join_endpoint(&base),
            model: config.llm_model.clone(),
            system_prompt: config.system_prompt.clone(),
            developer_harness: config.llm_developer_harness,
        }
    }
}

/// `{base}/chat/completions`, tolerating a trailing slash.
pub fn join_endpoint(base: &str) -> String {
    format!("{}/chat/completions", base.trim_end_matches('/'))
}

/// Timeouts, injectable so tests can exercise failure paths in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiTimeouts {
    /// TCP + TLS connect budget.
    pub connect: Duration,
    /// Socket read granularity inside the stream worker.
    pub read_poll: Duration,
    /// Silence budget between bytes before the turn fails.
    pub idle: Duration,
}

impl Default for OpenAiTimeouts {
    fn default() -> Self {
        Self {
            connect: CONNECT_TIMEOUT,
            read_poll: READ_POLL_TIMEOUT,
            idle: IDLE_TIMEOUT,
        }
    }
}

/// Streaming client for one OpenAI-compatible endpoint.
pub struct OpenAiLlm {
    settings: OpenAiSettings,
    api_key: Zeroizing<String>,
    agent: Agent,
    timeouts: OpenAiTimeouts,
    last_request_id: Mutex<Option<String>>,
    tool_events: Mutex<Vec<ToolTurnEvent>>,
}

impl OpenAiLlm {
    /// Create an adapter with the built-in connect and idle timeouts.
    pub fn new(settings: OpenAiSettings, api_key: Zeroizing<String>) -> Self {
        Self::with_timeouts(settings, api_key, OpenAiTimeouts::default())
    }

    /// Constructor with explicit timeouts (tests) and rustls-only agent.
    pub fn with_timeouts(
        settings: OpenAiSettings,
        api_key: Zeroizing<String>,
        timeouts: OpenAiTimeouts,
    ) -> Self {
        Self {
            settings,
            api_key,
            agent: build_agent(timeouts),
            timeouts,
            last_request_id: Mutex::new(None),
            tool_events: Mutex::new(Vec::new()),
        }
    }

    /// Endpoint used by this run.
    pub fn settings(&self) -> &OpenAiSettings {
        &self.settings
    }
}

/// rustls-only HTTP agent (ureq's `tls` feature in this workspace pin adds no
/// libssl) with the connect and socket-read budgets applied.
fn build_agent(timeouts: OpenAiTimeouts) -> Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(timeouts.connect)
        .timeout_read(timeouts.read_poll)
        .user_agent(concat!("syllabix/", env!("CARGO_PKG_VERSION")))
        .build()
}

/// Events forwarded from the stream worker to the generate loop.
enum StreamEvent {
    /// One non-empty assistant delta.
    Delta(String),
    /// One fragment of an OpenAI-compatible streamed tool call.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
    /// Response id header captured for the diagnostics sidecar.
    RequestId(String),
    /// Server sent `data: [DONE]`.
    Finished,
}

impl Llm for OpenAiLlm {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        Some(LlmDebugMeta {
            provider: PROVIDER_NAME.into(),
            model: self.settings.model.clone(),
            endpoint: self.settings.endpoint.clone(),
            request_id: self
                .last_request_id
                .lock()
                .expect("openai request-id mutex")
                .clone()
                .unwrap_or_default(),
        })
    }

    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        std::mem::take(&mut *self.tool_events.lock().expect("openai tool-events mutex"))
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
        if self.settings.developer_harness {
            return self.generate_with_tools(history, user, cancel, on_token);
        }
        let generation = cancel.generation();
        let body = request_body(
            &self.settings.model,
            history,
            user,
            &self.settings.system_prompt,
        )
        .to_string();

        // One worker per turn owns the blocking HTTP read; the generate loop
        // polls events so barge-in cancel surfaces within POLL_TICK. Setting
        // the abandon flag (on any exit) makes the worker drop the reader,
        // which closes the socket and aborts the server-side stream.
        let abandon = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>();
        let worker = StreamWorker {
            agent: self.agent.clone(),
            endpoint: self.settings.endpoint.clone(),
            api_key: self.api_key.as_str().to_string(),
            body,
            abandon: Arc::clone(&abandon),
            tx,
        };
        std::thread::Builder::new()
            .name("syllabix-openai".into())
            .spawn(move || worker.run())
            .map_err(|err| Error::Provider {
                provider: PROVIDER_NAME,
                message: format!("failed to start stream worker: {err}"),
            })?;
        let _abandon_on_drop = AbandonOnDrop(Arc::clone(&abandon));

        let mut index = 0u32;
        // One-deep lookahead: the final delta carries `is_last`, mirroring the
        // local engines also mark the final token as the end of the generation.
        let mut pending: Option<TokenChunk> = None;
        let mut idle_since = Instant::now();
        let outcome = loop {
            if cancel.is_stale(generation) || cancel.is_shutdown() {
                break Err(Error::Cancelled);
            }
            match rx.recv_timeout(POLL_TICK) {
                Ok(Ok(StreamEvent::RequestId(id))) => {
                    *self
                        .last_request_id
                        .lock()
                        .expect("openai request-id mutex") = Some(id);
                    idle_since = Instant::now();
                }
                Ok(Ok(StreamEvent::Delta(text))) => {
                    idle_since = Instant::now();
                    if let Some(earlier) = pending.replace(TokenChunk {
                        turn: user.turn,
                        generation,
                        index,
                        text,
                        is_last: false,
                    }) {
                        index += 1;
                        if on_token(earlier).is_err() {
                            break Err(Error::Cancelled);
                        }
                    }
                }
                Ok(Ok(StreamEvent::ToolCallDelta { .. })) => {
                    break Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "received a tool call without developer harness enabled".into(),
                    });
                }
                Ok(Ok(StreamEvent::Finished)) => {
                    let mut chunk = pending.take().unwrap_or(TokenChunk {
                        turn: user.turn,
                        generation,
                        index,
                        text: String::new(),
                        is_last: false,
                    });
                    chunk.index = index;
                    chunk.is_last = true;
                    if on_token(chunk).is_err() {
                        break Err(Error::Cancelled);
                    }
                    break Ok(());
                }
                Ok(Err(err)) => break Err(err),
                Err(RecvTimeoutError::Timeout) => {
                    if idle_since.elapsed() >= self.timeouts.idle {
                        break Err(Error::Provider {
                            provider: PROVIDER_NAME,
                            message: format!("no data for {:?} (idle timeout)", self.timeouts.idle),
                        });
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    break Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "stream worker exited without a verdict".into(),
                    });
                }
            }
        };

        match outcome {
            Ok(()) => Ok(()),
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(err) => {
                tracing::warn!(provider = PROVIDER_NAME, error = %err, "cloud turn failed");
                speak_fallback(user.turn, generation, cancel, on_token)
            }
        }
    }
}

#[derive(Debug, Default)]
struct RawToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAiLlm {
    /// API-native developer-harness loop. Calls are normalized once, then
    /// dispatched through the host-owned bounded executor.
    fn generate_with_tools(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        let generation = cancel.generation();
        let mut body = request_body(
            &self.settings.model,
            history,
            user,
            &self.settings.system_prompt,
        );
        body["tools"] = tool_definitions();
        let mut call_count = 0usize;

        loop {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            let (content, raw_calls) = match self.collect_tool_response(body.to_string(), cancel) {
                Ok(response) => response,
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(err) => {
                    tracing::warn!(provider = PROVIDER_NAME, error = %err, "cloud tool turn failed");
                    return speak_fallback(user.turn, generation, cancel, on_token);
                }
            };
            if raw_calls.is_empty() {
                return on_token(TokenChunk {
                    turn: user.turn,
                    generation,
                    index: 0,
                    text: content,
                    is_last: true,
                });
            }

            let calls = raw_calls
                .into_iter()
                .map(normalize_tool_call)
                .collect::<std::result::Result<Vec<_>, _>>();
            let calls = match calls {
                Ok(calls) => calls,
                Err(message) => {
                    self.note_tool_event("rejected", "", "", "", &message);
                    return speak_fallback(user.turn, generation, cancel, on_token);
                }
            };
            if call_count.saturating_add(calls.len()) > MAX_TOOL_CALLS_PER_TURN {
                self.note_tool_event("limit", "", "", "", TOOL_LIMIT_TEXT);
                return on_token(TokenChunk {
                    turn: user.turn,
                    generation,
                    index: 0,
                    text: TOOL_LIMIT_TEXT.into(),
                    is_last: true,
                });
            }
            call_count += calls.len();
            append_tool_call_message(&mut body, &calls);
            for call in calls {
                self.note_tool_event(
                    "call",
                    &call.name,
                    &call.id,
                    &call.arguments.to_string(),
                    "",
                );
                let workspace = std::env::current_dir().map_err(|err| Error::Provider {
                    provider: PROVIDER_NAME,
                    message: format!("developer workspace is unavailable: {err}"),
                })?;
                let result = executor::execute(&call, &workspace, cancel);
                if cancel.is_shutdown() || cancel.is_stale(generation) {
                    return Err(Error::Cancelled);
                }
                self.note_tool_event(
                    "result",
                    &call.name,
                    &call.id,
                    &call.arguments.to_string(),
                    &result.content,
                );
                append_tool_result_message(&mut body, &result);
            }
        }
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
            .expect("openai tool-events mutex")
            .push(ToolTurnEvent {
                kind: kind.into(),
                name: name.into(),
                call_id: call_id.into(),
                arguments: truncate_bytes(arguments, MAX_TOOL_RESULT_BYTES),
                content: truncate_bytes(content, MAX_TOOL_RESULT_BYTES),
            });
    }

    fn collect_tool_response(
        &mut self,
        body: String,
        cancel: &Cancel,
    ) -> Result<(String, Vec<RawToolCall>)> {
        let abandon = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<StreamEvent>>();
        let worker = StreamWorker {
            agent: self.agent.clone(),
            endpoint: self.settings.endpoint.clone(),
            api_key: self.api_key.as_str().to_string(),
            body,
            abandon: Arc::clone(&abandon),
            tx,
        };
        std::thread::Builder::new()
            .name("syllabix-openai".into())
            .spawn(move || worker.run())
            .map_err(|err| Error::Provider {
                provider: PROVIDER_NAME,
                message: format!("failed to start stream worker: {err}"),
            })?;
        let _abandon_on_drop = AbandonOnDrop(abandon);
        let generation = cancel.generation();
        let mut idle_since = Instant::now();
        let mut content = String::new();
        let mut calls: Vec<RawToolCall> = Vec::new();
        loop {
            if cancel.is_shutdown() || cancel.is_stale(generation) {
                return Err(Error::Cancelled);
            }
            match rx.recv_timeout(POLL_TICK) {
                Ok(Ok(StreamEvent::RequestId(id))) => {
                    *self
                        .last_request_id
                        .lock()
                        .expect("openai request-id mutex") = Some(id);
                    idle_since = Instant::now();
                }
                Ok(Ok(StreamEvent::Delta(text))) => {
                    content.push_str(&text);
                    idle_since = Instant::now();
                }
                Ok(Ok(StreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments,
                })) => {
                    let slot = calls.get_mut(index);
                    if slot.is_none() {
                        calls.resize_with(index + 1, RawToolCall::default);
                    }
                    let slot = calls.get_mut(index).expect("tool call slot");
                    if id.is_some() {
                        slot.id = id;
                    }
                    if name.is_some() {
                        slot.name = name;
                    }
                    if let Some(arguments) = arguments {
                        slot.arguments.push_str(&arguments);
                    }
                    idle_since = Instant::now();
                }
                Ok(Ok(StreamEvent::Finished)) => return Ok((content, calls)),
                Ok(Err(err)) => return Err(err),
                Err(RecvTimeoutError::Timeout) if idle_since.elapsed() >= self.timeouts.idle => {
                    return Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: format!("no data for {:?} (idle timeout)", self.timeouts.idle),
                    })
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "stream worker exited without a verdict".into(),
                    })
                }
            }
        }
    }
}

fn tool_definitions() -> serde_json::Value {
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
                "name": "web_search",
                "description": "Search the public web without an API key. Use plain keywords only; site:, filetype:, quotes, and other search operators are not supported.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (1-256 chars)."
                        },
                        "count": {
                            "type": "number",
                            "description": "How many results to return (1-8, default 5)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a read-only command via direct argv in the workspace. Allowed commands: date (current time), df (disk space, e.g. ['df', '-h', '.']), pwd, ls (list directory), git (status, diff, log, show, branch), find (find files by name), cargo (metadata, tree). Content search is not available; use find to locate files by name.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["argv"],
                    "properties": {
                        "argv": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Command and argument array, e.g. ['df', '-h', '.'] or ['git', 'status', '--short']."
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
}

fn normalize_tool_call(raw: RawToolCall) -> std::result::Result<ToolCall, String> {
    let id = raw
        .id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "tool call is missing id".to_string())?;
    let name = raw
        .name
        .filter(|value| matches!(value.as_str(), "web_fetch" | "web_search" | "shell"))
        .ok_or_else(|| "tool call has an unknown name".to_string())?;
    let arguments: serde_json::Value = serde_json::from_str(&raw.arguments)
        .map_err(|_| "tool call arguments are not valid JSON".to_string())?;
    if !arguments.is_object() {
        return Err("tool call arguments must be a JSON object".into());
    }
    Ok(ToolCall {
        id,
        name,
        arguments,
    })
}

fn append_tool_call_message(body: &mut serde_json::Value, calls: &[ToolCall]) {
    let wire_calls: Vec<_> = calls.iter().map(|call| serde_json::json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}})).collect();
    body["messages"]
        .as_array_mut()
        .expect("request messages")
        .push(serde_json::json!({"role":"assistant","tool_calls":wire_calls}));
}

fn append_tool_result_message(body: &mut serde_json::Value, result: &ToolResult) {
    body["messages"].as_array_mut().expect("request messages").push(serde_json::json!({"role":"tool","tool_call_id":result.tool_call_id,"content":result.content}));
}

fn truncate_bytes(value: &str, max: usize) -> String {
    if value.len() <= max {
        return value.to_string();
    }
    let mut end = max;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

struct StreamWorker {
    agent: Agent,
    endpoint: String,
    api_key: String,
    body: String,
    abandon: Arc<AtomicBool>,
    tx: Sender<Result<StreamEvent>>,
}

impl StreamWorker {
    fn run(self) {
        self.post_and_stream();
        // Dropping `self.tx` here marks end-of-events for the generate loop.
    }

    fn post_and_stream(self) {
        let Self {
            agent,
            endpoint,
            api_key,
            body,
            abandon,
            tx,
        } = self;
        let send = |event: Result<StreamEvent>| -> bool { tx.send(event).is_ok() };
        let response = match agent
            .post(&endpoint)
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_string(&body)
        {
            Ok(response) => response,
            Err(ureq::Error::Status(code, response)) => {
                let mut text = String::new();
                let _ = response
                    .into_reader()
                    .take(16_384)
                    .read_to_string(&mut text);
                let hint = if code == 401 || code == 403 {
                    format!(" check {API_KEY_ENV}")
                } else {
                    String::new()
                };
                let body_trim = text.trim();
                let detail = if body_trim.is_empty() {
                    String::new()
                } else {
                    format!(": {}", body_trim.chars().take(512).collect::<String>())
                };
                send(Err(Error::Provider {
                    provider: PROVIDER_NAME,
                    message: format!("HTTP {code}{detail}{hint}"),
                }));
                return;
            }
            Err(err) => {
                send(Err(Error::Provider {
                    provider: PROVIDER_NAME,
                    message: format!("request failed: {err}"),
                }));
                return;
            }
        };
        if let Some(id) = response.header(REQUEST_ID_HEADER) {
            if !id.is_empty() && !send(Ok(StreamEvent::RequestId(id.to_string()))) {
                return;
            }
        }
        let mut reader = BufReader::new(response.into_reader());
        let mut line = String::new();
        loop {
            if abandon.load(Ordering::SeqCst) {
                return;
            }
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    send(Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "connection closed before [DONE]".into(),
                    }));
                    return;
                }
                Ok(_) => {
                    let trimmed = line.trim_end_matches(['\n', '\r']);
                    let Some(payload) = sse_data_payload(trimmed) else {
                        continue;
                    };
                    if payload.trim() == "[DONE]" {
                        send(Ok(StreamEvent::Finished));
                        return;
                    }
                    match serde_json::from_str::<serde_json::Value>(payload.trim()) {
                        Ok(value) => {
                            if let Some(text) = delta_content(&value) {
                                if !send(Ok(StreamEvent::Delta(text))) {
                                    return;
                                }
                            }
                            for delta in delta_tool_calls(&value) {
                                if !send(Ok(StreamEvent::ToolCallDelta {
                                    index: delta.index,
                                    id: delta.id,
                                    name: delta.name,
                                    arguments: delta.arguments,
                                })) {
                                    return;
                                }
                            }
                        }
                        Err(err) => {
                            send(Err(Error::Provider {
                                provider: PROVIDER_NAME,
                                message: format!("malformed SSE payload: {err}"),
                            }));
                            return;
                        }
                    }
                }
                Err(err) if is_poll_timeout(&err) => continue,
                Err(err) => {
                    send(Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: format!("stream read failed: {err}"),
                    }));
                    return;
                }
            }
        }
    }
}

fn is_poll_timeout(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// Set on drop so the worker stops reading and closes the connection even
/// when the generate loop exits early (barge-in cancel, shutdown, errors).
struct AbandonOnDrop(Arc<AtomicBool>);

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// `data:` payload of an SSE line; comments, event names, and blanks are None.
fn sse_data_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    let rest = rest.strip_prefix(' ').unwrap_or(rest);
    Some(rest)
}

/// Assistant text from one chat-completions chunk (`choices[0].delta.content`).
fn delta_content(value: &serde_json::Value) -> Option<String> {
    let choice = value.get("choices")?.get(0)?;
    let delta = choice.get("delta")?;
    match delta.get("content") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(text)) if !text.is_empty() => Some(text.clone()),
        Some(_) => None,
    }
}

struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
}

/// Extract the OpenAI-compatible `choices[0].delta.tool_calls` fragments.
/// Fragment joining and JSON validation happen once in the normalized loop.
fn delta_tool_calls(value: &serde_json::Value) -> Vec<ToolCallDelta> {
    let Some(entries) = value
        .pointer("/choices/0/delta/tool_calls")
        .and_then(serde_json::Value::as_array)
    else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let index = entry
                .get("index")?
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())?;
            let function = entry.get("function");
            Some(ToolCallDelta {
                index,
                id: entry
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                name: function
                    .and_then(|f| f.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                arguments: function
                    .and_then(|f| f.get("arguments"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

/// Same message shape as the local engine: system prompt pins the reply
/// language, rolling history keeps the last turns, newest question last.
fn request_body(
    model: &str,
    history: &[HistoryTurn],
    user: &Transcript,
    system_prompt: &str,
) -> serde_json::Value {
    let kept = if history.len() > LLAMA_MAX_HISTORY_TURNS {
        &history[history.len() - LLAMA_MAX_HISTORY_TURNS..]
    } else {
        history
    };
    let mut messages = Vec::with_capacity(2 + kept.len() * 2);
    messages.push(serde_json::json!({
        "role": "system",
        "content": crate::llm::render_system_prompt(system_prompt, &user.language),
    }));
    for turn in kept {
        messages.push(serde_json::json!({"role": "user", "content": turn.user.text}));
        messages.push(serde_json::json!({"role": "assistant", "content": turn.assistant}));
    }
    messages.push(serde_json::json!({"role": "user", "content": user.text}));
    serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
    })
}

/// A failed turn speaks a short fallback instead of hanging. No auto-retry.
fn speak_fallback(
    turn: crate::types::TurnId,
    generation: crate::types::GenerationId,
    cancel: &Cancel,
    on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
) -> Result<()> {
    if cancel.is_shutdown() || cancel.is_stale(generation) {
        return Err(Error::Cancelled);
    }
    on_token(TokenChunk {
        turn,
        generation,
        index: 0,
        text: CLOUD_FALLBACK_TEXT.to_string(),
        is_last: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TurnId;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};

    const TEST_KEY: &str = "sk-test-key";

    fn user(text: &str) -> Transcript {
        Transcript {
            turn: TurnId(0),
            text: text.into(),
            language: "en".into(),
        }
    }

    fn test_llm(endpoint: String) -> OpenAiLlm {
        OpenAiLlm::with_timeouts(
            OpenAiSettings {
                endpoint,
                model: "gpt-test".into(),
                system_prompt: crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
                developer_harness: false,
            },
            Zeroizing::new(TEST_KEY.into()),
            OpenAiTimeouts {
                connect: Duration::from_secs(2),
                read_poll: Duration::from_millis(20),
                idle: Duration::from_millis(400),
            },
        )
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    /// Serve exactly one HTTP request on an ephemeral port. The handler writes
    /// whatever bytes it wants (paced, truncated, or error responses); the
    /// captured request head+body comes back through the join handle.
    fn serve_one<F>(handler: F) -> (String, std::thread::JoinHandle<String>)
    where
        F: FnOnce(&mut TcpStream, &str) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("one connection");
            let mut buf = [0u8; 4096];
            let mut request: Vec<u8> = Vec::new();
            let head_end = loop {
                let n = stream.read(&mut buf).expect("read request head");
                assert!(n > 0, "client closed before sending a request");
                request.extend_from_slice(&buf[..n]);
                if let Some(pos) = find(&request, b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let head = String::from_utf8_lossy(&request[..head_end]).to_string();
            let content_length: usize = head
                .lines()
                .find_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            while request.len() < head_end + content_length {
                let n = stream.read(&mut buf).expect("read request body");
                assert!(n > 0, "client closed before sending the body");
                request.extend_from_slice(&buf[..n]);
            }
            let body = String::from_utf8_lossy(&request[head_end..]).to_string();
            let captured = format!("{head}---BODY---{body}");
            handler(&mut stream, &captured);
            captured
        });
        (
            format!("http://127.0.0.1:{port}/v1/chat/completions"),
            handle,
        )
    }

    /// Bind a mock endpoint that must receive no connection within `wait`.
    /// The handle resolves to `true` when a client connected (i.e. the
    /// no-request expectation was violated). Unlike `serve_one`, this never
    /// blocks past `wait`, so a pre-cancelled turn cannot hang the suite.
    fn serve_expect_no_connection(wait: Duration) -> (String, std::thread::JoinHandle<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let handle = std::thread::spawn(move || {
            let start = Instant::now();
            loop {
                match listener.accept() {
                    Ok(_) => return true,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if start.elapsed() >= wait {
                            return false;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    // Any other accept outcome counts as activity; the
                    // test then fails on `connected` instead of hanging.
                    Err(_) => return true,
                }
            }
        });
        (
            format!("http://127.0.0.1:{port}/v1/chat/completions"),
            handle,
        )
    }

    /// Serve a tool-call response followed by its continuation request. This
    /// uses the production blocking HTTP/SSE path without models or devices.
    fn serve_two<F>(mut handler: F) -> (String, std::thread::JoinHandle<Vec<String>>)
    where
        F: FnMut(usize, &mut TcpStream, &str) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for index in 0..2 {
                let (mut stream, _) = listener.accept().expect("tool loop connection");
                let request = read_captured_request(&mut stream);
                handler(index, &mut stream, &request);
                requests.push(request);
            }
            requests
        });
        (
            format!("http://127.0.0.1:{port}/v1/chat/completions"),
            handle,
        )
    }

    fn read_captured_request(stream: &mut TcpStream) -> String {
        let mut buf = [0u8; 4096];
        let mut request: Vec<u8> = Vec::new();
        let head_end = loop {
            let n = stream.read(&mut buf).expect("read request head");
            assert!(n > 0, "client closed before sending a request");
            request.extend_from_slice(&buf[..n]);
            if let Some(pos) = find(&request, b"\r\n\r\n") {
                break pos + 4;
            }
        };
        let head = String::from_utf8_lossy(&request[..head_end]).to_string();
        let content_length: usize = head
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|value| value.trim().parse::<usize>().unwrap_or(0))
            })
            .unwrap_or(0);
        while request.len() < head_end + content_length {
            let n = stream.read(&mut buf).expect("read request body");
            assert!(n > 0, "client closed before sending the body");
            request.extend_from_slice(&buf[..n]);
        }
        format!(
            "{head}---BODY---{}",
            String::from_utf8_lossy(&request[head_end..])
        )
    }

    fn write_response(
        stream: &mut TcpStream,
        status_line: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> std::io::Result<()> {
        let mut response = format!("{status_line}\r\n");
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        response.push_str(body);
        stream.write_all(response.as_bytes())?;
        stream.flush()
    }

    fn sse_body(events: &[&str], done: bool) -> String {
        let mut body = String::new();
        for event in events {
            body.push_str("data: ");
            body.push_str(event);
            body.push_str("\n\n");
        }
        if done {
            body.push_str("data: [DONE]\n\n");
        }
        body
    }

    fn happy_handler(stream: &mut TcpStream, _request: &str) {
        let events = [
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo there"}}]}"#,
            r#"{"choices":[{"finish_reason":"stop"}],"usage":{}}"#,
        ];
        write_response(
            stream,
            "HTTP/1.1 200 OK",
            &[
                ("Content-Type", "text/event-stream"),
                ("x-request-id", "req-123"),
            ],
            &sse_body(&events, true),
        )
        .unwrap();
    }

    fn collect(llm: &mut OpenAiLlm, cancel: &Cancel) -> (Vec<String>, Result<()>) {
        let mut tokens = Vec::new();
        let result = Llm::generate(llm, &[], &user("hi"), cancel, &mut |chunk| {
            tokens.push(chunk.text);
            Ok(())
        });
        (tokens, result)
    }

    #[test]
    fn streams_deltas_and_finishes_on_done() {
        let (endpoint, server) = serve_one(happy_handler);
        let mut llm = test_llm(endpoint);
        let start = Instant::now();
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("happy stream completes");
        // One-deep lookahead: the final delta carries is_last; no empty marker.
        assert_eq!(tokens, vec!["Hel".to_string(), "lo there".to_string()]);
        let meta = Llm::debug_meta(&llm).expect("openai reports debug meta");
        assert_eq!(meta.provider, "online");
        assert_eq!(meta.model, "gpt-test");
        assert_eq!(meta.request_id, "req-123");
        assert!(meta.endpoint.ends_with("/v1/chat/completions"));
        assert!(start.elapsed() < Duration::from_secs(5));
        server.join().unwrap();
    }

    #[test]
    fn last_token_is_flagged() {
        let (endpoint, server) = serve_one(happy_handler);
        let mut llm = test_llm(endpoint);
        let mut flags = Vec::new();
        Llm::generate(&mut llm, &[], &user("hi"), &Cancel::new(), &mut |chunk| {
            flags.push(chunk.is_last);
            Ok(())
        })
        .unwrap();
        assert_eq!(flags, vec![false, true]);
        server.join().unwrap();
    }

    #[test]
    fn request_matches_the_openai_contract() {
        let (endpoint, server) = serve_one(happy_handler);
        let mut llm = test_llm(endpoint);
        let _ = collect(&mut llm, &Cancel::new());
        let request = server.join().unwrap();
        assert!(
            request.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{request}"
        );
        let lowered = request.to_ascii_lowercase();
        assert!(
            lowered.contains("authorization: bearer sk-test-key"),
            "{request}"
        );
        assert!(lowered.contains("accept: text/event-stream"), "{request}");
        assert!(request.contains("\"model\":\"gpt-test\""), "{request}");
        assert!(request.contains("\"stream\":true"), "{request}");
        assert!(request.contains("smart assistant"), "{request}");
        // serde_json orders map keys alphabetically: messages, model, stream.
        assert!(request.contains("---BODY---{\"messages\":"), "{request}");
    }

    #[test]
    fn harness_continues_a_fragmented_tool_call_without_speaking_the_trace() {
        let first = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"shell","arguments":"{\"argv\":[\"pw"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d\"]}"}}]}}]}"#,
        ];
        let final_reply = r#"{"choices":[{"delta":{"content":"Live lookup is unavailable."}}]}"#;
        let (endpoint, server) = serve_two(move |index, stream, _request| {
            let events: &[&str] = if index == 0 { &first } else { &[final_reply] };
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(events, true),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        llm.settings.developer_harness = true;
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("tool continuation completes");
        assert_eq!(tokens, ["Live lookup is unavailable."]);
        let events = Llm::take_tool_events(&mut llm);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, "call");
        assert_eq!(events[1].kind, "result");
        let requests = server.join().expect("mock server");
        assert!(requests[0].contains("\"tools\""), "{}", requests[0]);
        assert!(requests[0].contains("web_fetch"), "{}", requests[0]);
        assert!(requests[0].contains("shell"), "{}", requests[0]);
        assert!(requests[1].contains("\"role\":\"tool\""), "{}", requests[1]);
        assert!(requests[1].contains("call-1"), "{}", requests[1]);
        assert!(requests[1].contains("exit"), "{}", requests[1]);
    }

    #[test]
    fn harness_rejects_unknown_calls_and_speaks_only_the_fallback() {
        let event = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"rm_everything","arguments":"{}"}}]}}]}"#;
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&[event], true),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        llm.settings.developer_harness = true;
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("rejection falls back");
        assert_eq!(tokens, [CLOUD_FALLBACK_TEXT]);
        assert_eq!(Llm::take_tool_events(&mut llm)[0].kind, "rejected");
        server.join().unwrap();
    }

    #[test]
    fn harness_stops_after_five_calls() {
        let calls: Vec<_> = (0..6)
            .map(|index| serde_json::json!({"index":index,"id":format!("call-{index}"),"function":{"name":"shell","arguments":"{\"argv\":[\"pwd\"]}"}}))
            .collect();
        let event = serde_json::json!({"choices":[{"delta":{"tool_calls":calls}}]}).to_string();
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&[&event], true),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        llm.settings.developer_harness = true;
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("limit is a completed answer");
        assert_eq!(tokens, [TOOL_LIMIT_TEXT]);
        assert_eq!(Llm::take_tool_events(&mut llm)[0].kind, "limit");
        server.join().unwrap();
    }

    #[test]
    fn harness_final_reply_without_a_call_is_spoken() {
        let event = r#"{"choices":[{"delta":{"content":"No tool is needed."}}]}"#;
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&[event], true),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        llm.settings.developer_harness = true;
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("final reply completes");
        assert_eq!(tokens, ["No tool is needed."]);
        assert!(Llm::take_tool_events(&mut llm).is_empty());
        server.join().unwrap();
    }

    #[test]
    fn harness_cancelled_turn_quiesces_without_continuation() {
        // Pre-cancelled: generate must return before dialing, so the mock
        // asserts *no* connection arrives within a short bound instead of
        // blocking on `accept()` like `serve_one` does.
        let (endpoint, server) = serve_expect_no_connection(Duration::from_millis(300));
        let mut llm = test_llm(endpoint);
        llm.settings.developer_harness = true;
        let cancel = Cancel::new();
        cancel.shutdown();
        let (tokens, result) = collect(&mut llm, &cancel);
        assert!(matches!(result, Err(Error::Cancelled)));
        assert!(tokens.is_empty());
        assert!(
            Llm::take_tool_events(&mut llm).is_empty(),
            "no tool runs after shutdown"
        );
        assert!(
            !server.join().expect("mock server"),
            "pre-cancelled turn must not dial"
        );
    }

    #[test]
    fn empty_and_null_delta_chunks_are_skipped() {
        let events = [
            r#"{"choices":[{"delta":{"content":""}}]}"#,
            r#"{"choices":[{"delta":{"content":null}}]}"#,
            r#"{"choices":[{"delta":{"content":"real"}}]}"#,
        ];
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&events, true),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.unwrap();
        assert_eq!(tokens, vec!["real".to_string()]);
        server.join().unwrap();
    }

    #[test]
    fn malformed_payload_speaks_the_fallback() {
        let events = [r#"{"choices":[{"delta":{"content":"Hi"}}]}"#, "{not json"];
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&events, false),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("fallback keeps the loop alive");
        assert_eq!(tokens, vec![CLOUD_FALLBACK_TEXT.to_string()]);
        server.join().unwrap();
    }

    #[test]
    fn mid_stream_disconnect_speaks_the_fallback() {
        let (endpoint, server) = serve_one(|stream, _| {
            // Claim more body than we send, then hang up mid-stream (FIN).
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 9999\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"par";
            stream.write_all(head.as_bytes()).unwrap();
            stream.flush().unwrap();
            let _ = std::net::TcpStream::shutdown(stream, std::net::Shutdown::Both);
        });
        let mut llm = test_llm(endpoint);
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("disconnect becomes a spoken fallback");
        assert_eq!(tokens, vec![CLOUD_FALLBACK_TEXT.to_string()]);
        server.join().unwrap();
    }

    #[test]
    fn clean_eof_without_done_speaks_the_fallback() {
        let events = [r#"{"choices":[{"delta":{"content":"cut"}}]}"#];
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&events, false),
            )
            .unwrap();
        });
        let mut llm = test_llm(endpoint);
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("eof without done becomes a spoken fallback");
        assert_eq!(tokens, vec![CLOUD_FALLBACK_TEXT.to_string()]);
        server.join().unwrap();
    }

    #[test]
    fn idle_timeout_aborts_a_stalled_stream() {
        let (endpoint, server) = serve_one(move |stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "text/event-stream")],
                &sse_body(&[r#"{"choices":[{"delta":{"content":"slow"}}]}"#], false),
            )
            .unwrap();
            std::thread::sleep(Duration::from_millis(900));
        });
        let mut llm = test_llm(endpoint);
        let start = Instant::now();
        let (tokens, result) = collect(&mut llm, &Cancel::new());
        result.expect("idle timeout becomes a spoken fallback");
        assert_eq!(tokens, vec![CLOUD_FALLBACK_TEXT.to_string()]);
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "idle timeout must fire well before the server wakes up"
        );
        server.join().unwrap();
    }

    #[test]
    fn http_error_statuses_become_fallback_with_key_hint_on_401() {
        for status in [
            "HTTP/1.1 401 Unauthorized",
            "HTTP/1.1 500 Internal Server Error",
        ] {
            let (endpoint, server) = serve_one(move |stream, _| {
                write_response(
                    stream,
                    status,
                    &[("Content-Type", "application/json")],
                    r#"{"error":{"message":"bad key"}}"#,
                )
                .unwrap();
            });
            let mut llm = test_llm(endpoint);
            let (tokens, result) = collect(&mut llm, &Cancel::new());
            result.expect("http errors keep the loop alive");
            assert_eq!(tokens, vec![CLOUD_FALLBACK_TEXT.to_string()], "{status}");
            server.join().unwrap();
        }
    }

    #[test]
    fn barge_in_cancel_mid_stream_beats_the_server() {
        let (endpoint, server) = serve_one(move |stream, _| {
            // A real SSE response declares a body length up front; we then
            // pace events well inside it while the client cancels.
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 16384\r\n\r\n";
            if stream.write_all(head.as_bytes()).is_err() {
                return;
            }
            for i in 0..60 {
                let event =
                    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"t{i}\"}}}}]}}\n\n");
                if stream.write_all(event.as_bytes()).is_err() {
                    return;
                }
                let _ = stream.flush();
                std::thread::sleep(Duration::from_millis(30));
            }
        });
        let mut llm = test_llm(endpoint);
        let cancel = Cancel::new();
        let mut seen = Vec::new();
        let start = Instant::now();
        let result = Llm::generate(&mut llm, &[], &user("hi"), &cancel, &mut |chunk| {
            if chunk.index == 0 {
                cancel.cancel_generation();
            }
            seen.push(chunk.text);
            Ok(())
        });
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "cancel must win the race: result={result:?} seen={seen:?}"
        );
        assert!(seen.len() <= 2, "no tokens flow after the cancel: {seen:?}");
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "cancel must beat the 1.8s scripted stream"
        );
        server.join().unwrap();
    }

    #[test]
    fn shutdown_before_generate_is_cancelled_without_connecting() {
        let (endpoint, _server) = serve_one(|_, _| panic!("no request should arrive"));
        let mut llm = test_llm(endpoint);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = Llm::generate(&mut llm, &[], &user("hi"), &cancel, &mut |_| Ok(()))
            .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled));
    }

    #[test]
    fn join_endpoint_tolerates_trailing_slash() {
        assert_eq!(
            join_endpoint("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_endpoint("https://api.openai.com/v1/"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn settings_from_config_defaults_and_overrides() {
        let mut config = crate::AgentConfig::v0();
        config.llm = crate::LlmProvider::Online;
        config.llm_model = "gpt-test".into();
        config.llm_base_url = None;
        assert_eq!(
            OpenAiSettings::from_config(&config).endpoint,
            format!("{DEFAULT_LLM_BASE_URL}/chat/completions")
        );
        assert_eq!(OpenAiSettings::from_config(&config).model, "gpt-test");
        assert_eq!(
            OpenAiSettings::from_config(&config).system_prompt,
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE
        );
        config.llm_base_url = Some("https://api.groq.com/openai/v1".into());
        assert_eq!(
            OpenAiSettings::from_config(&config).endpoint,
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn validate_base_url_accepts_http_https_and_rejects_the_rest() {
        for good in [
            "https://api.openai.com/v1",
            "http://127.0.0.1:8080/v1",
            "https://api.groq.com/openai/v1/",
        ] {
            assert!(validate_base_url(good).is_ok(), "{good}");
        }
        for bad in [
            "",
            "   ",
            "api.example.com/v1",
            "ftp://api.example.com",
            "file:///tmp/socket",
            "https://user:pass@api.example.com/v1",
        ] {
            let message = validate_base_url(bad).err().unwrap_or_default();
            assert!(!message.is_empty(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn api_key_resolution_is_env_only_and_actionable() {
        let missing = resolve_api_key(|_| None).err().unwrap();
        assert!(missing.to_string().contains(API_KEY_ENV), "{missing}");
        assert!(missing.to_string().contains("export SYLLABIX_LLM_API_KEY"));
        // The one-shot prefix form is offered for runs without persistence.
        assert!(
            missing
                .to_string()
                .contains("SYLLABIX_LLM_API_KEY=sk-… syllabix run"),
            "{missing}"
        );
        assert!(missing.to_string().contains("never from syllabix.yaml"));
        let empty = resolve_api_key(|_| Some("   ".into())).err().unwrap();
        assert!(empty.to_string().contains("set but empty"), "{empty}");
        let ok = resolve_api_key(|name| (name == API_KEY_ENV).then(|| "sk-x".into())).unwrap();
        assert_eq!(ok.as_str(), "sk-x");
    }

    #[test]
    fn sse_data_payload_parses_only_data_lines() {
        assert_eq!(sse_data_payload("data: hello"), Some("hello"));
        assert_eq!(sse_data_payload("data:hello"), Some("hello"));
        assert_eq!(sse_data_payload("data: [DONE]"), Some("[DONE]"));
        assert_eq!(sse_data_payload(": keep-alive comment"), None);
        assert_eq!(sse_data_payload("event: delta"), None);
        assert_eq!(sse_data_payload(""), None);
    }

    #[test]
    fn delta_content_handles_chunk_shapes() {
        let parsed = serde_json::from_str::<serde_json::Value>(
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#,
        )
        .unwrap();
        assert_eq!(delta_content(&parsed), Some("hi".to_string()));
        let role_only = serde_json::json!({"choices":[{"delta":{"role":"assistant"}}]});
        assert_eq!(delta_content(&role_only), None);
        let null_content = serde_json::json!({"choices":[{"index":0,"delta":{"content":null}}]});
        assert_eq!(delta_content(&null_content), None);
        let no_choices = serde_json::json!({"usage":{}});
        assert_eq!(delta_content(&no_choices), None);
        let finish_only = serde_json::json!({"choices":[{"finish_reason":"stop"}]});
        assert_eq!(delta_content(&finish_only), None);
    }

    #[test]
    fn request_body_pins_language_and_trims_history_like_local() {
        let history: Vec<HistoryTurn> = (0..10)
            .map(|i| HistoryTurn {
                user: user(&format!("u{i}")),
                assistant: format!("a{i}"),
            })
            .collect();
        let mut french = user("bonjour");
        french.language = "fr".into();
        let body = request_body(
            "gpt-test",
            &history,
            &french,
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE,
        );
        let serialized = body.to_string();
        assert!(serialized.contains("spoken French"), "{serialized}");
        assert!(serialized.contains("\"u2\""), "{serialized}");
        assert!(!serialized.contains("\"u1\""), "{serialized}");
        assert!(serialized.contains("\"bonjour\""), "{serialized}");
        let fresh = request_body(
            "gpt-test",
            &[],
            &user("hi"),
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE,
        );
        assert_eq!(
            fresh["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| m["role"] == "assistant")
                .count(),
            0
        );
    }

    #[test]
    fn default_request_has_no_tool_schemas_but_harness_defines_only_two() {
        let default = request_body(
            "gpt-test",
            &[],
            &user("hi"),
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE,
        );
        assert!(default.get("tools").is_none());
        let tools = tool_definitions();
        let names: Vec<_> = tools
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["web_fetch", "web_search", "shell"]);
    }

    #[test]
    fn fragmented_tool_call_normalizes_once_and_rejects_bad_shapes() {
        let deltas = [
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"shell","arguments":"{\"argv\":[\"git\""}}]}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"status\"]}"}}]}}]}),
        ];
        let fragments: Vec<_> = deltas.iter().flat_map(delta_tool_calls).collect();
        let mut raw = RawToolCall::default();
        for fragment in fragments {
            raw.id = fragment.id.or(raw.id);
            raw.name = fragment.name.or(raw.name);
            raw.arguments
                .push_str(&fragment.arguments.unwrap_or_default());
        }
        let call = normalize_tool_call(raw).expect("fragmented call parses");
        assert_eq!(call.name, "shell");
        assert_eq!(call.arguments["argv"][1], "status");
        assert!(normalize_tool_call(RawToolCall {
            id: Some("x".into()),
            name: Some("unknown".into()),
            arguments: "{}".into()
        })
        .is_err());
        assert!(normalize_tool_call(RawToolCall {
            id: Some("x".into()),
            name: Some("shell".into()),
            arguments: "not-json".into()
        })
        .is_err());
    }

    #[test]
    fn tool_results_continue_as_standard_provider_messages_and_stay_bounded() {
        let call = ToolCall {
            id: "call-1".into(),
            name: "web_fetch".into(),
            arguments: serde_json::json!({"url":"https://example.test"}),
        };
        let mut body = request_body(
            "gpt-test",
            &[],
            &user("hi"),
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE,
        );
        append_tool_call_message(&mut body, std::slice::from_ref(&call));
        let result = ToolResult {
            tool_call_id: call.id.clone(),
            ok: false,
            content: "bounded test result".into(),
        };
        append_tool_result_message(&mut body, &result);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.last().unwrap()["role"], "tool");
        assert_eq!(messages.last().unwrap()["tool_call_id"], "call-1");
        assert!(messages.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("bounded test result"));
        let oversized = "é".repeat(MAX_TOOL_RESULT_BYTES);
        assert!(
            truncate_bytes(&oversized, MAX_TOOL_RESULT_BYTES).len() <= MAX_TOOL_RESULT_BYTES + 3
        );
    }

    #[test]
    fn tool_call_limit_is_fixed() {
        assert_eq!(MAX_TOOL_CALLS_PER_TURN, 5);
    }

    #[test]
    fn launch_timeouts_stay_bounded_for_voice() {
        let timeouts = OpenAiTimeouts::default();
        assert_eq!(timeouts.connect, CONNECT_TIMEOUT);
        assert_eq!(timeouts.read_poll, READ_POLL_TIMEOUT);
        assert_eq!(timeouts.idle, IDLE_TIMEOUT);
        assert!(CONNECT_TIMEOUT <= Duration::from_secs(10));
        assert!(IDLE_TIMEOUT <= Duration::from_secs(30));
    }
}
