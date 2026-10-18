//! Background tasks and agent-initiated follow-up turns.
//!
//! When the developer harness marks a tool call with `background: true`, or
//! uses the `sleep` tool, work continues on a worker thread while the model
//! receives an immediate acknowledgement. Finished tasks queue a follow-up turn
//! for the conversation loop once the user is not speaking and no turn is
//! in progress.

use std::collections::VecDeque;
use std::io::{BufRead, ErrorKind, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::cancel::Cancel;
use crate::executor::{execute_with_permissions, validate_call, ValidatedCall};
use crate::policy::DeveloperPermissions;
use crate::types::{ToolCall, ToolResult, Transcript, TurnId};

/// Maximum concurrent background workers in one live session.
pub const MAX_BACKGROUND_JOBS: usize = 4;

/// Environment variable holding the per-session background socket token (MCP child).
pub const BACKGROUND_TOKEN_ENV: &str = "SYLLABIX_BACKGROUND_TOKEN";

/// Maximum bytes read from one socket line (spawn request or ack).
const MAX_SOCKET_LINE_BYTES: usize = 64 * 1024;

/// Synthetic turn ids start above normal VAD allocation for tests.
const SYNTHETIC_TURN_BASE: u64 = 1_000_000;

static REGISTRY: OnceLock<Mutex<Option<Arc<BackgroundRegistry>>>> = OnceLock::new();

fn registry_slot() -> &'static Mutex<Option<Arc<BackgroundRegistry>>> {
    REGISTRY.get_or_init(|| Mutex::new(None))
}

/// Install the session registry for the current live loop.
pub fn install(registry: Arc<BackgroundRegistry>) {
    *registry_slot().lock().expect("background registry mutex") = Some(registry);
}

/// Remove the session registry when the live loop exits.
pub fn clear() {
    if let Ok(mut slot) = registry_slot().lock() {
        if let Some(registry) = slot.take() {
            registry.shutdown();
        }
    }
}

/// Active registry for in-process tool execution, if any.
pub fn registry() -> Option<Arc<BackgroundRegistry>> {
    registry_slot()
        .lock()
        .expect("background registry mutex")
        .clone()
}

/// Unix socket path for MCP children to report completions, if configured.
pub fn notify_socket_path() -> Option<PathBuf> {
    registry().and_then(|registry| registry.socket_path())
}

/// Session token MCP children must send with socket requests.
pub fn notify_session_token() -> Option<String> {
    registry().map(|registry| registry.session_token().to_string())
}

/// One finished background task waiting for a follow-up turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundCompletion {
    pub label: String,
    pub ok: bool,
    pub content: String,
}

/// Whether a queued completion may start a follow-up turn now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FollowUpGate {
    pub user_speaking: bool,
    pub assistant_active: bool,
    pub mic_muted: bool,
    pub queued: usize,
}

impl FollowUpGate {
    /// True when the pipeline may inject the next completion.
    pub fn may_deliver(&self) -> bool {
        self.queued > 0 && !self.user_speaking && !self.assistant_active && !self.mic_muted
    }
}

/// Host-owned background job queue for one conversation loop.
pub struct BackgroundRegistry {
    cancel: Cancel,
    language: String,
    workspace: PathBuf,
    permissions: DeveloperPermissions,
    session_token: String,
    running: AtomicUsize,
    completions: Arc<Mutex<VecDeque<BackgroundCompletion>>>,
    next_synthetic_turn: Mutex<u64>,
    socket_path: Option<PathBuf>,
    socket_dir: Option<PathBuf>,
    listener: Mutex<Option<UnixListener>>,
    listener_handle: Mutex<Option<JoinHandle<()>>>,
    listener_shutdown: Arc<AtomicBool>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    worker_cancels: Mutex<Vec<Cancel>>,
    shutdown: AtomicBool,
}

impl BackgroundRegistry {
    /// Create a registry for one live loop.
    pub fn new(
        cancel: Cancel,
        language: String,
        workspace: PathBuf,
        permissions: DeveloperPermissions,
    ) -> Arc<Self> {
        let session_token = new_session_token();
        let (socket_dir, socket_path, listener) =
            create_private_socket().unwrap_or((None, None, None));
        let registry = Arc::new(Self {
            cancel,
            language,
            workspace,
            permissions,
            session_token,
            running: AtomicUsize::new(0),
            completions: Arc::new(Mutex::new(VecDeque::new())),
            next_synthetic_turn: Mutex::new(SYNTHETIC_TURN_BASE),
            socket_path,
            socket_dir,
            listener: Mutex::new(listener),
            listener_handle: Mutex::new(None),
            listener_shutdown: Arc::new(AtomicBool::new(false)),
            workers: Mutex::new(Vec::new()),
            worker_cancels: Mutex::new(Vec::new()),
            shutdown: AtomicBool::new(false),
        });
        Self::start_listener(&registry);
        registry
    }

    pub fn socket_path(&self) -> Option<PathBuf> {
        self.socket_path.clone()
    }

    pub fn session_token(&self) -> &str {
        &self.session_token
    }

    fn start_listener(self_arc: &Arc<BackgroundRegistry>) {
        let Some(listener) = self_arc.listener.lock().expect("bg listener").take() else {
            return;
        };
        let host = Arc::clone(self_arc);
        let cancel = self_arc.cancel.clone();
        let shutdown = Arc::clone(&self_arc.listener_shutdown);
        let handle = thread::Builder::new()
            .name("syllabix-bg-listen".into())
            .spawn(move || listen_loop(listener, host, cancel, shutdown))
            .expect("background listener thread");
        *self_arc.listener_handle.lock().expect("bg listener handle") = Some(handle);
    }

    /// Queue depth of finished tasks not yet spoken.
    pub fn queued_completions(&self) -> usize {
        self.completions.lock().expect("bg completions").len()
    }

    pub fn running_jobs(&self) -> usize {
        self.running.load(Ordering::SeqCst)
    }

    /// Try to start `call` in the background. Returns an immediate tool result.
    pub fn try_spawn(
        self: &Arc<Self>,
        call: &ToolCall,
        workspace: &Path,
        session: &DeveloperPermissions,
    ) -> ToolResult {
        let (validated, background) = match validate_call(call, workspace) {
            Ok(pair) => pair,
            Err(message) => {
                return ToolResult {
                    tool_call_id: call.id.clone(),
                    ok: false,
                    content: message,
                };
            }
        };
        if !background {
            return ToolResult {
                tool_call_id: call.id.clone(),
                ok: false,
                content: "background flag must be true".into(),
            };
        }
        self.try_spawn_validated(&call.id, validated, workspace, session)
    }

    pub fn try_spawn_validated(
        self: &Arc<Self>,
        call_id: &str,
        validated: ValidatedCall,
        workspace: &Path,
        session: &DeveloperPermissions,
    ) -> ToolResult {
        if self.shutdown.load(Ordering::SeqCst) || self.cancel.is_shutdown() {
            return ToolResult {
                tool_call_id: call_id.into(),
                ok: false,
                content: "background tasks are shutting down".into(),
            };
        }
        if !self.reserve_job_slot() {
            return ToolResult {
                tool_call_id: call_id.into(),
                ok: false,
                content: format!(
                    "background job limit reached ({MAX_BACKGROUND_JOBS} at once); wait for a running task to finish"
                ),
            };
        }
        let label = task_label(&validated);
        let ack_label = label.clone();
        let workspace = workspace.to_path_buf();
        let permissions = session.clone();
        let registry = Arc::clone(self);
        let call_id = call_id.to_string();
        let worker_cancel = Cancel::new();
        self.worker_cancels
            .lock()
            .expect("bg worker cancels")
            .push(worker_cancel.clone());
        let handle = thread::Builder::new()
            .name("syllabix-bg-worker".into())
            .spawn(move || {
                let result =
                    run_validated_for_mcp(&validated, &workspace, &permissions, &worker_cancel);
                if !registry.shutdown.load(Ordering::SeqCst) {
                    registry.push_completion(BackgroundCompletion {
                        label,
                        ok: result.is_ok(),
                        content: result.unwrap_or_else(|message| message),
                    });
                }
                registry.running.fetch_sub(1, Ordering::SeqCst);
            })
            .map_err(|err| format!("could not start background worker: {err}"));
        match handle {
            Ok(join) => {
                self.workers.lock().expect("bg workers").push(join);
                ToolResult {
                    tool_call_id: call_id,
                    ok: true,
                    content: format!(
                        "Started in the background ({ack_label}). Syllabix will speak when it finishes."
                    ),
                }
            }
            Err(message) => {
                self.running.fetch_sub(1, Ordering::SeqCst);
                ToolResult {
                    tool_call_id: call_id,
                    ok: false,
                    content: message,
                }
            }
        }
    }

    /// Handle one MCP socket request (spawn or completion).
    pub fn handle_socket_line(self: &Arc<Self>, line: &str) -> Option<String> {
        let request: ListenerRequest = match serde_json::from_str(line.trim()) {
            Ok(request) => request,
            Err(_) => return None,
        };
        if request.token != self.session_token {
            return Some(
                serde_json::to_string(&ListenerResponse::Error {
                    message: "unauthorized".into(),
                })
                .unwrap_or_default(),
            );
        }
        match request.op.as_str() {
            "spawn" => {
                let name = request.name.unwrap_or_default();
                let arguments = request
                    .arguments
                    .unwrap_or(Value::Object(Default::default()));
                let call_id = request.call_id.unwrap_or_else(|| "mcp-background".into());
                let call = ToolCall {
                    id: call_id,
                    name,
                    arguments,
                };
                let result = self.try_spawn(&call, &self.workspace, &self.permissions);
                Some(
                    serde_json::to_string(&ListenerResponse::SpawnAck {
                        ok: result.ok,
                        message: result.content,
                    })
                    .unwrap_or_default(),
                )
            }
            _ => Some(
                serde_json::to_string(&ListenerResponse::Error {
                    message: "unknown op".into(),
                })
                .unwrap_or_default(),
            ),
        }
    }

    fn reserve_job_slot(&self) -> bool {
        let mut current = self.running.load(Ordering::SeqCst);
        while current < MAX_BACKGROUND_JOBS {
            match self.running.compare_exchange_weak(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
        false
    }

    /// Push a completion reported by an MCP child or tests.
    pub fn push_completion(&self, completion: BackgroundCompletion) {
        self.completions
            .lock()
            .expect("bg completions")
            .push_back(completion);
    }

    /// Take the next completion when `gate_ok` returns true. Re-checks before delivery.
    pub fn take_next_follow_up<F>(&self, mut gate_ok: F) -> Option<Transcript>
    where
        F: FnMut() -> bool,
    {
        if !gate_ok() {
            return None;
        }
        let completion = {
            let mut queue = self.completions.lock().expect("bg completions");
            if queue.is_empty() {
                return None;
            }
            if !gate_ok() {
                return None;
            }
            queue.pop_front()?
        };
        if !gate_ok() {
            self.completions
                .lock()
                .expect("bg completions")
                .push_front(completion);
            return None;
        }
        let turn = self.allocate_turn();
        Some(Transcript::from_background(
            turn,
            self.language.clone(),
            completion.label,
            completion.ok,
            completion.content,
        ))
    }

    fn allocate_turn(&self) -> TurnId {
        let mut slot = self.next_synthetic_turn.lock().expect("bg turn id");
        let id = TurnId(*slot);
        *slot += 1;
        id
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.listener_shutdown.store(true, Ordering::SeqCst);
        self.cancel.shutdown();
        for worker in self
            .worker_cancels
            .lock()
            .expect("bg worker cancels")
            .drain(..)
        {
            worker.shutdown();
        }
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
        }
        if let Some(dir) = &self.socket_dir {
            let _ = std::fs::remove_dir_all(dir);
        }
        let workers = self
            .workers
            .lock()
            .expect("bg workers")
            .drain(..)
            .collect::<Vec<_>>();
        for worker in workers {
            let _ = worker.join();
        }
        if let Some(handle) = self
            .listener_handle
            .lock()
            .expect("bg listener handle")
            .take()
        {
            let _ = handle.join();
        }
    }
}

fn listen_loop(
    listener: UnixListener,
    host: Arc<BackgroundRegistry>,
    cancel: Cancel,
    shutdown: Arc<AtomicBool>,
) {
    listener
        .set_nonblocking(true)
        .expect("background listener nonblocking");
    while !cancel.is_shutdown() && !shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = handle_client(stream, &host);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
}

fn read_bounded_line(reader: &mut impl BufRead) -> std::io::Result<String> {
    let mut line = String::new();
    let mut limited = reader.take((MAX_SOCKET_LINE_BYTES + 1) as u64);
    limited.read_line(&mut line)?;
    if line.len() > MAX_SOCKET_LINE_BYTES {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "background socket line too long",
        ));
    }
    Ok(line)
}

fn handle_client(stream: UnixStream, host: &Arc<BackgroundRegistry>) -> std::io::Result<()> {
    let mut reader = std::io::BufReader::new(stream);
    let line = read_bounded_line(&mut reader)?;
    if line.trim().is_empty() {
        return Ok(());
    }
    if let Some(response) = host.handle_socket_line(&line) {
        let mut stream = reader.into_inner();
        stream.write_all(response.as_bytes())?;
        stream.write_all(b"\n")?;
    }
    Ok(())
}

#[derive(Deserialize)]
struct ListenerRequest {
    op: String,
    token: String,
    call_id: Option<String>,
    name: Option<String>,
    arguments: Option<Value>,
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum ListenerResponse {
    SpawnAck { ok: bool, message: String },
    Error { message: String },
}

/// Read the background socket token from the MCP child environment.
pub fn token_from_env() -> Option<String> {
    std::env::var(BACKGROUND_TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty())
}

/// Ask the parent live loop to start one background job (MCP child).
pub fn request_spawn(
    socket_path: &Path,
    token: &str,
    call_id: &str,
    name: &str,
    arguments: &Value,
) -> Result<String, String> {
    let mut stream = UnixStream::connect(socket_path).map_err(|err| err.to_string())?;
    let payload = json!({
        "op": "spawn",
        "token": token,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    });
    let line = serde_json::to_string(&payload).map_err(|err| err.to_string())?;
    stream
        .write_all(line.as_bytes())
        .map_err(|err| err.to_string())?;
    stream.write_all(b"\n").map_err(|err| err.to_string())?;
    let mut reader = std::io::BufReader::new(stream);
    let response = read_bounded_line(&mut reader).map_err(|err| err.to_string())?;
    let value: Value = serde_json::from_str(response.trim()).map_err(|err| err.to_string())?;
    if value.get("op").and_then(Value::as_str) == Some("spawn_ack") {
        if value.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("started")
                .into())
        } else {
            Err(value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("spawn failed")
                .into())
        }
    } else {
        Err(value
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("spawn failed")
            .into())
    }
}

type PrivateSocketParts = (Option<PathBuf>, Option<PathBuf>, Option<UnixListener>);

fn create_private_socket() -> Result<PrivateSocketParts, std::io::Error> {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(
        "syllabix-bg-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir)?;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    let socket_path = dir.join("notify.sock");
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;
    std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))?;
    Ok((Some(dir), Some(socket_path), Some(listener)))
}

fn new_session_token() -> String {
    format!(
        "{:x}{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

/// Execute one validated call on a background worker thread (MCP child).
pub fn run_validated_for_mcp(
    validated: &ValidatedCall,
    workspace: &Path,
    session: &DeveloperPermissions,
    cancel: &Cancel,
) -> Result<String, String> {
    match validated {
        ValidatedCall::Shell(request) => {
            let mut args = serde_json::Map::new();
            args.insert(
                "command".into(),
                serde_json::Value::String(request.command.clone()),
            );
            if let Some(permission) = request.permission {
                args.insert(
                    "permission".into(),
                    serde_json::Value::String(permission.as_str().into()),
                );
            }
            let workdir = request
                .cwd
                .strip_prefix(workspace)
                .unwrap_or(&request.cwd)
                .to_string_lossy()
                .to_string();
            if !workdir.is_empty() && workdir != "." {
                args.insert("workdir".into(), serde_json::Value::String(workdir));
            }
            let call = ToolCall {
                id: "background".into(),
                name: "shell".into(),
                arguments: serde_json::Value::Object(args),
            };
            let result = execute_with_permissions(&call, workspace, session, cancel);
            if result.ok {
                Ok(result.content)
            } else {
                Err(result.content)
            }
        }
        ValidatedCall::WebFetch { url } => {
            let call = ToolCall {
                id: "background".into(),
                name: "web_fetch".into(),
                arguments: serde_json::json!({ "url": url }),
            };
            let result = execute_with_permissions(&call, workspace, session, cancel);
            if result.ok {
                Ok(result.content)
            } else {
                Err(result.content)
            }
        }
        ValidatedCall::WebSearch { query, count } => {
            let call = ToolCall {
                id: "background".into(),
                name: "web_search".into(),
                arguments: serde_json::json!({ "query": query, "count": count }),
            };
            let result = execute_with_permissions(&call, workspace, session, cancel);
            if result.ok {
                Ok(result.content)
            } else {
                Err(result.content)
            }
        }
        ValidatedCall::Sleep { seconds } => sleep_worker(*seconds, cancel),
    }
}

fn sleep_worker(seconds: u32, cancel: &Cancel) -> Result<String, String> {
    let deadline = Instant::now() + Duration::from_secs(seconds as u64);
    while Instant::now() < deadline {
        if cancel.is_shutdown() {
            return Err("sleep cancelled".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
    Ok(format!("slept {seconds} seconds"))
}

fn task_label(validated: &ValidatedCall) -> String {
    match validated {
        ValidatedCall::Shell(_) => "shell".into(),
        ValidatedCall::WebFetch { .. } => "web_fetch".into(),
        ValidatedCall::WebSearch { .. } => "web_search".into(),
        ValidatedCall::Sleep { seconds } => format!("sleep {seconds}s"),
    }
}

#[cfg(test)]
pub(crate) fn background_integration_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().expect("background integration lock")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> Arc<BackgroundRegistry> {
        BackgroundRegistry::new(
            Cancel::new(),
            "en".into(),
            std::env::temp_dir(),
            DeveloperPermissions::default_session(),
        )
    }

    #[test]
    fn gate_queues_while_user_speaks() {
        let gate = FollowUpGate {
            user_speaking: true,
            assistant_active: false,
            mic_muted: false,
            queued: 1,
        };
        assert!(!gate.may_deliver());
    }

    #[test]
    fn gate_queues_while_assistant_active() {
        let gate = FollowUpGate {
            user_speaking: false,
            assistant_active: true,
            mic_muted: false,
            queued: 1,
        };
        assert!(!gate.may_deliver());
    }

    #[test]
    fn gate_respects_mic_mute() {
        let gate = FollowUpGate {
            user_speaking: false,
            assistant_active: false,
            mic_muted: true,
            queued: 1,
        };
        assert!(!gate.may_deliver());
    }

    #[test]
    fn gate_delivers_when_idle() {
        let gate = FollowUpGate {
            user_speaking: false,
            assistant_active: false,
            mic_muted: false,
            queued: 1,
        };
        assert!(gate.may_deliver());
    }

    #[test]
    fn cap_rejects_fifth_job() {
        let registry = test_registry();
        for _ in 0..MAX_BACKGROUND_JOBS {
            let call = ToolCall {
                id: "sleep".into(),
                name: "sleep".into(),
                arguments: serde_json::json!({ "seconds": 120 }),
            };
            let result = registry.try_spawn(
                &call,
                &std::env::temp_dir(),
                &DeveloperPermissions::default_session(),
            );
            assert!(result.ok, "{}", result.content);
        }
        let call = ToolCall {
            id: "overflow".into(),
            name: "sleep".into(),
            arguments: serde_json::json!({ "seconds": 120 }),
        };
        let result = registry.try_spawn(
            &call,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
        );
        assert!(!result.ok);
        assert!(result.content.contains("limit"));
        registry.shutdown();
        clear();
    }

    #[test]
    fn cap_atomic_under_contention() {
        let registry = test_registry();
        install(Arc::clone(&registry));
        let handles: Vec<_> = (0..12)
            .map(|i| {
                let registry = Arc::clone(&registry);
                thread::spawn(move || {
                    let call = ToolCall {
                        id: format!("job-{i}"),
                        name: "sleep".into(),
                        arguments: serde_json::json!({ "seconds": 60 }),
                    };
                    registry.try_spawn(
                        &call,
                        &std::env::temp_dir(),
                        &DeveloperPermissions::default_session(),
                    )
                })
            })
            .collect();
        let started = handles
            .into_iter()
            .map(|handle| handle.join().expect("join"))
            .filter(|result| result.ok)
            .count();
        assert_eq!(started, MAX_BACKGROUND_JOBS);
        registry.shutdown();
        clear();
    }

    #[test]
    fn cancel_on_shutdown_drops_completions() {
        let cancel = Cancel::new();
        let registry = BackgroundRegistry::new(
            cancel.clone(),
            "en".into(),
            std::env::temp_dir(),
            DeveloperPermissions::default_session(),
        );
        let call = ToolCall {
            id: "1".into(),
            name: "sleep".into(),
            arguments: serde_json::json!({ "seconds": 2 }),
        };
        let result = registry.try_spawn(
            &call,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
        );
        assert!(result.ok);
        cancel.shutdown();
        registry.shutdown();
        thread::sleep(Duration::from_millis(50));
        assert_eq!(registry.queued_completions(), 0);
        clear();
    }

    #[test]
    fn take_next_follow_up_respects_gate() {
        let registry = test_registry();
        registry.push_completion(BackgroundCompletion {
            label: "sleep 1s".into(),
            ok: true,
            content: "done".into(),
        });
        let blocked = registry.take_next_follow_up(|| false);
        assert!(blocked.is_none());
        let transcript = registry.take_next_follow_up(|| true).expect("deliver");
        assert!(transcript.background.is_some());
        assert_eq!(transcript.text, "done");
        registry.shutdown();
        clear();
    }

    #[test]
    fn handle_socket_line_rejects_bad_token() {
        let registry = test_registry();
        let line = r#"{"op":"spawn","token":"wrong","name":"sleep","arguments":{"seconds":1}}"#;
        let response = registry.handle_socket_line(line).expect("error");
        assert!(response.contains("unauthorized"));
        registry.shutdown();
        clear();
    }

    #[test]
    fn handle_socket_line_rejects_removed_complete_op() {
        let registry = test_registry();
        let token = registry.session_token();
        let line = format!(r#"{{"op":"complete","token":"{token}"}}"#, token = token);
        let response = registry.handle_socket_line(&line).expect("error response");
        assert!(response.contains("unknown"));
        registry.shutdown();
        clear();
    }

    #[test]
    fn handle_socket_line_spawn_starts_job() {
        let registry = test_registry();
        install(Arc::clone(&registry));
        let token = registry.session_token();
        let line = format!(
            r#"{{"op":"spawn","token":"{token}","call_id":"x","name":"sleep","arguments":{{"seconds":1}}}}"#,
            token = token
        );
        let response = registry.handle_socket_line(&line).expect("ack");
        assert!(response.contains("spawn_ack"));
        registry.shutdown();
        clear();
    }

    #[test]
    fn handle_socket_line_ignores_malformed_json() {
        let registry = test_registry();
        assert!(registry.handle_socket_line("not-json").is_none());
        registry.shutdown();
        clear();
    }

    #[test]
    fn token_from_env_reads_non_empty_value() {
        // SAFETY: test-local env mutation; other tests do not rely on this variable.
        unsafe { std::env::set_var(BACKGROUND_TOKEN_ENV, "test-token-value") };
        assert_eq!(token_from_env().as_deref(), Some("test-token-value"));
        unsafe { std::env::remove_var(BACKGROUND_TOKEN_ENV) };
    }

    #[test]
    fn registry_install_exposes_socket_and_token() {
        clear();
        let installed = test_registry();
        install(Arc::clone(&installed));
        assert!(registry().is_some());
        assert_eq!(
            notify_socket_path(),
            installed.socket_path(),
            "socket path should match installed registry"
        );
        assert_eq!(
            notify_session_token().as_deref(),
            Some(installed.session_token())
        );
        assert_eq!(installed.running_jobs(), 0);
        installed.shutdown();
        clear();
        assert!(registry().is_none());
        assert!(notify_socket_path().is_none());
    }

    #[test]
    fn try_spawn_rejects_non_background_tools() {
        let registry = test_registry();
        let call = ToolCall {
            id: "wf".into(),
            name: "web_fetch".into(),
            arguments: serde_json::json!({
                "url": "https://example.com/",
                "background": false
            }),
        };
        let result = registry.try_spawn(
            &call,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
        );
        assert!(!result.ok);
        assert!(result.content.contains("background flag"));
        registry.shutdown();
        clear();
    }

    #[test]
    fn try_spawn_rejects_invalid_tool() {
        let registry = test_registry();
        let call = ToolCall {
            id: "bad".into(),
            name: "not_a_tool".into(),
            arguments: serde_json::json!({ "background": true }),
        };
        let result = registry.try_spawn(
            &call,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
        );
        assert!(!result.ok);
        registry.shutdown();
        clear();
    }

    #[test]
    fn try_spawn_after_shutdown_returns_error() {
        let registry = test_registry();
        registry.shutdown();
        let call = ToolCall {
            id: "late".into(),
            name: "sleep".into(),
            arguments: serde_json::json!({ "seconds": 1 }),
        };
        let result = registry.try_spawn(
            &call,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
        );
        assert!(!result.ok);
        assert!(
            result.content.contains("shutting down"),
            "{}",
            result.content
        );
        clear();
    }

    #[test]
    fn take_next_follow_up_requeues_when_gate_closes() {
        let registry = test_registry();
        registry.push_completion(BackgroundCompletion {
            label: "task".into(),
            ok: true,
            content: "body".into(),
        });
        let mut checks = 0;
        let blocked = registry.take_next_follow_up(|| {
            checks += 1;
            checks <= 2
        });
        assert!(blocked.is_none());
        assert_eq!(registry.queued_completions(), 1);
        let transcript = registry.take_next_follow_up(|| true).expect("deliver");
        assert_eq!(transcript.text, "body");
        registry.shutdown();
        clear();
    }

    #[test]
    fn sleep_worker_finishes_and_can_cancel() {
        let cancel = Cancel::new();
        let validated = validate_call(
            &ToolCall {
                id: "s".into(),
                name: "sleep".into(),
                arguments: serde_json::json!({ "seconds": 1 }),
            },
            &std::env::temp_dir(),
        )
        .expect("validate")
        .0;
        let ok = run_validated_for_mcp(
            &validated,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
            &Cancel::new(),
        )
        .expect("sleep");
        assert!(ok.contains("slept 1"));
        cancel.shutdown();
        let validated = validate_call(
            &ToolCall {
                id: "s2".into(),
                name: "sleep".into(),
                arguments: serde_json::json!({ "seconds": 30 }),
            },
            &std::env::temp_dir(),
        )
        .expect("validate")
        .0;
        let err = run_validated_for_mcp(
            &validated,
            &std::env::temp_dir(),
            &DeveloperPermissions::default_session(),
            &cancel,
        )
        .expect_err("cancelled");
        assert!(err.contains("cancelled"));
    }

    #[test]
    fn request_spawn_roundtrip_via_listener() {
        let _lock = background_integration_lock();
        clear();
        let registry = test_registry();
        let path = registry.socket_path().expect("socket");
        let token = registry.session_token();
        let ack = request_spawn(
            &path,
            token,
            "roundtrip",
            "sleep",
            &serde_json::json!({ "seconds": 1 }),
        )
        .expect("spawn ack");
        assert!(ack.contains("background"));
        thread::sleep(Duration::from_millis(1_200));
        assert!(registry.queued_completions() >= 1);
        registry.shutdown();
        clear();
    }

    #[test]
    fn request_spawn_surfaces_spawn_errors() {
        clear();
        let registry = test_registry();
        let path = registry.socket_path().expect("socket");
        let token = registry.session_token();
        let err = request_spawn(
            &path,
            token,
            "bad",
            "not_a_tool",
            &serde_json::json!({ "background": true }),
        )
        .expect_err("invalid tool");
        assert!(!err.is_empty());
        registry.shutdown();
        clear();
    }

    #[test]
    fn private_socket_uses_restrictive_permissions() {
        let registry = test_registry();
        let socket = registry.socket_path().expect("socket");
        let dir = socket.parent().expect("socket parent");
        assert_eq!(
            std::fs::metadata(dir).expect("dir").permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(&socket)
                .expect("socket")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        registry.shutdown();
        clear();
    }

    #[test]
    fn socket_listener_rejects_oversized_request() {
        use std::io::Write;
        use std::os::unix::net::UnixStream;

        let registry = test_registry();
        let path = registry.socket_path().expect("socket");
        let token = registry.session_token();
        let pad = "x".repeat(MAX_SOCKET_LINE_BYTES);
        let mut stream = UnixStream::connect(path).expect("connect");
        let payload = format!(
            r#"{{"op":"spawn","token":"{token}","pad":"{pad}"}}"#,
            token = token,
            pad = pad
        );
        stream.write_all(payload.as_bytes()).expect("write");
        stream.write_all(b"\n").expect("newline");
        stream.shutdown(std::net::Shutdown::Write).ok();
        thread::sleep(Duration::from_millis(30));
        registry.shutdown();
        clear();
    }
}
