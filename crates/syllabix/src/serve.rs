//! `syllabix serve`: localhost HTTP worker for speech weights.
//!
//! This slice is listen + auth + `GET /health` only. Inference routes land in
//! follow-up PRs. Bind defaults to loopback. `SYLLABIX_SERVE_TOKEN` is required
//! unless the bind is loopback and the variable is unset (dev only).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::json;
use syllabix_core::{Error, Result};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::cli::ServeCommand;

/// Default STT listen address (loopback).
pub const DEFAULT_STT_BIND: &str = "127.0.0.1:8091";
/// Default TTS listen address (loopback).
pub const DEFAULT_TTS_BIND: &str = "127.0.0.1:8092";
/// Default yaml STT id advertised on `/health`.
pub const DEFAULT_STT_MODEL: &str = "whisper-small";
/// Default yaml TTS id advertised on `/health`.
pub const DEFAULT_TTS_MODEL: &str = "pocket-tts";
/// Bearer token for the worker. Never yaml.
pub const SERVE_TOKEN_ENV: &str = "SYLLABIX_SERVE_TOKEN";

/// STT or TTS worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeKind {
    Stt,
    Tts,
}

/// Listen address, advertised model, and optional bearer token.
pub struct ServeState {
    /// Worker kind (STT or TTS).
    pub kind: ServeKind,
    /// Yaml model id returned by `/health`.
    pub model: String,
    /// Required bearer token. `None` is loopback-dev only.
    pub token: Option<String>,
    job: Mutex<()>,
}

impl ServeState {
    /// Build state for a listen loop. Does not load weights.
    pub fn new(kind: ServeKind, model: String, token: Option<String>) -> Self {
        Self {
            kind,
            model,
            token,
            job: Mutex::new(()),
        }
    }

    /// True while another request holds the job mutex.
    pub fn busy(&self) -> bool {
        self.job.try_lock().is_err()
    }

    #[cfg(test)]
    fn lock_job(&self) -> std::sync::MutexGuard<'_, ()> {
        self.job.lock().unwrap_or_else(|err| err.into_inner())
    }
}

/// Parse `--bind` (`host:port`).
pub fn parse_bind(bind: &str) -> Result<SocketAddr> {
    bind.parse().map_err(|err| Error::Config {
        field: "serve.bind".into(),
        message: format!("invalid listen address {bind:?}: {err}"),
    })
}

/// Loopback IPv4/IPv6 (the default bind).
pub fn is_loopback(addr: SocketAddr) -> bool {
    addr.ip().is_loopback()
}

/// Read `SYLLABIX_SERVE_TOKEN`. Empty is a config error. Unset is `None`.
pub fn read_token(getenv: impl Fn(&str) -> Option<String>) -> Result<Option<String>> {
    match getenv(SERVE_TOKEN_ENV) {
        None => Ok(None),
        Some(token) if token.trim().is_empty() => Err(Error::Config {
            field: SERVE_TOKEN_ENV.into(),
            message: "is set but empty. Export a bearer token, or unset it for loopback-only dev."
                .into(),
        }),
        Some(token) => Ok(Some(token.trim().to_string())),
    }
}

/// Non-loopback binds require a token. Loopback may omit it (dev).
pub fn require_token(addr: SocketAddr, token: Option<&str>) -> Result<()> {
    if is_loopback(addr) || token.is_some() {
        return Ok(());
    }
    Err(Error::Config {
        field: SERVE_TOKEN_ENV.into(),
        message: format!(
            "is required when serving on {addr} (not loopback). \
             Export {SERVE_TOKEN_ENV} or bind 127.0.0.1 / ::1."
        ),
    })
}

/// Resolve bind + token without listening.
pub fn prepare(
    bind: &str,
    getenv: impl Fn(&str) -> Option<String>,
) -> Result<(SocketAddr, Option<String>)> {
    let addr = parse_bind(bind)?;
    let token = read_token(getenv)?;
    require_token(addr, token.as_deref())?;
    Ok((addr, token))
}

/// Dispatch `syllabix serve stt|tts`.
pub fn execute(command: ServeCommand) -> Result<()> {
    let (kind, bind, model) = match command {
        ServeCommand::Stt { bind, model } => (ServeKind::Stt, bind, model),
        ServeCommand::Tts { bind, model } => (ServeKind::Tts, bind, model),
    };
    let (addr, token) = prepare(&bind, |key| std::env::var(key).ok())?;
    let server = listen(addr)?;
    let bound = bound_addr(&server)?;
    let state = ServeState::new(kind, model, token);
    tracing::info!(
        kind = ?state.kind,
        model = %state.model,
        %bound,
        auth = state.token.is_some(),
        "serving /health"
    );
    eprintln!(
        "syllabix serve {:?}: {} on {bound} (GET /health)",
        state.kind, state.model
    );
    run_loop(&server, &state, None)
}

fn bound_addr(server: &Server) -> Result<SocketAddr> {
    server.server_addr().to_ip().ok_or_else(|| Error::Config {
        field: "serve.bind".into(),
        message: "listener is not a TCP address".into(),
    })
}

/// Bind `tiny_http` on `addr` (`:0` picks an ephemeral port).
pub fn listen(addr: SocketAddr) -> Result<Server> {
    Server::http(addr).map_err(|err| Error::Config {
        field: "serve.bind".into(),
        message: format!("could not listen on {addr}: {err}"),
    })
}

/// Serve until `stop` is set, or forever when `stop` is `None`.
pub fn run_loop(server: &Server, state: &ServeState, stop: Option<&AtomicBool>) -> Result<()> {
    loop {
        if stop.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            return Ok(());
        }
        let request = match stop {
            Some(_) => match server.recv_timeout(Duration::from_millis(50)) {
                Ok(Some(request)) => request,
                Ok(None) => continue,
                Err(err) => {
                    return Err(Error::Io(err));
                }
            },
            None => match server.recv() {
                Ok(request) => request,
                Err(err) => return Err(Error::Io(err)),
            },
        };
        handle_request(request, state);
    }
}

/// Route one HTTP request. Always responds (or logs a disconnect).
pub fn handle_request(request: Request, state: &ServeState) {
    let method = request.method().clone();
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or(url.as_str());
    if let Err(status) = authorize(&request, state.token.as_deref()) {
        let _ = request.respond(json_status(status, error_message(status)));
        return;
    }
    let response = if method == Method::Get && path == "/health" {
        health_response(state)
    } else {
        json_status(404, "not found")
    };
    if let Err(err) = request.respond(response) {
        tracing::debug!("serve response failed: {err}");
    }
}

fn authorize(request: &Request, token: Option<&str>) -> std::result::Result<(), u16> {
    let Some(expected) = token else {
        return Ok(());
    };
    match bearer_from_headers(request.headers()) {
        Some(got) if got == expected => Ok(()),
        Some(_) => Err(401),
        None => Err(401),
    }
}

fn bearer_from_headers(headers: &[Header]) -> Option<String> {
    let value = headers
        .iter()
        .find(|header| header.field.equiv("Authorization"))?;
    let raw = value.value.as_str().trim();
    let token = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn health_response(state: &ServeState) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = json!({
        "model": state.model,
        "busy": state.busy(),
        "kind": match state.kind {
            ServeKind::Stt => "stt",
            ServeKind::Tts => "tts",
        },
    })
    .to_string();
    json_response(200, body)
}

fn error_message(status: u16) -> &'static str {
    match status {
        401 => "unauthorized",
        503 => "busy",
        _ => "not found",
    }
}

fn json_status(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(
        status,
        json!({ "error": { "message": message } }).to_string(),
    )
}

fn json_response(status: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("Content-Type header");
    Response::from_string(body)
        .with_status_code(StatusCode(status))
        .with_header(header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    fn getenv_none(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn parse_bind_rejects_junk() {
        let err = parse_bind("not-an-addr").expect_err("junk");
        assert!(matches!(err, Error::Config { .. }));
        let addr = parse_bind(DEFAULT_STT_BIND).expect("default");
        assert!(is_loopback(addr));
        assert!(!is_loopback("0.0.0.0:8091".parse().expect("any")));
        assert!(is_loopback("[::1]:8092".parse().expect("v6")));
    }

    #[test]
    fn token_rules() {
        assert!(read_token(getenv_none).expect("unset").is_none());
        let err = read_token(|_| Some(String::new())).expect_err("empty");
        assert!(matches!(err, Error::Config { .. }));
        assert_eq!(
            read_token(|_| Some(String::from("secret")))
                .expect("set")
                .as_deref(),
            Some("secret")
        );
        assert_eq!(
            read_token(|_| Some(String::from(" secret \n")))
                .expect("trim")
                .as_deref(),
            Some("secret")
        );
        let err = read_token(|_| Some(String::from(" \t "))).expect_err("whitespace");
        assert!(matches!(err, Error::Config { .. }));

        let loopback: SocketAddr = DEFAULT_STT_BIND.parse().unwrap();
        require_token(loopback, None).expect("loopback dev");
        require_token(loopback, Some("t")).expect("loopback with token");

        let public: SocketAddr = "0.0.0.0:8091".parse().unwrap();
        require_token(public, None).expect_err("public needs token");
        require_token(public, Some("t")).expect("public with token");
    }

    #[test]
    fn prepare_loopback_without_token() {
        let (addr, token) = prepare("127.0.0.1:0", getenv_none).expect("prepare");
        assert!(is_loopback(addr));
        assert!(token.is_none());
    }

    #[test]
    fn prepare_public_without_token_fails() {
        let err = prepare("0.0.0.0:8091", getenv_none).expect_err("token");
        assert!(matches!(err, Error::Config { .. }));
        let (addr, token) = prepare("0.0.0.0:8091", |_| Some(String::from("tok"))).expect("ok");
        assert!(!is_loopback(addr));
        assert_eq!(token.as_deref(), Some("tok"));
    }

    #[test]
    fn lock_job_marks_busy() {
        let state = ServeState::new(ServeKind::Stt, "whisper-small".into(), None);
        assert!(!state.busy());
        let _held = state.lock_job();
        assert!(state.busy());
    }

    #[cfg(target_os = "linux")]
    fn spawn_server(state: ServeState) -> (SocketAddr, Arc<AtomicBool>, thread::JoinHandle<()>) {
        let server = listen("127.0.0.1:0".parse().unwrap()).expect("listen");
        let addr = bound_addr(&server).expect("tcp");
        let stop = Arc::new(AtomicBool::new(false));
        let stop_t = Arc::clone(&stop);
        let state = Arc::new(state);
        let handle = thread::spawn(move || {
            run_loop(&server, &state, Some(&stop_t)).expect("loop");
        });
        (addr, stop, handle)
    }

    #[cfg(target_os = "linux")]
    fn http(addr: SocketAddr, request: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(request.as_bytes()).expect("write");
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (status, text)
    }

    #[cfg(target_os = "linux")]
    fn wait_up(addr: SocketAddr) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if TcpStream::connect(addr).is_ok() {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("server did not listen on {addr}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn loopback_health_without_token() {
        let (addr, stop, handle) = spawn_server(ServeState::new(
            ServeKind::Stt,
            "whisper-small".into(),
            None,
        ));
        wait_up(addr);
        let host = addr.to_string();
        let (status, body) = http(
            addr,
            &format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"model\":\"whisper-small\""), "{body}");
        assert!(body.contains("\"busy\":false"), "{body}");
        assert!(body.contains("\"kind\":\"stt\""), "{body}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn token_protects_health_and_unknown_is_404() {
        let (addr, stop, handle) = spawn_server(ServeState::new(
            ServeKind::Tts,
            "pocket-tts".into(),
            Some(String::from("s3cret")),
        ));
        wait_up(addr);
        let host = addr.to_string();
        let (status, body) = http(
            addr,
            &format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        );
        assert_eq!(status, 401, "{body}");
        assert!(body.contains("unauthorized"), "{body}");

        let (status, body) = http(
            addr,
            &format!(
                "GET /health HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer wrong\r\nConnection: close\r\n\r\n"
            ),
        );
        assert_eq!(status, 401, "{body}");

        let (status, body) = http(
            addr,
            &format!(
                "GET /health HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer s3cret\r\nConnection: close\r\n\r\n"
            ),
        );
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("pocket-tts"), "{body}");

        let (status, body) = http(
            addr,
            &format!(
                "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer s3cret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            ),
        );
        assert_eq!(status, 404, "{body}");
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn trimmed_env_token_matches_bearer() {
        let token = read_token(|_| Some(String::from(" secret "))).expect("trim");
        let (addr, stop, handle) = spawn_server(ServeState::new(
            ServeKind::Stt,
            "whisper-small".into(),
            token,
        ));
        wait_up(addr);
        let host = addr.to_string();
        let (status, body) = http(
            addr,
            &format!(
                "GET /health HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer secret\r\nConnection: close\r\n\r\n"
            ),
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("whisper-small"), "{body}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn health_busy_when_job_held() {
        let state = ServeState::new(ServeKind::Stt, "qwen3-asr-0.6".into(), None);
        let server = listen("127.0.0.1:0".parse().unwrap()).expect("listen");
        let addr = bound_addr(&server).expect("tcp");
        let stop = Arc::new(AtomicBool::new(false));
        let stop_t = Arc::clone(&stop);
        let state = Arc::new(state);
        let held = state.lock_job();
        let state_t = Arc::clone(&state);
        let handle = thread::spawn(move || {
            run_loop(&server, &state_t, Some(&stop_t)).expect("loop");
        });
        wait_up(addr);
        let host = addr.to_string();
        let (status, body) = http(
            addr,
            &format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"),
        );
        drop(held);
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{body}");
        assert!(body.contains("\"busy\":true"), "{body}");
        assert!(body.contains("qwen3-asr-0.6"), "{body}");
    }
}
