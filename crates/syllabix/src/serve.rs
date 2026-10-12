//! `syllabix serve`: localhost HTTP worker for speech weights.
//!
//! `serve stt` loads one yaml finalize model (`whisper-*` / `qwen3-asr-0.6`) and
//! answers OpenAI `POST /v1/audio/transcriptions`. `serve tts` loads Pocket,
//! Kokoro, or Qwen and answers `POST /v1/audio/speech` (full body or
//! `stream_format=audio`).
//! Bind defaults to loopback. `SYLLABIX_SERVE_TOKEN` is required unless the bind
//! is loopback and the variable is unset (dev only).

use std::io::{self, Read};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use syllabix_core::audio::{
    f32_to_i16, i16_to_f32, read_wav, write_wav, FrameSplitter, PcmConverter, PcmFormat, WavPcm,
};
use syllabix_core::defaults::{SttModel, TtsModel};
use syllabix_core::remote_tts::openai_pcm_format;
use syllabix_core::types::{GenerationId, TokenChunk, TurnId, DEFAULT_SAMPLE_RATE_HZ};
use syllabix_core::{Cancel, Error, Result, Stt, Transcript, Tts, Utterance};
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
/// Extra clients that may wait on the in-flight job mutex before `503`.
pub const MAX_QUEUED: usize = 8;
/// Bound a multipart upload so a client cannot fill RAM.
const MAX_BODY_BYTES: u64 = 16 * 1024 * 1024;

/// STT or TTS worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeKind {
    Stt,
    Tts,
}

/// Listen address, advertised model, optional bearer token, and optional STT engine.
pub struct ServeState {
    /// Worker kind (STT or TTS).
    pub kind: ServeKind,
    /// Yaml model id returned by `/health`.
    pub model: String,
    /// Required bearer token. `None` is loopback-dev only.
    pub token: Option<String>,
    job: Mutex<()>,
    waiters: AtomicUsize,
    stt: Mutex<Option<Box<dyn Stt + Send>>>,
    tts: Mutex<Option<Box<dyn Tts + Send>>>,
}

impl ServeState {
    /// Build state for a listen loop. Weights are optional (`None` = health only).
    pub fn new(kind: ServeKind, model: String, token: Option<String>) -> Self {
        Self {
            kind,
            model,
            token,
            job: Mutex::new(()),
            waiters: AtomicUsize::new(0),
            stt: Mutex::new(None),
            tts: Mutex::new(None),
        }
    }

    /// Attach a finalize STT engine (tests inject a stub; `execute` loads weights).
    pub fn with_stt(self, stt: Box<dyn Stt + Send>) -> Self {
        *self.stt.lock().unwrap_or_else(|err| err.into_inner()) = Some(stt);
        self
    }

    /// Attach a TTS engine (tests inject a stub; `execute` loads weights).
    pub fn with_tts(self, tts: Box<dyn Tts + Send>) -> Self {
        *self.tts.lock().unwrap_or_else(|err| err.into_inner()) = Some(tts);
        self
    }

    /// True while another request holds the job mutex.
    pub fn busy(&self) -> bool {
        self.job.try_lock().is_err()
    }

    #[cfg(test)]
    fn lock_job(&self) -> MutexGuard<'_, ()> {
        self.job.lock().unwrap_or_else(|err| err.into_inner())
    }

    #[cfg(test)]
    fn fill_queue(&self) {
        self.waiters.store(MAX_QUEUED + 1, Ordering::SeqCst);
    }
}

/// Hold the single in-flight job slot.
struct JobPermit<'a> {
    _guard: MutexGuard<'a, ()>,
    waiters: &'a AtomicUsize,
}

impl Drop for JobPermit<'_> {
    fn drop(&mut self) {
        self.waiters.fetch_sub(1, Ordering::SeqCst);
    }
}

fn acquire_job(state: &ServeState) -> std::result::Result<JobPermit<'_>, u16> {
    let waiting = state.waiters.fetch_add(1, Ordering::SeqCst);
    if waiting > MAX_QUEUED {
        state.waiters.fetch_sub(1, Ordering::SeqCst);
        return Err(503);
    }
    let guard = state.job.lock().unwrap_or_else(|err| err.into_inner());
    Ok(JobPermit {
        _guard: guard,
        waiters: &state.waiters,
    })
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
    let mut state = ServeState::new(kind, model.clone(), token);
    if kind == ServeKind::Stt {
        state = state.with_stt(load_local_stt(&model)?);
    }
    if kind == ServeKind::Tts {
        state = state.with_tts(load_local_tts(&model)?);
    }
    let server = listen(addr)?;
    let bound = bound_addr(&server)?;
    tracing::info!(
        kind = ?state.kind,
        model = %state.model,
        %bound,
        auth = state.token.is_some(),
        "serving"
    );
    let routes = match kind {
        ServeKind::Stt => "GET /health, POST /v1/audio/transcriptions",
        ServeKind::Tts => "GET /health, POST /v1/audio/speech",
    };
    eprintln!(
        "syllabix serve {:?}: {} on {bound} ({routes})",
        state.kind, state.model
    );
    run_loop(&server, Arc::new(state), None)
}

/// Reject unknown ids and Moonshine (turn sessions are a later PR).
pub fn validate_finalize_stt_model(model: &str) -> Result<SttModel> {
    let parsed = SttModel::parse(model).ok_or_else(|| Error::Config {
        field: "serve.stt.model".into(),
        message: format!("{model:?} is not a yaml STT id"),
    })?;
    if parsed.is_moonshine() {
        return Err(Error::Config {
            field: "serve.stt.model".into(),
            message: format!(
                "{model} is a streaming engine; this worker serves Whisper and Qwen3-ASR on POST /v1/audio/transcriptions"
            ),
        });
    }
    Ok(parsed)
}

fn load_local_stt(model: &str) -> Result<Box<dyn Stt + Send>> {
    let parsed = validate_finalize_stt_model(model)?;
    #[cfg(coverage)]
    {
        let _ = parsed;
        return Ok(Box::new(FixedStt {
            text: String::from("coverage"),
        }));
    }
    #[cfg(not(coverage))]
    {
        use syllabix_core::models::{HttpFetcher, ModelCache, StderrProgress};
        use syllabix_core::real::build_stt;
        use syllabix_core::AgentConfig;
        let mut config = AgentConfig::v0();
        config.stt_model = parsed;
        config.stt_model_id = model.to_string();
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let cancel = Cancel::new();
        let stt = build_stt(&cache, &HttpFetcher, &mut progress, &cancel, &config, None)?;
        Ok(Box::new(stt))
    }
}

/// Yaml TTS menu ids this worker can load.
pub fn validate_tts_model(model: &str) -> Result<TtsModel> {
    TtsModel::parse(model).ok_or_else(|| Error::Config {
        field: "serve.tts.model".into(),
        message: format!("{model:?} is not a yaml TTS id"),
    })
}

fn load_local_tts(model: &str) -> Result<Box<dyn Tts + Send>> {
    let parsed = validate_tts_model(model)?;
    #[cfg(coverage)]
    {
        let _ = parsed;
        return Ok(Box::new(FixedTts {
            chunks: vec![vec![1, 2, 3, 4]],
        }));
    }
    #[cfg(not(coverage))]
    {
        use syllabix_core::models::{HttpFetcher, ModelCache, StderrProgress};
        use syllabix_core::real::build_tts;
        use syllabix_core::AgentConfig;
        let mut config = AgentConfig::v0();
        config.tts_model = parsed;
        config.tts_model_id = model.to_string();
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let cancel = Cancel::new();
        let tts = build_tts(&cache, &HttpFetcher, &mut progress, &cancel, &config, None)?;
        Ok(Box::new(tts))
    }
}

#[cfg(any(test, coverage))]
struct FixedTts {
    chunks: Vec<Vec<i16>>,
}

#[cfg(any(test, coverage))]
impl Tts for FixedTts {
    fn name(&self) -> &'static str {
        "serve-tts"
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<syllabix_core::SynthesizedAudio>> {
        let mut out = Vec::new();
        self.synthesize_chunk_into(token, cancel, &mut |audio| {
            out.push(audio);
            Ok(())
        })?;
        Ok(out)
    }

    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(syllabix_core::SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let last = self.chunks.len().saturating_sub(1);
        for (i, samples) in self.chunks.iter().enumerate() {
            on_audio(syllabix_core::SynthesizedAudio {
                turn: token.turn,
                generation: token.generation,
                index: i as u32,
                samples: samples.clone(),
                is_last: token.is_last && i == last,
            })?;
        }
        Ok(())
    }
}

#[cfg(any(test, coverage))]
struct FixedStt {
    text: String,
}

#[cfg(any(test, coverage))]
impl Stt for FixedStt {
    fn name(&self) -> &'static str {
        "serve-stt"
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(Transcript {
            turn: utterance.turn,
            text: self.text.clone(),
            language: String::from("en"),
        })
    }
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
///
/// Each request runs on a worker thread so one in-flight job does not block
/// accept. Extra clients sit on [`acquire_job`] until `MAX_QUEUED`, then 503.
pub fn run_loop(server: &Server, state: Arc<ServeState>, stop: Option<&AtomicBool>) -> Result<()> {
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
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            handle_request(request, &state);
        });
    }
}

/// Route one HTTP request. Always responds (or logs a disconnect).
pub fn handle_request(mut request: Request, state: &ServeState) {
    let method = request.method().clone();
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or(url.as_str());
    if let Err(status) = authorize(&request, state.token.as_deref()) {
        let _ = request.respond(json_status(status, error_message(status)));
        return;
    }
    if method == Method::Get && path == "/health" {
        if let Err(err) = request.respond(health_response(state)) {
            tracing::debug!("serve response failed: {err}");
        }
        return;
    }
    if method == Method::Post && path == "/v1/audio/transcriptions" {
        let content_type = header_value(request.headers(), "Content-Type");
        let mut body = Vec::new();
        let read = request
            .as_reader()
            .take(MAX_BODY_BYTES + 1)
            .read_to_end(&mut body);
        let response = match read {
            Err(err) => json_status(400, &format!("failed to read body: {err}")),
            Ok(_) if body.len() as u64 > MAX_BODY_BYTES => json_status(400, "body too large"),
            Ok(_) => transcribe_response(state, content_type.as_deref(), &body),
        };
        if let Err(err) = request.respond(response) {
            tracing::debug!("serve response failed: {err}");
        }
        return;
    }
    if method == Method::Post && path == "/v1/audio/speech" {
        handle_speech(request, state);
        return;
    }
    if let Err(err) = request.respond(json_status(404, "not found")) {
        tracing::debug!("serve response failed: {err}");
    }
}

fn header_value(headers: &[Header], name: &'static str) -> Option<String> {
    headers
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_string())
}

fn transcribe_response(
    state: &ServeState,
    content_type: Option<&str>,
    body: &[u8],
) -> Response<std::io::Cursor<Vec<u8>>> {
    if state.kind != ServeKind::Stt {
        return json_status(404, "not found");
    }
    let _permit = match acquire_job(state) {
        Ok(permit) => permit,
        Err(status) => return json_status(status, error_message(status)),
    };
    let form = match parse_transcription_form(content_type.unwrap_or(""), body) {
        Ok(form) => form,
        Err(message) => return json_status(400, &message),
    };
    if !form.model.is_empty() && form.model != state.model {
        tracing::warn!(
            requested = %form.model,
            worker = %state.model,
            "transcriptions model does not match the worker; using loaded weights"
        );
    }
    let utterance = match utterance_from_wav(&form.file) {
        Ok(utterance) => utterance,
        Err(err) => return json_status(400, &err.to_string()),
    };
    let mut slot = state.stt.lock().unwrap_or_else(|err| err.into_inner());
    let Some(stt) = slot.as_mut() else {
        return json_status(404, "not found");
    };
    let cancel = Cancel::new();
    match stt.transcribe(&utterance, &cancel) {
        Ok(transcript) => {
            transcription_json(&transcript, &form, utterance_duration_secs(&utterance))
        }
        Err(Error::Cancelled) => json_status(499, "cancelled"),
        Err(err) => json_status(500, &err.to_string()),
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
        400 => "bad request",
        499 => "cancelled",
        500 => "provider failed",
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

struct TranscriptionForm {
    model: String,
    language: Option<String>,
    response_format: String,
    file: Vec<u8>,
}

fn parse_transcription_form(
    content_type: &str,
    body: &[u8],
) -> std::result::Result<TranscriptionForm, String> {
    let boundary = multipart_boundary(content_type)
        .ok_or_else(|| "Content-Type must be multipart/form-data with a boundary".to_string())?;
    let parts = split_multipart(body, boundary)?;
    let mut model = String::new();
    let mut language = None;
    let mut response_format = String::from("json");
    let mut file = None;
    for (name, value) in parts {
        match name.as_str() {
            "model" => model = String::from_utf8_lossy(&value).trim().to_string(),
            "language" => {
                let code = String::from_utf8_lossy(&value).trim().to_string();
                if !code.is_empty() {
                    language = Some(code);
                }
            }
            "response_format" => {
                response_format = String::from_utf8_lossy(&value).trim().to_string();
            }
            "file" => file = Some(value),
            _ => {}
        }
    }
    let file = file.ok_or_else(|| "multipart is missing file".to_string())?;
    if file.is_empty() {
        return Err("file is empty".into());
    }
    Ok(TranscriptionForm {
        model,
        language,
        response_format,
        file,
    })
}

fn multipart_boundary(content_type: &str) -> Option<&str> {
    let lower = content_type.to_ascii_lowercase();
    if !lower.contains("multipart/form-data") {
        return None;
    }
    let marker = "boundary=";
    let idx = lower.find(marker)?;
    let rest = content_type[idx + marker.len()..].trim();
    let rest = rest.split(';').next()?.trim();
    Some(rest.trim_matches('"'))
}

fn split_multipart(
    body: &[u8],
    boundary: &str,
) -> std::result::Result<Vec<(String, Vec<u8>)>, String> {
    let token = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = find_bytes(&body[search..], token.as_bytes()) {
        let start = search + rel + token.len();
        if body.get(start..start + 2) == Some(b"--") {
            break;
        }
        let content_start = if body.get(start..start + 2) == Some(b"\r\n") {
            start + 2
        } else {
            start
        };
        let next_rel = find_bytes(&body[content_start..], token.as_bytes())
            .ok_or_else(|| "multipart part is not terminated".to_string())?;
        let mut part = &body[content_start..content_start + next_rel];
        if let Some(stripped) = part.strip_suffix(b"\r\n") {
            part = stripped;
        }
        if let Some((headers, data)) = split_once_bytes(part, b"\r\n\r\n") {
            if let Some(name) = disposition_name(headers) {
                parts.push((name, data.to_vec()));
            }
        }
        search = content_start + next_rel;
    }
    Ok(parts)
}

fn disposition_name(headers: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(headers);
    for line in text.split("\r\n") {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("content-disposition:") {
            continue;
        }
        for piece in line.split(';') {
            let piece = piece.trim();
            let Some(rest) = piece.strip_prefix("name=") else {
                continue;
            };
            return Some(rest.trim_matches('"').to_string());
        }
    }
    None
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn split_once_bytes<'a>(haystack: &'a [u8], needle: &[u8]) -> Option<(&'a [u8], &'a [u8])> {
    let idx = find_bytes(haystack, needle)?;
    Some((&haystack[..idx], &haystack[idx + needle.len()..]))
}

fn utterance_from_wav(bytes: &[u8]) -> Result<Utterance> {
    let wav = read_wav(std::io::Cursor::new(bytes))?;
    let samples = pipeline_pcm(&wav)?;
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(&samples)?;
    frames.extend(splitter.flush()?);
    if frames.is_empty() {
        return Err(Error::InvalidAudio {
            message: "WAV has no samples".into(),
        });
    }
    Ok(Utterance {
        turn: TurnId(0),
        frames,
    })
}

fn pipeline_pcm(wav: &WavPcm) -> Result<Vec<i16>> {
    if wav.format == PcmFormat::v0() {
        return Ok(wav.samples.clone());
    }
    let mut converter = PcmConverter::new(wav.format, PcmFormat::v0())?;
    let mut out = converter.push(&i16_to_f32(&wav.samples));
    out.extend(converter.flush());
    Ok(f32_to_i16(&out))
}

fn utterance_duration_secs(utterance: &Utterance) -> f64 {
    utterance.pcm().len() as f64 / f64::from(DEFAULT_SAMPLE_RATE_HZ)
}

fn transcription_json(
    transcript: &Transcript,
    form: &TranscriptionForm,
    duration: f64,
) -> Response<std::io::Cursor<Vec<u8>>> {
    if form.language.is_some() {
        tracing::debug!(
            requested = ?form.language,
            engine = %transcript.language,
            "transcriptions language field is not applied to local engines"
        );
    }
    let body = if form.response_format == "verbose_json" {
        json!({
            "text": transcript.text,
            "language": transcript.language,
            "duration": duration,
        })
        .to_string()
    } else {
        json!({ "text": transcript.text }).to_string()
    };
    json_response(200, body)
}

#[derive(Debug, Deserialize)]
struct SpeechRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    voice: String,
    #[serde(default)]
    response_format: String,
    #[serde(default)]
    stream_format: String,
}

fn handle_speech(mut request: Request, state: &ServeState) {
    if state.kind != ServeKind::Tts {
        let _ = request.respond(json_status(404, "not found"));
        return;
    }
    let mut body = Vec::new();
    if let Err(err) = request
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_end(&mut body)
    {
        let _ = request.respond(json_status(400, &format!("failed to read body: {err}")));
        return;
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        let _ = request.respond(json_status(400, "body too large"));
        return;
    }
    let req: SpeechRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(err) => {
            let _ = request.respond(json_status(400, &format!("malformed JSON: {err}")));
            return;
        }
    };
    if req.input.trim().is_empty() {
        let _ = request.respond(json_status(400, "input is empty"));
        return;
    }
    let _voice = req.voice;
    if !req.model.is_empty() && req.model != state.model {
        tracing::warn!(
            requested = %req.model,
            worker = %state.model,
            "speech model does not match the worker; using loaded weights"
        );
    }
    let stream = req.stream_format == "audio";
    let wav = req.response_format == "wav";
    let permit = match acquire_job(state) {
        Ok(permit) => permit,
        Err(status) => {
            let _ = request.respond(json_status(status, error_message(status)));
            return;
        }
    };
    let token = TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text: req.input,
        is_last: true,
    };
    let cancel = Cancel::new();
    if stream && !wav {
        {
            let slot = state.tts.lock().unwrap_or_else(|err| err.into_inner());
            if slot.is_none() {
                drop(permit);
                let _ = request.respond(json_status(404, "not found"));
                return;
            }
        }
        let mut encoder = match OpenaiPcmEncoder::new() {
            Ok(encoder) => encoder,
            Err(err) => {
                drop(permit);
                let _ = request.respond(json_status(500, &err.to_string()));
                return;
            }
        };
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut slot = state.tts.lock().unwrap_or_else(|err| err.into_inner());
                if let Some(tts) = slot.as_mut() {
                    let _ = tts.synthesize_chunk_into(&token, &cancel, &mut |audio| {
                        let mut bytes = encoder.push(&audio.samples);
                        if audio.is_last {
                            bytes.extend(encoder.finish());
                        }
                        if !bytes.is_empty() && tx.send(bytes).is_err() {
                            cancel.shutdown();
                            return Err(Error::Cancelled);
                        }
                        Ok(())
                    });
                    let tail = encoder.finish();
                    if !tail.is_empty() && tx.send(tail).is_err() {
                        cancel.shutdown();
                    }
                }
                drop(tx);
            });
            let header = Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..])
                .expect("content-type");
            let response = Response::new(
                StatusCode(200),
                vec![header],
                ChannelReader {
                    rx,
                    buf: Vec::new(),
                },
                None,
                None,
            );
            if let Err(err) = request.respond(response) {
                tracing::debug!("serve speech stream failed: {err}");
            }
        });
        drop(permit);
        return;
    }
    let mut samples = Vec::new();
    let result = {
        let mut slot = state.tts.lock().unwrap_or_else(|err| err.into_inner());
        let Some(tts) = slot.as_mut() else {
            drop(permit);
            let _ = request.respond(json_status(404, "not found"));
            return;
        };
        tts.synthesize_chunk_into(&token, &cancel, &mut |audio| {
            samples.extend_from_slice(&audio.samples);
            Ok(())
        })
    };
    drop(permit);
    if let Err(err) = result {
        let status = if matches!(err, Error::Cancelled) {
            499
        } else {
            500
        };
        let _ = request.respond(json_status(status, &err.to_string()));
        return;
    }
    let bytes = if wav {
        let wav = WavPcm {
            format: PcmFormat::v0(),
            samples,
        };
        let mut buf = Vec::new();
        if let Err(err) = write_wav(&mut buf, &wav) {
            let _ = request.respond(json_status(500, &err.to_string()));
            return;
        }
        buf
    } else {
        match upsample_to_openai_pcm(&samples) {
            Ok(pcm) => pcm16le(&pcm),
            Err(err) => {
                let _ = request.respond(json_status(500, &err.to_string()));
                return;
            }
        }
    };
    let ctype = if wav {
        "audio/wav"
    } else {
        "application/octet-stream"
    };
    let header = Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes()).expect("content-type");
    let response = Response::from_data(bytes)
        .with_status_code(StatusCode(200))
        .with_header(header);
    if let Err(err) = request.respond(response) {
        tracing::debug!("serve speech failed: {err}");
    }
}

fn pcm16le(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Local engines emit 16 kHz; OpenAI `response_format=pcm` is 24 kHz mono.
fn upsample_to_openai_pcm(samples: &[i16]) -> Result<Vec<i16>> {
    let mut encoder = OpenaiPcmEncoder::new()?;
    let mut out = encoder.push_samples(samples);
    out.extend(encoder.finish_samples());
    Ok(out)
}

struct OpenaiPcmEncoder {
    converter: PcmConverter,
    finished: bool,
}

impl OpenaiPcmEncoder {
    fn new() -> Result<Self> {
        Ok(Self {
            converter: PcmConverter::new(PcmFormat::v0(), openai_pcm_format())?,
            finished: false,
        })
    }

    fn push_samples(&mut self, samples: &[i16]) -> Vec<i16> {
        f32_to_i16(&self.converter.push(&i16_to_f32(samples)))
    }

    fn finish_samples(&mut self) -> Vec<i16> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        f32_to_i16(&self.converter.flush())
    }

    fn push(&mut self, samples: &[i16]) -> Vec<u8> {
        pcm16le(&self.push_samples(samples))
    }

    fn finish(&mut self) -> Vec<u8> {
        pcm16le(&self.finish_samples())
    }
}

struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
}

impl Read for ChannelReader {
    fn read(&mut self, dest: &mut [u8]) -> io::Result<usize> {
        if self.buf.is_empty() {
            match self.rx.recv() {
                Ok(chunk) => self.buf = chunk,
                Err(_) => return Ok(0),
            }
        }
        let n = dest.len().min(self.buf.len());
        dest[..n].copy_from_slice(&self.buf[..n]);
        self.buf.drain(..n);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpStream};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

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
            run_loop(&server, state, Some(&stop_t)).expect("loop");
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
            run_loop(&server, state_t, Some(&stop_t)).expect("loop");
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

    fn sample_wav() -> Vec<u8> {
        use syllabix_core::audio::write_wav;
        let wav = WavPcm {
            format: PcmFormat::v0(),
            samples: vec![0; 512],
        };
        let mut buf = Vec::new();
        write_wav(&mut buf, &wav).expect("wav");
        buf
    }

    fn multipart(wav: &[u8], extra: &[(&str, &str)]) -> (String, Vec<u8>) {
        let boundary = "----SyllabixSttBoundary";
        let mut body = Vec::new();
        for (name, value) in extra {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
        );
        body.extend_from_slice(wav);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    #[test]
    fn moonshine_is_rejected_for_transcriptions() {
        let err = validate_finalize_stt_model("moonshine-streaming-small").expect_err("moonshine");
        assert!(matches!(err, Error::Config { .. }));
        validate_finalize_stt_model("whisper-small").expect("whisper");
        validate_finalize_stt_model("qwen3-asr-0.6").expect("qwen");
        validate_finalize_stt_model("nope").expect_err("unknown");
    }

    #[test]
    fn parse_multipart_reads_file_and_fields() {
        let wav = sample_wav();
        let (ct, body) = multipart(
            &wav,
            &[
                ("model", "whisper-small"),
                ("language", "en"),
                ("response_format", "verbose_json"),
            ],
        );
        let form = parse_transcription_form(&ct, &body).expect("form");
        assert_eq!(form.model, "whisper-small");
        assert_eq!(form.language.as_deref(), Some("en"));
        assert_eq!(form.response_format, "verbose_json");
        assert_eq!(form.file, wav);
    }

    #[cfg(target_os = "linux")]
    fn http_raw(addr: SocketAddr, bytes: &[u8]) -> (u16, String) {
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(bytes).expect("write");
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
    #[test]
    fn transcriptions_returns_text_from_stub() {
        let state = ServeState::new(ServeKind::Stt, "whisper-small".into(), None).with_stt(
            Box::new(FixedStt {
                text: String::from("hello there"),
            }),
        );
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let wav = sample_wav();
        let (ct, body) = multipart(&wav, &[("model", "whisper-small")]);
        let host = addr.to_string();
        let mut req = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        req.extend_from_slice(&body);
        let (status, resp) = http_raw(addr, &req);
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{resp}");
        assert!(resp.contains("\"text\":\"hello there\""), "{resp}");
        assert!(!resp.contains("\"language\""), "{resp}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn transcriptions_verbose_json_and_queue_full() {
        let state = ServeState::new(ServeKind::Stt, "whisper-small".into(), None).with_stt(
            Box::new(FixedStt {
                text: String::from("hi"),
            }),
        );
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let wav = sample_wav();
        let (ct, body) = multipart(
            &wav,
            &[
                ("model", "other-id"),
                ("language", "fr"),
                ("response_format", "verbose_json"),
            ],
        );
        let host = addr.to_string();
        let mut req = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        req.extend_from_slice(&body);
        let (status, resp) = http_raw(addr, &req);
        assert_eq!(status, 200, "{resp}");
        assert!(resp.contains("\"language\":\"en\""), "{resp}");
        assert!(!resp.contains("\"language\":\"fr\""), "{resp}");
        assert!(resp.contains("\"duration\""), "{resp}");

        // Fill the waiter count on a second server to hit 503 without blocking.
        let full = ServeState::new(ServeKind::Stt, "whisper-small".into(), None).with_stt(
            Box::new(FixedStt {
                text: String::from("hi"),
            }),
        );
        full.fill_queue();
        let (addr2, stop2, handle2) = spawn_server(full);
        wait_up(addr2);
        let host2 = addr2.to_string();
        let mut req2 = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host2}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        req2.extend_from_slice(&body);
        let (status, resp) = http_raw(addr2, &req2);
        stop.store(true, Ordering::SeqCst);
        stop2.store(true, Ordering::SeqCst);
        let _ = handle.join();
        let _ = handle2.join();
        assert_eq!(status, 503, "{resp}");
        assert!(resp.contains("busy"), "{resp}");
    }

    #[cfg(target_os = "linux")]
    struct HoldStt {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        text: String,
    }

    #[cfg(target_os = "linux")]
    impl Stt for HoldStt {
        fn name(&self) -> &'static str {
            "hold-stt"
        }

        fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
            self.started.store(true, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(5);
            while !self.release.load(Ordering::SeqCst) {
                if cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                if Instant::now() > deadline {
                    return Err(Error::Provider {
                        provider: "hold-stt",
                        message: "hold timed out".into(),
                    });
                }
                thread::sleep(Duration::from_millis(5));
            }
            Ok(Transcript {
                turn: utterance.turn,
                text: self.text.clone(),
                language: String::from("en"),
            })
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn in_flight_job_lets_overflow_return_503() {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let state = ServeState::new(ServeKind::Stt, "whisper-small".into(), None).with_stt(
            Box::new(HoldStt {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
                text: String::from("hi"),
            }),
        );
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let wav = sample_wav();
        let (ct, body) = multipart(&wav, &[("model", "whisper-small")]);
        let holder_ct = ct.clone();
        let holder_body = body.clone();
        let holder = thread::spawn(move || post_transcriptions(addr, &holder_ct, &holder_body));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !started.load(Ordering::SeqCst) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(started.load(Ordering::SeqCst), "first job never started");

        let (tx, rx) = std::sync::mpsc::channel();
        let mut extras = Vec::new();
        for _ in 0..MAX_QUEUED + 2 {
            let ct = ct.clone();
            let body = body.clone();
            let tx = tx.clone();
            extras.push(thread::spawn(move || {
                let result = post_transcriptions(addr, &ct, &body);
                let _ = tx.send(result.0);
            }));
        }
        let overflow_deadline = Instant::now() + Duration::from_secs(2);
        let mut saw_503 = false;
        while Instant::now() < overflow_deadline {
            match rx.try_recv() {
                Ok(503) => {
                    saw_503 = true;
                    break;
                }
                Ok(_) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }
        release.store(true, Ordering::SeqCst);
        let _ = holder.join();
        for extra in extras {
            let _ = extra.join();
        }
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert!(
            saw_503,
            "queued overflow should return 503 while a job is in flight"
        );
    }

    #[cfg(target_os = "linux")]
    fn post_transcriptions(addr: SocketAddr, ct: &str, body: &[u8]) -> (u16, String) {
        let host = addr.to_string();
        let mut req = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        req.extend_from_slice(body);
        http_raw(addr, &req)
    }

    #[test]
    fn multipart_errors_and_8khz_wav_resample() {
        assert!(parse_transcription_form("text/plain", b"x").is_err());
        let (ct, body) = multipart(&[], &[("model", "whisper-small")]);
        assert!(parse_transcription_form(&ct, &body).is_err());
        let wav = {
            let wav = WavPcm {
                format: PcmFormat {
                    sample_rate_hz: 8_000,
                    channels: 1,
                },
                samples: vec![0; 256],
            };
            let mut buf = Vec::new();
            syllabix_core::audio::write_wav(&mut buf, &wav).expect("wav");
            buf
        };
        utterance_from_wav(&wav).expect("resample 8 kHz");
        let mut stt = FixedStt {
            text: String::from("x"),
        };
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = stt
            .transcribe(&utterance_from_wav(&sample_wav()).unwrap(), &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_stt_loader_returns_stub() {
        let mut stt = load_local_stt("whisper-small").expect("stub");
        let transcript = stt
            .transcribe(&utterance_from_wav(&sample_wav()).unwrap(), &Cancel::new())
            .expect("transcribe");
        assert_eq!(transcript.text, "coverage");
        assert!(
            load_local_stt("moonshine-streaming-small").is_err(),
            "moonshine"
        );
    }

    #[test]
    fn tts_model_menu_is_yaml_ids() {
        validate_tts_model("pocket-tts").expect("pocket");
        validate_tts_model("kokoro").expect("kokoro");
        validate_tts_model("qwen3-0.6").expect("qwen");
        validate_tts_model("nope").expect_err("unknown");
    }

    #[test]
    fn openai_pcm_is_24khz_from_16khz_pipeline() {
        let src = vec![1_000i16, -1_000, 2_000, -2_000];
        let out = upsample_to_openai_pcm(&src).expect("upsample");
        assert!(
            out.len() > src.len(),
            "24 kHz must hold more samples than 16 kHz: {} vs {}",
            out.len(),
            src.len()
        );
        assert_eq!(
            openai_pcm_format().sample_rate_hz,
            syllabix_core::remote_tts::OPENAI_PCM_SAMPLE_RATE_HZ
        );
        assert_eq!(syllabix_core::remote_tts::OPENAI_PCM_SAMPLE_RATE_HZ, 24_000);
    }

    #[cfg(target_os = "linux")]
    fn speech_post(addr: SocketAddr, json: &str) -> (u16, Vec<u8>) {
        let host = addr.to_string();
        let mut req = format!(
            "POST /v1/audio/speech HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            json.len()
        )
        .into_bytes();
        req.extend_from_slice(json.as_bytes());
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream.write_all(&req).expect("write");
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf);
        let status = buf
            .split(|&b| b == b'\n')
            .next()
            .and_then(|line| std::str::from_utf8(line).ok())
            .and_then(|t| t.split_whitespace().nth(1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (status, buf)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn transcriptions_rejects_bad_body_and_missing_engine() {
        let (addr, stop, handle) = spawn_server(ServeState::new(
            ServeKind::Stt,
            "whisper-small".into(),
            None,
        ));
        wait_up(addr);
        let host = addr.to_string();
        let (status, body) = http(
            addr,
            &format!(
                "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            ),
        );
        assert_eq!(status, 400, "{body}");
        let wav = sample_wav();
        let (ct, payload) = multipart(&wav, &[("model", "whisper-small")]);
        let mut req = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {host}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        )
        .into_bytes();
        req.extend_from_slice(&payload);
        let (status, body) = http_raw(addr, &req);
        assert_eq!(status, 404, "{body}");
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn speech_returns_full_pcm_body() {
        let state = ServeState::new(ServeKind::Tts, "pocket-tts".into(), None).with_tts(Box::new(
            FixedTts {
                chunks: vec![vec![1, 2], vec![3, 4]],
            },
        ));
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let (status, resp) = speech_post(
            addr,
            r#"{"model":"pocket-tts","input":"Hello there.","response_format":"pcm"}"#,
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&resp));
        let expected = pcm16le(&upsample_to_openai_pcm(&[1, 2, 3, 4]).expect("upsample"));
        let body = http_payload(&resp);
        assert_eq!(body, expected, "{}", String::from_utf8_lossy(&resp));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn speech_stream_format_audio_is_chunked_pcm() {
        let state = ServeState::new(ServeKind::Tts, "qwen3-0.6".into(), None).with_tts(Box::new(
            FixedTts {
                chunks: vec![vec![9], vec![8]],
            },
        ));
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let (status, resp) = speech_post(
            addr,
            r#"{"model":"qwen3-0.6","input":"Hello.","stream_format":"audio"}"#,
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&resp));
        let text = String::from_utf8_lossy(&resp);
        assert!(text.to_ascii_lowercase().contains("chunked"), "{text}");
        let expected = pcm16le(&upsample_to_openai_pcm(&[9, 8]).expect("upsample"));
        assert_eq!(http_payload(&resp), expected, "{text}");
    }

    #[cfg(target_os = "linux")]
    struct WatchTts {
        chunks: usize,
        produced: Arc<AtomicUsize>,
        cancelled: Arc<AtomicBool>,
    }

    #[cfg(target_os = "linux")]
    impl Tts for WatchTts {
        fn name(&self) -> &'static str {
            "watch-tts"
        }

        fn synthesize_chunk(
            &mut self,
            token: &TokenChunk,
            cancel: &Cancel,
        ) -> Result<Vec<syllabix_core::SynthesizedAudio>> {
            let mut out = Vec::new();
            self.synthesize_chunk_into(token, cancel, &mut |audio| {
                out.push(audio);
                Ok(())
            })?;
            Ok(out)
        }

        fn synthesize_chunk_into(
            &mut self,
            token: &TokenChunk,
            cancel: &Cancel,
            on_audio: &mut dyn FnMut(syllabix_core::SynthesizedAudio) -> Result<()>,
        ) -> Result<()> {
            for i in 0..self.chunks {
                if cancel.is_shutdown() {
                    self.cancelled.store(true, Ordering::SeqCst);
                    return Err(Error::Cancelled);
                }
                let result = on_audio(syllabix_core::SynthesizedAudio {
                    turn: token.turn,
                    generation: token.generation,
                    index: i as u32,
                    samples: vec![i as i16; 16_384],
                    is_last: token.is_last && i + 1 == self.chunks,
                });
                if let Err(err) = result {
                    if matches!(err, Error::Cancelled) {
                        self.cancelled.store(true, Ordering::SeqCst);
                    }
                    return Err(err);
                }
                self.produced.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn dropping_speech_stream_cancels_synthesis() {
        let produced = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let state = ServeState::new(ServeKind::Tts, "pocket-tts".into(), None).with_tts(Box::new(
            WatchTts {
                chunks: 12,
                produced: Arc::clone(&produced),
                cancelled: Arc::clone(&cancelled),
            },
        ));
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let json = r#"{"model":"pocket-tts","input":"Hello there.","stream_format":"audio"}"#;
        let host = addr.to_string();
        let mut req = format!(
            "POST /v1/audio/speech HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            json.len()
        )
        .into_bytes();
        req.extend_from_slice(json.as_bytes());
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        stream.write_all(&req).expect("write");
        let started = Instant::now() + Duration::from_secs(2);
        let mut buf = [0u8; 256];
        while produced.load(Ordering::SeqCst) == 0 && Instant::now() < started {
            let _ = stream.read(&mut buf);
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            produced.load(Ordering::SeqCst) >= 1,
            "synthesis never started"
        );
        let _ = stream.shutdown(Shutdown::Both);
        drop(stream);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !cancelled.load(Ordering::SeqCst) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert!(
            cancelled.load(Ordering::SeqCst),
            "dropped stream should cancel synthesis (produced {})",
            produced.load(Ordering::SeqCst)
        );
        assert!(
            produced.load(Ordering::SeqCst) < 12,
            "synthesis kept running after drop: {}",
            produced.load(Ordering::SeqCst)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn speech_wav_header_is_16khz() {
        let state = ServeState::new(ServeKind::Tts, "pocket-tts".into(), None).with_tts(Box::new(
            FixedTts {
                chunks: vec![vec![1, 2, 3, 4]],
            },
        ));
        let (addr, stop, handle) = spawn_server(state);
        wait_up(addr);
        let (status, resp) = speech_post(
            addr,
            r#"{"model":"pocket-tts","input":"Hello.","response_format":"wav"}"#,
        );
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&resp));
        let body = http_payload(&resp);
        assert!(
            body.starts_with(b"RIFF"),
            "{}",
            String::from_utf8_lossy(&resp)
        );
        assert!(body.len() >= 28, "{}", String::from_utf8_lossy(&resp));
        let rate = u32::from_le_bytes(body[24..28].try_into().expect("rate"));
        assert_eq!(rate, DEFAULT_SAMPLE_RATE_HZ);
        assert_eq!(DEFAULT_SAMPLE_RATE_HZ, 16_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn speech_rejects_empty_malformed_queue_and_missing_engine() {
        let (addr, stop, handle) =
            spawn_server(ServeState::new(ServeKind::Tts, "pocket-tts".into(), None));
        wait_up(addr);
        let (status, resp) = speech_post(addr, r#"{"model":"pocket-tts","input":"   "}"#);
        assert_eq!(status, 400, "{}", String::from_utf8_lossy(&resp));
        let (status, resp) = speech_post(addr, "{");
        assert_eq!(status, 400, "{}", String::from_utf8_lossy(&resp));
        let (status, resp) = speech_post(addr, r#"{"model":"other","input":"Hi"}"#);
        assert_eq!(status, 404, "{}", String::from_utf8_lossy(&resp));
        stop.store(true, Ordering::SeqCst);
        let _ = handle.join();

        let full = ServeState::new(ServeKind::Tts, "pocket-tts".into(), None).with_tts(Box::new(
            FixedTts {
                chunks: vec![vec![1]],
            },
        ));
        full.fill_queue();
        let (addr2, stop2, handle2) = spawn_server(full);
        wait_up(addr2);
        let (status, resp) = speech_post(addr2, r#"{"model":"pocket-tts","input":"Hi"}"#);
        stop2.store(true, Ordering::SeqCst);
        let _ = handle2.join();
        assert_eq!(status, 503, "{}", String::from_utf8_lossy(&resp));
        assert!(
            String::from_utf8_lossy(&resp).contains("busy"),
            "{}",
            String::from_utf8_lossy(&resp)
        );
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_tts_loader_returns_stub() {
        let mut tts = load_local_tts("pocket-tts").expect("stub");
        let token = TokenChunk {
            turn: TurnId(0),
            generation: GenerationId(0),
            index: 0,
            text: String::from("hi"),
            is_last: true,
        };
        let chunks = tts
            .synthesize_chunk(&token, &Cancel::new())
            .expect("synthesize");
        assert_eq!(chunks[0].samples, vec![1, 2, 3, 4]);
        assert!(load_local_tts("nope").is_err(), "unknown");
    }

    #[test]
    fn fixed_tts_cancels_and_encoder_flushes_once() {
        let mut tts = FixedTts {
            chunks: vec![vec![1]],
        };
        let cancel = Cancel::new();
        cancel.shutdown();
        let token = TokenChunk {
            turn: TurnId(0),
            generation: GenerationId(0),
            index: 0,
            text: String::from("hi"),
            is_last: true,
        };
        let err = tts
            .synthesize_chunk(&token, &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled));

        let mut encoder = OpenaiPcmEncoder::new().expect("encoder");
        let _ = encoder.push_samples(&[1, 2, 3, 4]);
        let first = encoder.finish_samples();
        let second = encoder.finish_samples();
        assert!(!first.is_empty());
        assert!(second.is_empty());
        assert_eq!(
            openai_pcm_format().sample_rate_hz,
            syllabix_core::remote_tts::OPENAI_PCM_SAMPLE_RATE_HZ
        );
    }

    fn http_payload(resp: &[u8]) -> Vec<u8> {
        let Some(idx) = find_bytes(resp, b"\r\n\r\n") else {
            return Vec::new();
        };
        let headers = String::from_utf8_lossy(&resp[..idx]).to_ascii_lowercase();
        let body = &resp[idx + 4..];
        if headers.contains("chunked") {
            decode_chunked(body)
        } else {
            body.to_vec()
        }
    }

    fn decode_chunked(mut src: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let Some(nl) = find_bytes(src, b"\r\n") else {
                break;
            };
            let size_txt = std::str::from_utf8(&src[..nl]).unwrap_or("0").trim();
            let size = usize::from_str_radix(size_txt, 16).unwrap_or(0);
            src = &src[nl + 2..];
            if size == 0 {
                break;
            }
            if src.len() < size {
                break;
            }
            out.extend_from_slice(&src[..size]);
            src = src.get(size + 2..).unwrap_or(&[]);
        }
        out
    }
}
