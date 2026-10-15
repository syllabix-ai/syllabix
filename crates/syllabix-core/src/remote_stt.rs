//! OpenAI-compatible `audio/transcriptions` adapter for one finished utterance.
//!
//! VAD still runs on-device. After a turn closes, this client posts the
//! utterance WAV to `POST {base_url}/audio/transcriptions` and maps the JSON
//! body to a [`Transcript`]. The API key comes from the environment only
//! ([`crate::config::STT_API_KEY_ENV`]).
//! Cancel, connect/idle timeouts, and a best-effort warm-up match [`crate::openai`].

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ureq::Agent;
use zeroize::Zeroizing;

use crate::audio::{write_wav, PcmFormat, WavPcm};
use crate::cancel::Cancel;
use crate::config::{AgentConfig, STT_API_KEY_ENV};
use crate::error::{Error, Result};
use crate::language::LANGUAGE_AUTO;
use crate::openai::OpenAiTimeouts;
use crate::providers::Stt;
use crate::types::{
    Transcript, Utterance, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};

/// Config / log name. Same word as [`crate::defaults::SttProvider::Online`].
pub const PROVIDER_NAME: &str = "online";

/// Attempts used to establish the initial transcriptions connection.
const WARM_UP_ATTEMPTS: usize = 5;
/// Short pause between failed cold-connection attempts.
const WARM_UP_RETRY_DELAY: Duration = Duration::from_secs(1);
/// How often the transcribe loop wakes to check cancel and idle deadlines.
const POLL_TICK: Duration = Duration::from_millis(100);
/// Bound a JSON transcript body so a misbehaving endpoint cannot fill RAM.
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;

/// `{base}/audio/transcriptions`, tolerating a trailing slash.
pub fn join_transcriptions_endpoint(base: &str) -> String {
    format!("{}/audio/transcriptions", base.trim_end_matches('/'))
}

/// Resolve the BYO STT key from the environment. [`STT_API_KEY_ENV`] only;
/// missing or empty fails fast. There is no TTS or shared-key fallback.
pub fn resolve_api_key(getenv: impl Fn(&str) -> Option<String>) -> Result<Zeroizing<String>> {
    match getenv(STT_API_KEY_ENV) {
        Some(key) if !key.trim().is_empty() => Ok(Zeroizing::new(key)),
        Some(_) => Err(Error::Config {
            field: STT_API_KEY_ENV.into(),
            message: "is set but empty. Export your key, e.g. export SYLLABIX_STT_API_KEY=sk-…, \
                      and start syllabix again."
                .into(),
        }),
        None => Err(Error::Config {
            field: STT_API_KEY_ENV.into(),
            message: format!(
                "is required when pipeline.stt.provider is {PROVIDER_NAME}. \
                 Supply it for one run ({STT_API_KEY_ENV}=sk-… syllabix run) or export it. \
                 Keys are read from the environment only — never from syllabix.yaml or a .env file."
            ),
        }),
    }
}

/// Endpoint + model resolved from [`AgentConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteSttSettings {
    /// Absolute transcriptions URL (`{base_url}/audio/transcriptions`).
    pub endpoint: String,
    /// Model id sent verbatim; the endpoint decides what it serves.
    pub model: String,
    /// Yaml STT language (`en`, `fr`, … or `auto`). `auto` omits the
    /// multipart `language` field so the endpoint can detect it.
    pub language: String,
}

impl RemoteSttSettings {
    /// Build from a validated online STT config (`base_url` already required).
    pub fn from_config(config: &AgentConfig) -> Result<Self> {
        let base = config
            .stt_base_url
            .as_deref()
            .ok_or_else(|| Error::Config {
                field: "pipeline.stt.base_url".into(),
                message: "is required when pipeline.stt.provider is online".into(),
            })?;
        Ok(Self {
            endpoint: join_transcriptions_endpoint(base),
            model: config.stt_model_id.clone(),
            language: config.language.clone(),
        })
    }
}

/// Blocking client for one OpenAI-compatible transcriptions endpoint.
pub struct RemoteStt {
    settings: RemoteSttSettings,
    api_key: Zeroizing<String>,
    agent: Agent,
    timeouts: OpenAiTimeouts,
}

impl RemoteStt {
    /// Create an adapter with the built-in connect and idle timeouts.
    pub fn new(settings: RemoteSttSettings, api_key: Zeroizing<String>) -> Self {
        Self::with_timeouts(settings, api_key, OpenAiTimeouts::default())
    }

    /// Constructor with explicit timeouts (tests) and rustls-only agent.
    pub fn with_timeouts(
        settings: RemoteSttSettings,
        api_key: Zeroizing<String>,
        timeouts: OpenAiTimeouts,
    ) -> Self {
        Self {
            settings,
            api_key,
            agent: build_agent(timeouts),
            timeouts,
        }
    }

    /// Endpoint used by this run.
    pub fn settings(&self) -> &RemoteSttSettings {
        &self.settings
    }

    /// Set the yaml STT language after construction (same as local engines).
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.settings.language = language.into();
        self
    }

    /// Exercise the cold DNS, TCP, TLS, and transcriptions POST before the
    /// voice loop starts. Normal turns never retry. The probe is not added to
    /// conversation history.
    pub fn warm_up(&self, cancel: &Cancel) -> Result<()> {
        let wav = utterance_wav(&silent_utterance())?;
        let (content_type, body) =
            transcriptions_multipart(&self.settings.model, &self.settings.language, &wav);
        let mut last_transport = None;

        for attempt in 0..WARM_UP_ATTEMPTS {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            match self
                .agent
                .post(&self.settings.endpoint)
                .set(
                    "Authorization",
                    &format!("Bearer {}", self.api_key.as_str()),
                )
                .set("Content-Type", &content_type)
                .send_bytes(&body)
            {
                Ok(response) => {
                    let mut text = String::new();
                    response
                        .into_reader()
                        .take(MAX_RESPONSE_BYTES)
                        .read_to_string(&mut text)
                        .map_err(|err| Error::Provider {
                            provider: PROVIDER_NAME,
                            message: format!("warm-up read failed: {err}"),
                        })?;
                    parse_transcription_json(&text)?;
                    return Ok(());
                }
                Err(ureq::Error::Status(code, response)) => {
                    return Err(http_status_error(code, response, true));
                }
                Err(ureq::Error::Transport(err)) => {
                    last_transport = Some(err);
                    if attempt + 1 < WARM_UP_ATTEMPTS {
                        std::thread::sleep(WARM_UP_RETRY_DELAY);
                    }
                }
            }
        }

        Err(Error::Provider {
            provider: PROVIDER_NAME,
            message: format!(
                "warm-up request failed after {WARM_UP_ATTEMPTS} attempts: {}",
                last_transport.expect("warm-up attempts always record a transport error")
            ),
        })
    }
}

impl Stt for RemoteStt {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: PROVIDER_NAME,
                message: "utterance has no frames".into(),
            });
        }
        let pcm = utterance.pcm();
        if pcm.is_empty() {
            return Err(Error::Provider {
                provider: PROVIDER_NAME,
                message: "utterance has no samples".into(),
            });
        }

        let wav = utterance_wav(utterance)?;
        let (content_type, body) =
            transcriptions_multipart(&self.settings.model, &self.settings.language, &wav);
        let generation = cancel.generation();
        let abandon = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<String>>();
        let worker = TranscribeWorker {
            agent: self.agent.clone(),
            endpoint: self.settings.endpoint.clone(),
            api_key: self.api_key.as_str().to_string(),
            content_type,
            body,
            abandon: Arc::clone(&abandon),
            tx,
        };
        std::thread::Builder::new()
            .name("syllabix-stt".into())
            .spawn(move || worker.run())
            .map_err(|err| Error::Provider {
                provider: PROVIDER_NAME,
                message: format!("failed to start transcription worker: {err}"),
            })?;
        let _abandon_on_drop = AbandonOnDrop(abandon);

        let idle_since = Instant::now();
        let body = loop {
            if cancel.is_stale(generation) || cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            match rx.recv_timeout(POLL_TICK) {
                Ok(Ok(text)) => break text,
                Ok(Err(err)) => return Err(err),
                Err(RecvTimeoutError::Timeout) => {
                    if idle_since.elapsed() >= self.timeouts.idle {
                        return Err(Error::Provider {
                            provider: PROVIDER_NAME,
                            message: format!("no data for {:?} (idle timeout)", self.timeouts.idle),
                        });
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "transcription worker exited without a verdict".into(),
                    });
                }
            }
        };
        let parsed = parse_transcription_json(&body)?;
        let language = if self.settings.language == LANGUAGE_AUTO {
            parsed
                .language
                .unwrap_or_else(|| self.settings.language.clone())
        } else {
            self.settings.language.clone()
        };
        Ok(Transcript {
            turn: utterance.turn,
            text: parsed.text,
            language,
        })
    }
}

fn build_agent(timeouts: OpenAiTimeouts) -> Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(timeouts.connect)
        // Blocking transcriptions do not send the status line until the
        // engine finishes. `read_poll` (2s) is stream granularity and is
        // too short for `syllabix serve stt` on CPU.
        .timeout_read(timeouts.idle)
        .user_agent(concat!("syllabix/", env!("CARGO_PKG_VERSION")))
        .build()
}

struct TranscribeWorker {
    agent: Agent,
    endpoint: String,
    api_key: String,
    content_type: String,
    body: Vec<u8>,
    abandon: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<String>>,
}

impl TranscribeWorker {
    fn run(self) {
        let send = |event: Result<String>| {
            let _ = self.tx.send(event);
        };
        if self.abandon.load(Ordering::SeqCst) {
            return;
        }
        let response = match self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", &self.content_type)
            .send_bytes(&self.body)
        {
            Ok(response) => response,
            Err(ureq::Error::Status(code, response)) => {
                send(Err(http_status_error(code, response, false)));
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
        if self.abandon.load(Ordering::SeqCst) {
            return;
        }
        let mut reader = response.into_reader();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 2048];
        loop {
            if self.abandon.load(Ordering::SeqCst) {
                return;
            }
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if buf.len().saturating_add(n) as u64 > MAX_RESPONSE_BYTES {
                        send(Err(Error::Provider {
                            provider: PROVIDER_NAME,
                            message: "transcript body exceeded 64 KiB".into(),
                        }));
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                Err(err)
                    if err.kind() == std::io::ErrorKind::WouldBlock
                        || err.kind() == std::io::ErrorKind::TimedOut =>
                {
                    continue;
                }
                Err(err) => {
                    send(Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: format!("transcript read failed: {err}"),
                    }));
                    return;
                }
            }
        }
        send(Ok(String::from_utf8_lossy(&buf).into_owned()));
    }
}

struct AbandonOnDrop(Arc<AtomicBool>);

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn http_status_error(code: u16, response: ureq::Response, warm_up: bool) -> Error {
    let mut text = String::new();
    let _ = response
        .into_reader()
        .take(16_384)
        .read_to_string(&mut text);
    let hint = if code == 401 || code == 403 {
        format!(" check {STT_API_KEY_ENV}")
    } else {
        String::new()
    };
    let detail = if text.trim().is_empty() {
        String::new()
    } else {
        format!(": {}", text.trim().chars().take(512).collect::<String>())
    };
    let prefix = if warm_up { "warm-up HTTP" } else { "HTTP" };
    Error::Provider {
        provider: PROVIDER_NAME,
        message: format!("{prefix} {code}{detail}{hint}"),
    }
}

struct ParsedTranscript {
    text: String,
    language: Option<String>,
}

fn parse_transcription_json(body: &str) -> Result<ParsedTranscript> {
    let value: serde_json::Value =
        serde_json::from_str(body.trim()).map_err(|err| Error::Provider {
            provider: PROVIDER_NAME,
            message: format!("malformed transcription JSON: {err}"),
        })?;
    let text = value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::Provider {
            provider: PROVIDER_NAME,
            message: "transcription JSON has no string \"text\" field".into(),
        })?
        .to_string();
    let language = value
        .get("language")
        .and_then(serde_json::Value::as_str)
        .filter(|code| !code.is_empty())
        .map(str::to_string);
    Ok(ParsedTranscript { text, language })
}

fn utterance_wav(utterance: &Utterance) -> Result<Vec<u8>> {
    let samples = utterance.pcm();
    let wav = WavPcm {
        format: PcmFormat {
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
        },
        samples,
    };
    let mut buf = Vec::new();
    write_wav(&mut buf, &wav)?;
    Ok(buf)
}

fn transcriptions_multipart(model: &str, language: &str, wav: &[u8]) -> (String, Vec<u8>) {
    let boundary = "----SyllabixSttBoundary";
    let mut body = Vec::new();
    push_multipart_field(&mut body, boundary, "model", model.as_bytes(), None);
    if language != LANGUAGE_AUTO {
        push_multipart_field(&mut body, boundary, "language", language.as_bytes(), None);
    }
    // Default OpenAI Audio format is json. verbose_json is only needed when
    // yaml language is auto so we can read the detected code back.
    let response_format: &[u8] = if language == LANGUAGE_AUTO {
        b"verbose_json"
    } else {
        b"json"
    };
    push_multipart_field(
        &mut body,
        boundary,
        "response_format",
        response_format,
        None,
    );
    push_multipart_field(
        &mut body,
        boundary,
        "file",
        wav,
        Some(("utterance.wav", "audio/wav")),
    );
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn push_multipart_field(
    body: &mut Vec<u8>,
    boundary: &str,
    name: &str,
    value: &[u8],
    file: Option<(&str, &str)>,
) {
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    match file {
        Some((filename, content_type)) => {
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
                     Content-Type: {content_type}\r\n\r\n"
                )
                .as_bytes(),
            );
        }
        None => {
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
        }
    }
    body.extend_from_slice(value);
    body.extend_from_slice(b"\r\n");
}

fn silent_utterance() -> Utterance {
    Utterance {
        turn: crate::types::TurnId(0),
        frames: vec![crate::types::AudioFrame {
            seq: 0,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
            samples: vec![0; FRAME_SAMPLES],
            capture_pcm: None,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TTS_API_KEY_ENV;
    use crate::defaults::SttProvider;
    use crate::types::{AudioFrame, TurnId};
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;

    const TEST_KEY: &str = "sk-stt-test";

    fn frame(samples: Vec<i16>) -> AudioFrame {
        AudioFrame {
            seq: 0,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
            samples,
            capture_pcm: None,
        }
    }

    fn utterance() -> Utterance {
        Utterance {
            turn: TurnId(7),
            frames: vec![frame(vec![1; FRAME_SAMPLES])],
        }
    }

    fn test_stt(endpoint: String) -> RemoteStt {
        RemoteStt::with_timeouts(
            RemoteSttSettings {
                endpoint,
                model: "whisper-small".into(),
                language: "en".into(),
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

    fn serve_one<F>(handler: F) -> (String, std::thread::JoinHandle<Vec<u8>>)
    where
        F: FnOnce(&mut TcpStream, &[u8]) + Send + 'static,
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
            handler(&mut stream, &request);
            request
        });
        (
            format!("http://127.0.0.1:{port}/v1/audio/transcriptions"),
            handle,
        )
    }

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
                    Err(_) => return true,
                }
            }
        });
        (
            format!("http://127.0.0.1:{port}/v1/audio/transcriptions"),
            handle,
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

    #[test]
    fn join_endpoint_strips_a_trailing_slash() {
        assert_eq!(
            join_transcriptions_endpoint("http://127.0.0.1:8091/v1/"),
            "http://127.0.0.1:8091/v1/audio/transcriptions"
        );
    }

    #[test]
    fn settings_require_online_base_url() {
        let mut config = AgentConfig::v0();
        config.stt = SttProvider::Online;
        let err = RemoteSttSettings::from_config(&config).expect_err("missing base_url");
        assert!(err.to_string().contains("pipeline.stt.base_url"), "{err}");

        config.stt_base_url = Some("http://127.0.0.1:8091/v1".into());
        config.stt_model_id = "whisper-large-v3".into();
        let settings = RemoteSttSettings::from_config(&config).expect("online settings");
        assert_eq!(
            settings.endpoint,
            "http://127.0.0.1:8091/v1/audio/transcriptions"
        );
        assert_eq!(settings.model, "whisper-large-v3");
    }

    #[test]
    fn api_key_requires_stt_env_only() {
        let missing = resolve_api_key(|_| None).expect_err("missing key");
        assert!(missing.to_string().contains(STT_API_KEY_ENV), "{missing}");

        let empty = resolve_api_key(|name| (name == STT_API_KEY_ENV).then(|| "  ".into()))
            .expect_err("empty STT key");
        assert!(empty.to_string().contains(STT_API_KEY_ENV), "{empty}");

        let tts_only = resolve_api_key(|name| (name == TTS_API_KEY_ENV).then(|| "sk-tts".into()))
            .expect_err("TTS key is not an STT fallback");
        assert!(tts_only.to_string().contains(STT_API_KEY_ENV), "{tts_only}");

        let from_stt =
            resolve_api_key(|name| (name == STT_API_KEY_ENV).then(|| "sk-stt".into())).unwrap();
        assert_eq!(from_stt.as_str(), "sk-stt");
    }

    #[test]
    fn empty_utterance_fails_without_a_request() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: Vec::new(),
                },
                &Cancel::new(),
            )
            .expect_err("empty utterance");
        assert!(err.to_string().contains("no frames"), "{err}");
        assert!(
            !handle.join().expect("listener"),
            "empty utterance must not open a socket"
        );
    }

    #[test]
    fn cancelled_before_send_opens_no_connection() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut stt = test_stt(endpoint);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = stt
            .transcribe(&utterance(), &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn transcribe_posts_multipart_and_maps_json() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"text":"hello there","language":"en","duration":1.2}"#,
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint);
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("transcribe");
        assert_eq!(transcript.turn, TurnId(7));
        assert_eq!(transcript.text, "hello there");
        assert_eq!(transcript.language, "en");
        assert_eq!(stt.name(), PROVIDER_NAME);

        let captured = handle.join().expect("server");
        let text = String::from_utf8_lossy(&captured);
        assert!(text.contains("POST /v1/audio/transcriptions"), "{text}");
        assert!(text.contains("Authorization: Bearer sk-stt-test"), "{text}");
        assert!(text.contains("name=\"model\""), "{text}");
        assert!(text.contains("whisper-small"), "{text}");
        assert!(text.contains("name=\"language\""), "{text}");
        assert!(text.contains("name=\"response_format\""), "{text}");
        assert!(text.contains("json"), "{text}");
        assert!(
            !text.contains("verbose_json"),
            "pinned language uses json: {text}"
        );
        assert!(text.contains("utterance.wav"), "{text}");
        assert!(captured.windows(4).any(|w| w == b"RIFF"), "WAV payload");
    }

    #[test]
    fn json_response_keeps_yaml_language() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"text":"bonjour"}"#,
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint).with_language("fr");
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("transcribe");
        assert_eq!(transcript.language, "fr");
        assert_eq!(transcript.text, "bonjour");
        let _ = handle.join();
    }

    #[test]
    fn auto_language_omits_the_field_and_uses_the_response() {
        let (endpoint, handle) = serve_one(|stream, request| {
            let text = String::from_utf8_lossy(request);
            assert!(
                !text.contains("name=\"language\""),
                "auto must omit language: {text}"
            );
            assert!(
                text.contains("verbose_json"),
                "auto needs verbose_json: {text}"
            );
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"text":"hola","language":"es"}"#,
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint).with_language(LANGUAGE_AUTO);
        let transcript = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect("transcribe");
        assert_eq!(transcript.language, "es");
        let _ = handle.join();
    }

    #[test]
    fn http_401_points_at_the_stt_key() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_response(
                stream,
                "HTTP/1.1 401 Unauthorized",
                &[("Content-Type", "application/json")],
                r#"{"error":"bad key"}"#,
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect_err("401");
        assert!(err.to_string().contains("HTTP 401"), "{err}");
        assert!(err.to_string().contains(STT_API_KEY_ENV), "{err}");
        let _ = handle.join();
    }

    #[test]
    fn missing_text_field_fails_the_turn() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"language":"en"}"#,
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect_err("missing text");
        assert!(err.to_string().contains("no string \"text\""), "{err}");
        let _ = handle.join();
    }

    #[test]
    fn malformed_json_fails_the_turn() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                "not-json",
            )
            .expect("write");
        });
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect_err("bad json");
        assert!(
            err.to_string().contains("malformed transcription JSON"),
            "{err}"
        );
        let _ = handle.join();
    }

    #[test]
    fn warm_up_posts_a_tiny_wav() {
        let (endpoint, handle) = serve_one(|stream, request| {
            assert!(find(request, b"RIFF").is_some());
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"text":"OK"}"#,
            )
            .expect("write");
        });
        test_stt(endpoint).warm_up(&Cancel::new()).expect("warm-up");
        let _ = handle.join();
    }

    #[test]
    fn warm_up_cancelled_before_send_opens_no_connection() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = test_stt(endpoint).warm_up(&cancel).expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn idle_timeout_when_the_body_never_arrives() {
        let (endpoint, handle) = serve_one(|stream, _| {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 64\r\n\r\n")
                .expect("headers");
            stream.flush().expect("flush");
            std::thread::sleep(Duration::from_millis(600));
        });
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(&utterance(), &Cancel::new())
            .expect_err("idle");
        assert!(
            err.to_string().contains("idle timeout")
                || err.to_string().contains("exited without a verdict"),
            "{err}"
        );
        let _ = handle.join();
    }

    #[test]
    fn empty_samples_fail_without_a_request() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut stt = test_stt(endpoint);
        let err = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(1),
                    frames: vec![frame(Vec::new())],
                },
                &Cancel::new(),
            )
            .expect_err("empty samples");
        assert!(err.to_string().contains("no samples"), "{err}");
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn session_loopback_transcribes_one_scripted_turn_without_a_mic() {
        use crate::session::Session;
        use crate::testing::{scripted_frames, CollectingSink, FakeLlm, FakeTts, FakeVad};

        let (endpoint, handle) = serve_one(|stream, request| {
            let text = String::from_utf8_lossy(request);
            assert!(text.contains("POST /v1/audio/transcriptions"), "{text}");
            assert!(find(request, b"RIFF").is_some(), "utterance WAV");
            write_response(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/json")],
                r#"{"text":"loopback hello","language":"en"}"#,
            )
            .expect("write");
        });
        let report = Session::v0()
            .with_vad(FakeVad::new())
            .with_stt(test_stt(endpoint))
            .with_llm(FakeLlm::new())
            .with_tts(FakeTts)
            .with_sink(CollectingSink::default())
            .with_frames(scripted_frames(1, 2, 1))
            .run(Cancel::new())
            .expect("session loopback");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.turns[0].user_text, "loopback hello");
        assert_eq!(report.skipped_turns, 0);
        let _ = handle.join();
    }
}
