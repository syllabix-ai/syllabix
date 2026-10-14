//! OpenAI-compatible `audio/speech` adapter for one finished sentence.
//!
//! Sentence buffering stays on the endpoint. Each completed sentence is posted
//! to `POST {base_url}/audio/speech` with `stream_format=audio` so the first
//! PCM can play before the rest of the body arrives. Barge-in drops the
//! in-flight request. The API key comes from the environment only
//! ([`crate::config::TTS_API_KEY_ENV`]).
//! Cancel, connect/idle timeouts, and a best-effort warm-up match [`crate::openai`].

use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ureq::Agent;
use zeroize::Zeroizing;

use crate::audio::{f32_to_i16, i16_to_f32, read_wav, PcmConverter, PcmFormat};
use crate::cancel::Cancel;
use crate::config::{AgentConfig, TTS_API_KEY_ENV};
use crate::error::{Error, Result};
use crate::openai::OpenAiTimeouts;
use crate::providers::Tts;
use crate::speech_text::{speak_text_for_tts, take_sentences, ThinkFilter};
use crate::types::{
    GenerationId, SynthesizedAudio, TokenChunk, TurnId, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ,
};

/// Config / log name. Same word as [`crate::defaults::TtsProvider::Online`].
pub const PROVIDER_NAME: &str = "online";

/// Default `voice` for OpenAI-compatible speech endpoints. Workers may ignore it.
pub const DEFAULT_VOICE: &str = "alloy";

/// Attempts used to establish the initial speech connection.
const WARM_UP_ATTEMPTS: usize = 5;
/// Short pause between failed cold-connection attempts.
const WARM_UP_RETRY_DELAY: Duration = Duration::from_secs(1);
/// How often the speech loop wakes to check cancel and idle deadlines.
const POLL_TICK: Duration = Duration::from_millis(100);
/// Bound a speech body so a misbehaving endpoint cannot fill RAM.
const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
/// OpenAI `response_format=pcm` is 24 kHz mono PCM16. Endpoints that ignore
/// `response_format=wav` still land here.
const OPENAI_PCM_SAMPLE_RATE_HZ: u32 = 24_000;

/// `{base}/audio/speech`, tolerating a trailing slash.
pub fn join_speech_endpoint(base: &str) -> String {
    format!("{}/audio/speech", base.trim_end_matches('/'))
}

/// Resolve the BYO TTS key from the environment. [`TTS_API_KEY_ENV`] only;
/// missing or empty fails fast. There is no STT or shared-key fallback.
pub fn resolve_api_key(getenv: impl Fn(&str) -> Option<String>) -> Result<Zeroizing<String>> {
    match getenv(TTS_API_KEY_ENV) {
        Some(key) if !key.trim().is_empty() => Ok(Zeroizing::new(key)),
        Some(_) => Err(Error::Config {
            field: TTS_API_KEY_ENV.into(),
            message: "is set but empty. Export your key, e.g. export SYLLABIX_TTS_API_KEY=sk-…, \
                      and start syllabix again."
                .into(),
        }),
        None => Err(Error::Config {
            field: TTS_API_KEY_ENV.into(),
            message: format!(
                "is required when pipeline.tts.provider is {PROVIDER_NAME}. \
                 Supply it for one run ({TTS_API_KEY_ENV}=sk-… syllabix run) or export it. \
                 Keys are read from the environment only — never from syllabix.yaml or a .env file."
            ),
        }),
    }
}

/// Endpoint + model resolved from [`AgentConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteTtsSettings {
    /// Absolute speech URL (`{base_url}/audio/speech`).
    pub endpoint: String,
    /// Model id sent verbatim; the endpoint decides what it serves.
    pub model: String,
    /// OpenAI-compatible `voice`. Pocket workers may ignore it.
    pub voice: String,
}

impl RemoteTtsSettings {
    /// Build from a validated online TTS config (`base_url` already required).
    pub fn from_config(config: &AgentConfig) -> Result<Self> {
        let base = config
            .tts_base_url
            .as_deref()
            .ok_or_else(|| Error::Config {
                field: "pipeline.tts.base_url".into(),
                message: "is required when pipeline.tts.provider is online".into(),
            })?;
        Ok(Self {
            endpoint: join_speech_endpoint(base),
            model: config.tts_model_id.clone(),
            voice: DEFAULT_VOICE.to_string(),
        })
    }
}

/// Per-generation sentence buffer. Same `take_sentences` loop as local TTS.
#[derive(Default)]
struct ChunkState {
    think: ThinkFilter,
    buffer: String,
    turn: Option<TurnId>,
    generation: Option<GenerationId>,
    next_index: u32,
}

impl ChunkState {
    fn reset(&mut self) {
        self.think = ThinkFilter::default();
        self.buffer.clear();
        self.turn = None;
        self.generation = None;
        self.next_index = 0;
    }

    fn emit(&mut self, samples: Vec<i16>, token: &TokenChunk, is_last: bool) -> SynthesizedAudio {
        let index = self.next_index;
        self.next_index += 1;
        SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index,
            samples,
            is_last,
        }
    }
}

/// Blocking client for one OpenAI-compatible speech endpoint.
pub struct RemoteTts {
    settings: RemoteTtsSettings,
    api_key: Zeroizing<String>,
    agent: Agent,
    timeouts: OpenAiTimeouts,
    state: ChunkState,
}

impl RemoteTts {
    /// Create an adapter with the built-in connect and idle timeouts.
    pub fn new(settings: RemoteTtsSettings, api_key: Zeroizing<String>) -> Self {
        Self::with_timeouts(settings, api_key, OpenAiTimeouts::default())
    }

    /// Constructor with explicit timeouts (tests) and rustls-only agent.
    pub fn with_timeouts(
        settings: RemoteTtsSettings,
        api_key: Zeroizing<String>,
        timeouts: OpenAiTimeouts,
    ) -> Self {
        Self {
            settings,
            api_key,
            agent: build_agent(timeouts),
            timeouts,
            state: ChunkState::default(),
        }
    }

    /// Endpoint used by this run.
    pub fn settings(&self) -> &RemoteTtsSettings {
        &self.settings
    }

    /// Override the OpenAI `voice` field after construction.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.settings.voice = voice.into();
        self
    }

    /// Exercise the cold DNS, TCP, TLS, and speech POST before the voice
    /// loop starts. Normal turns never retry. The probe is not spoken.
    pub fn warm_up(&self, cancel: &Cancel) -> Result<()> {
        let body = speech_json(&self.settings.model, "OK.", &self.settings.voice);
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
                .set("Content-Type", "application/json")
                .send_string(&body)
            {
                Ok(response) => {
                    let mut bytes = Vec::new();
                    response
                        .into_reader()
                        .take(MAX_RESPONSE_BYTES)
                        .read_to_end(&mut bytes)
                        .map_err(|err| Error::Provider {
                            provider: PROVIDER_NAME,
                            message: format!("warm-up read failed: {err}"),
                        })?;
                    decode_speech_body(&bytes)?;
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

    fn synthesize_sentences(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_stale(token.generation) {
            self.state.reset();
            return Err(Error::Cancelled);
        }
        if self.state.generation != Some(token.generation) || self.state.turn != Some(token.turn) {
            self.state.reset();
            self.state.generation = Some(token.generation);
            self.state.turn = Some(token.turn);
        }

        self.state
            .buffer
            .push_str(&self.state.think.push(&token.text, token.is_last));
        let sentences = take_sentences(&mut self.state.buffer, token.is_last);
        let spoken: Vec<String> = sentences
            .into_iter()
            .map(|sentence| speak_text_for_tts(&sentence))
            .filter(|sentence| !sentence.is_empty())
            .collect();
        let last_i = spoken.len().saturating_sub(1);
        let mut emitted_final = false;
        for (i, sentence) in spoken.into_iter().enumerate() {
            if cancel.is_stale(token.generation) || cancel.is_shutdown() {
                self.state.reset();
                return Err(Error::Cancelled);
            }
            let is_last_sentence = token.is_last && i == last_i;
            let mut state = std::mem::take(&mut self.state);
            let mut emitted = emitted_final;
            let result = self.post_speech(
                &sentence,
                cancel,
                is_last_sentence,
                &mut |samples, is_last| {
                    emitted |= is_last;
                    on_audio(state.emit(samples, token, is_last))
                },
            );
            self.state = state;
            emitted_final = emitted;
            result?;
        }
        if token.is_last && !emitted_final {
            on_audio(self.state.emit(vec![0; 16], token, true))?;
        }
        Ok(())
    }

    fn post_speech(
        &self,
        sentence: &str,
        cancel: &Cancel,
        is_last_sentence: bool,
        on_samples: &mut dyn FnMut(Vec<i16>, bool) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let body = speech_json(&self.settings.model, sentence, &self.settings.voice);
        let generation = cancel.generation();
        let abandon = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<WorkerMsg>>();
        let worker = SpeechWorker {
            agent: self.agent.clone(),
            endpoint: self.settings.endpoint.clone(),
            api_key: self.api_key.as_str().to_string(),
            body,
            abandon: Arc::clone(&abandon),
            tx,
        };
        std::thread::Builder::new()
            .name("syllabix-tts".into())
            .spawn(move || worker.run())
            .map_err(|err| Error::Provider {
                provider: PROVIDER_NAME,
                message: format!("failed to start speech worker: {err}"),
            })?;
        let _abandon_on_drop = AbandonOnDrop(abandon);

        let mut idle_since = Instant::now();
        let mut decoder = SpeechStream::default();
        loop {
            if cancel.is_stale(generation) || cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            match rx.recv_timeout(POLL_TICK) {
                Ok(Ok(WorkerMsg::Chunk(bytes))) => {
                    idle_since = Instant::now();
                    let samples = decoder.push(&bytes)?;
                    if !samples.is_empty() {
                        on_samples(samples, false)?;
                    }
                }
                Ok(Ok(WorkerMsg::Done)) => break,
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
                        message: "speech worker exited without a verdict".into(),
                    });
                }
            }
        }
        let tail = decoder.finish()?;
        if !tail.is_empty() {
            on_samples(tail, is_last_sentence)?;
        } else if is_last_sentence {
            on_samples(vec![0; 16], true)?;
        }
        Ok(())
    }
}

impl Tts for RemoteTts {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn model_id(&self) -> Option<&str> {
        Some(self.settings.model.as_str())
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        let mut out = Vec::new();
        self.synthesize_sentences(token, cancel, &mut |audio| {
            out.push(audio);
            Ok(())
        })?;
        Ok(out)
    }

    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        self.synthesize_sentences(token, cancel, on_audio)
    }
}

fn build_agent(timeouts: OpenAiTimeouts) -> Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(timeouts.connect)
        .timeout_read(timeouts.read_poll)
        .user_agent(concat!("syllabix/", env!("CARGO_PKG_VERSION")))
        .build()
}

enum WorkerMsg {
    Chunk(Vec<u8>),
    Done,
}

struct SpeechWorker {
    agent: Agent,
    endpoint: String,
    api_key: String,
    body: String,
    abandon: Arc<AtomicBool>,
    tx: mpsc::Sender<Result<WorkerMsg>>,
}

impl SpeechWorker {
    fn run(self) {
        let send = |event: Result<WorkerMsg>| {
            let _ = self.tx.send(event);
        };
        if self.abandon.load(Ordering::SeqCst) {
            return;
        }
        let response = match self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&self.body)
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
        let mut total = 0u64;
        let mut chunk = [0u8; 2048];
        loop {
            if self.abandon.load(Ordering::SeqCst) {
                return;
            }
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    total = total.saturating_add(n as u64);
                    if total > MAX_RESPONSE_BYTES {
                        send(Err(Error::Provider {
                            provider: PROVIDER_NAME,
                            message: "speech body exceeded 16 MiB".into(),
                        }));
                        return;
                    }
                    send(Ok(WorkerMsg::Chunk(chunk[..n].to_vec())));
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
                        message: format!("speech read failed: {err}"),
                    }));
                    return;
                }
            }
        }
        send(Ok(WorkerMsg::Done));
    }
}

/// Incremental decoder for a speech body: raw PCM16 streams as it arrives
/// through a held 24 kHz → 16 kHz converter; a WAV body is held until the
/// response finishes so the header can be read.
#[derive(Default)]
struct SpeechStream {
    buf: Vec<u8>,
    kind: StreamKind,
    converter: Option<PcmConverter>,
}

#[derive(Default)]
enum StreamKind {
    #[default]
    Unknown,
    Pcm,
    Wav,
}

impl SpeechStream {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        self.buf.extend_from_slice(bytes);
        self.classify();
        match self.kind {
            StreamKind::Pcm => self.take_pcm_frames(),
            StreamKind::Unknown | StreamKind::Wav => Ok(Vec::new()),
        }
    }

    fn finish(mut self) -> Result<Vec<i16>> {
        self.classify();
        match self.kind {
            StreamKind::Wav | StreamKind::Unknown => {
                if self.buf.is_empty() {
                    Ok(Vec::new())
                } else {
                    decode_speech_body(&self.buf)
                }
            }
            StreamKind::Pcm => {
                let mut samples = self.take_pcm_frames()?;
                if !self.buf.is_empty() {
                    return Err(Error::Provider {
                        provider: PROVIDER_NAME,
                        message: "PCM speech body is not a whole number of 16-bit samples".into(),
                    });
                }
                if let Some(converter) = self.converter.as_mut() {
                    samples.extend(f32_to_i16(&converter.flush()));
                }
                Ok(samples)
            }
        }
    }

    fn classify(&mut self) {
        if !matches!(self.kind, StreamKind::Unknown) || self.buf.len() < 4 {
            return;
        }
        self.kind = if self.buf.starts_with(b"RIFF") {
            StreamKind::Wav
        } else {
            StreamKind::Pcm
        };
    }

    fn take_pcm_frames(&mut self) -> Result<Vec<i16>> {
        let n = self.buf.len() / 2 * 2;
        if n == 0 {
            return Ok(Vec::new());
        }
        let taken: Vec<u8> = self.buf.drain(..n).collect();
        let pcm = pcm16le(&taken)?;
        if self.converter.is_none() {
            self.converter = Some(PcmConverter::new(openai_pcm_format(), PcmFormat::v0())?);
        }
        let converter = self
            .converter
            .as_mut()
            .expect("converter is initialized above");
        Ok(f32_to_i16(&converter.push(&i16_to_f32(&pcm))))
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
        format!(" check {TTS_API_KEY_ENV}")
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

fn speech_json(model: &str, input: &str, voice: &str) -> String {
    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "response_format": "pcm",
        "stream_format": "audio",
    });
    if !voice.is_empty() {
        body["voice"] = serde_json::Value::String(voice.to_string());
    }
    body.to_string()
}

fn openai_pcm_format() -> PcmFormat {
    PcmFormat {
        sample_rate_hz: OPENAI_PCM_SAMPLE_RATE_HZ,
        channels: DEFAULT_CHANNELS,
    }
}

fn decode_speech_body(bytes: &[u8]) -> Result<Vec<i16>> {
    if bytes.starts_with(b"RIFF") {
        let wav = read_wav(Cursor::new(bytes))?;
        return resample_to_pipeline(&wav.format, &wav.samples);
    }
    resample_to_pipeline(&openai_pcm_format(), &pcm16le(bytes)?)
}

fn pcm16le(bytes: &[u8]) -> Result<Vec<i16>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(Error::Provider {
            provider: PROVIDER_NAME,
            message: "PCM speech body is not a whole number of 16-bit samples".into(),
        });
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn resample_to_pipeline(format: &PcmFormat, samples: &[i16]) -> Result<Vec<i16>> {
    if format.sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ && format.channels == DEFAULT_CHANNELS {
        return Ok(samples.to_vec());
    }
    let mut converter = PcmConverter::new(*format, PcmFormat::v0())?;
    let mut out = converter.push(&i16_to_f32(samples));
    out.extend(converter.flush());
    Ok(f32_to_i16(&out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{write_wav, WavPcm};
    use crate::config::STT_API_KEY_ENV;
    use crate::defaults::TtsProvider;
    use crate::types::GenerationId;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::time::Duration;

    const TEST_KEY: &str = "sk-tts-test";

    fn token(text: &str, is_last: bool) -> TokenChunk {
        TokenChunk {
            turn: TurnId(3),
            generation: GenerationId(0),
            index: 0,
            text: text.into(),
            is_last,
        }
    }

    fn test_tts(endpoint: String) -> RemoteTts {
        RemoteTts::with_timeouts(
            RemoteTtsSettings {
                endpoint,
                model: "pocket-tts".into(),
                voice: DEFAULT_VOICE.into(),
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
        (format!("http://127.0.0.1:{port}/v1/audio/speech"), handle)
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
        (format!("http://127.0.0.1:{port}/v1/audio/speech"), handle)
    }

    fn write_bytes(
        stream: &mut TcpStream,
        status_line: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> std::io::Result<()> {
        let mut response = format!("{status_line}\r\n");
        for (name, value) in headers {
            response.push_str(&format!("{name}: {value}\r\n"));
        }
        response.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        stream.write_all(response.as_bytes())?;
        stream.write_all(body)?;
        stream.flush()
    }

    fn pcm_samples() -> Vec<i16> {
        vec![1, -2, 3, -4, 5, 6, 7, 8]
    }

    fn openai_pcm_samples() -> Vec<i16> {
        (1..=48).map(|i| i * 200).collect()
    }

    fn openai_pcm_bytes() -> Vec<u8> {
        openai_pcm_samples()
            .into_iter()
            .flat_map(|s| s.to_le_bytes())
            .collect()
    }

    fn wav_bytes() -> Vec<u8> {
        let wav = WavPcm {
            format: PcmFormat::v0(),
            samples: pcm_samples(),
        };
        let mut buf = Vec::new();
        write_wav(&mut buf, &wav).expect("wav");
        buf
    }

    #[test]
    fn join_endpoint_strips_a_trailing_slash() {
        assert_eq!(
            join_speech_endpoint("http://127.0.0.1:8092/v1/"),
            "http://127.0.0.1:8092/v1/audio/speech"
        );
    }

    #[test]
    fn settings_require_online_base_url() {
        let mut config = AgentConfig::v0();
        config.tts = TtsProvider::Online;
        let err = RemoteTtsSettings::from_config(&config).expect_err("missing base_url");
        assert!(err.to_string().contains("pipeline.tts.base_url"), "{err}");

        config.tts_base_url = Some("http://127.0.0.1:8092/v1".into());
        config.tts_model_id = "tts-1".into();
        let settings = RemoteTtsSettings::from_config(&config).expect("online settings");
        assert_eq!(settings.endpoint, "http://127.0.0.1:8092/v1/audio/speech");
        assert_eq!(settings.model, "tts-1");
        assert_eq!(settings.voice, DEFAULT_VOICE);
    }

    #[test]
    fn api_key_requires_tts_env_only() {
        let missing = resolve_api_key(|_| None).expect_err("missing key");
        assert!(missing.to_string().contains(TTS_API_KEY_ENV), "{missing}");

        let empty = resolve_api_key(|name| (name == TTS_API_KEY_ENV).then(|| "  ".into()))
            .expect_err("empty TTS key");
        assert!(empty.to_string().contains(TTS_API_KEY_ENV), "{empty}");

        let stt_only = resolve_api_key(|name| (name == STT_API_KEY_ENV).then(|| "sk-stt".into()))
            .expect_err("STT key is not a TTS fallback");
        assert!(stt_only.to_string().contains(TTS_API_KEY_ENV), "{stt_only}");

        let from_tts =
            resolve_api_key(|name| (name == TTS_API_KEY_ENV).then(|| "sk-tts".into())).unwrap();
        assert_eq!(from_tts.as_str(), "sk-tts");
    }

    #[test]
    fn unfinished_sentence_does_not_post() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut tts = test_tts(endpoint);
        let chunks = tts
            .synthesize_chunk(&token("Hello", false), &Cancel::new())
            .expect("buffering is not an error");
        assert!(chunks.is_empty());
        assert!(
            !handle.join().expect("listener"),
            "unfinished sentence must not open a socket"
        );
    }

    #[test]
    fn cancelled_before_send_opens_no_connection() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut tts = test_tts(endpoint);
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = tts
            .synthesize_chunk(&token("Hello.", true), &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn synthesize_posts_one_sentence_and_maps_pcm() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_bytes(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/octet-stream")],
                &openai_pcm_bytes(),
            )
            .expect("write");
        });
        let mut tts = test_tts(endpoint);
        let chunks = tts
            .synthesize_chunk(&token("Hello there.", true), &Cancel::new())
            .expect("synthesize");
        let expected =
            resample_to_pipeline(&openai_pcm_format(), &openai_pcm_samples()).expect("resample");
        let spoken: Vec<i16> = chunks
            .iter()
            .flat_map(|chunk| chunk.samples.iter().copied())
            .collect();
        assert_eq!(&spoken[..expected.len()], expected.as_slice());
        assert!(spoken[expected.len()..].iter().all(|s| *s == 0));
        assert!(chunks.last().expect("audio").is_last);
        assert_eq!(tts.name(), PROVIDER_NAME);
        assert_eq!(tts.model_id(), Some("pocket-tts"));

        let captured = handle.join().expect("server");
        let text = String::from_utf8_lossy(&captured);
        assert!(text.contains("POST /v1/audio/speech"), "{text}");
        assert!(text.contains("Authorization: Bearer sk-tts-test"), "{text}");
        assert!(text.contains("\"model\":\"pocket-tts\""), "{text}");
        assert!(text.contains("\"input\":\"Hello there.\""), "{text}");
        assert!(text.contains("\"voice\":\"alloy\""), "{text}");
        assert!(text.contains("\"response_format\":\"pcm\""), "{text}");
        assert!(text.contains("\"stream_format\":\"audio\""), "{text}");
    }

    #[test]
    fn raw_pcm_is_treated_as_24khz_and_resampled() {
        let src: Vec<i16> = (1..=48).map(|i| i * 200).collect();
        let bytes: Vec<u8> = src.iter().flat_map(|s| s.to_le_bytes()).collect();
        let out = decode_speech_body(&bytes).expect("pcm");
        let expected = resample_to_pipeline(&openai_pcm_format(), &src).expect("resample");
        assert_eq!(out, expected);
        assert_ne!(out, src);
    }

    #[test]
    fn streamed_pcm_resamples_across_chunks() {
        let src = openai_pcm_samples();
        let bytes = openai_pcm_bytes();
        let mut stream = SpeechStream::default();
        let mid = bytes.len() / 2;
        let mut out = stream.push(&bytes[..mid]).expect("first");
        out.extend(stream.push(&bytes[mid..]).expect("rest"));
        out.extend(stream.finish().expect("flush"));
        let expected = decode_speech_body(&bytes).expect("full body");
        assert_eq!(out, expected);
        let isolated_first =
            resample_to_pipeline(&openai_pcm_format(), &src[..src.len() / 2]).expect("isolated");
        assert_ne!(
            &out[..isolated_first.len().min(out.len())],
            isolated_first.as_slice(),
            "chunk-local resample would drop converter state"
        );
    }

    #[test]
    fn wav_body_is_decoded() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_bytes(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "audio/wav")],
                &wav_bytes(),
            )
            .expect("write");
        });
        let mut tts = test_tts(endpoint);
        let chunks = tts
            .synthesize_chunk(&token("Hi.", true), &Cancel::new())
            .expect("wav");
        assert_eq!(chunks[0].samples, pcm_samples());
        let _ = handle.join();
    }

    #[test]
    fn empty_voice_omits_the_field() {
        let (endpoint, handle) = serve_one(|stream, request| {
            let text = String::from_utf8_lossy(request);
            assert!(!text.contains("\"voice\""), "{text}");
            write_bytes(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "audio/wav")],
                &wav_bytes(),
            )
            .expect("write");
        });
        let mut tts = test_tts(endpoint).with_voice("");
        tts.synthesize_chunk(&token("Hi.", true), &Cancel::new())
            .expect("synthesize");
        let _ = handle.join();
    }

    #[test]
    fn http_401_points_at_the_tts_key() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_bytes(
                stream,
                "HTTP/1.1 401 Unauthorized",
                &[("Content-Type", "application/json")],
                br#"{"error":"bad key"}"#,
            )
            .expect("write");
        });
        let mut tts = test_tts(endpoint);
        let err = tts
            .synthesize_chunk(&token("Hello.", true), &Cancel::new())
            .expect_err("401");
        assert!(err.to_string().contains("HTTP 401"), "{err}");
        assert!(err.to_string().contains(TTS_API_KEY_ENV), "{err}");
        let _ = handle.join();
    }

    #[test]
    fn odd_pcm_length_fails_the_turn() {
        let (endpoint, handle) = serve_one(|stream, _| {
            write_bytes(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "application/octet-stream")],
                &[1],
            )
            .expect("write");
        });
        let mut tts = test_tts(endpoint);
        let err = tts
            .synthesize_chunk(&token("Hello.", true), &Cancel::new())
            .expect_err("odd pcm");
        assert!(err.to_string().contains("16-bit samples"), "{err}");
        let _ = handle.join();
    }

    #[test]
    fn markup_only_flush_emits_a_closing_chunk_without_a_request() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let mut tts = test_tts(endpoint);
        let chunks = tts
            .synthesize_chunk(&token("**", true), &Cancel::new())
            .expect("closing chunk");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
        assert_eq!(chunks[0].samples, vec![0; 16]);
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn warm_up_posts_ok() {
        let (endpoint, handle) = serve_one(|stream, request| {
            let text = String::from_utf8_lossy(request);
            assert!(text.contains("\"input\":\"OK.\""), "{text}");
            write_bytes(
                stream,
                "HTTP/1.1 200 OK",
                &[("Content-Type", "audio/wav")],
                &wav_bytes(),
            )
            .expect("write");
        });
        test_tts(endpoint).warm_up(&Cancel::new()).expect("warm-up");
        let _ = handle.join();
    }

    #[test]
    fn warm_up_cancelled_before_send_opens_no_connection() {
        let (endpoint, handle) = serve_expect_no_connection(Duration::from_millis(200));
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = test_tts(endpoint).warm_up(&cancel).expect_err("cancelled");
        assert!(matches!(err, Error::Cancelled), "{err}");
        assert!(!handle.join().expect("listener"));
    }

    #[test]
    fn idle_timeout_when_the_body_never_arrives() {
        let (endpoint, handle) = serve_one(|stream, _| {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 64\r\n\r\n")
                .expect("headers");
            stream.flush().expect("flush");
            std::thread::sleep(Duration::from_millis(600));
        });
        let mut tts = test_tts(endpoint);
        let err = tts
            .synthesize_chunk(&token("Hello.", true), &Cancel::new())
            .expect_err("idle");
        assert!(
            err.to_string().contains("idle timeout")
                || err.to_string().contains("exited without a verdict"),
            "{err}"
        );
        let _ = handle.join();
    }

    #[test]
    fn first_audio_arrives_before_the_body_finishes() {
        let (first_written, first_rx) = mpsc::channel();
        let (go, go_rx) = mpsc::channel();
        let (endpoint, handle) = serve_one(move |stream, _| {
            let body = openai_pcm_bytes();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
                )
                .expect("headers");
            let first = &body[..48];
            stream
                .write_all(format!("{:x}\r\n", first.len()).as_bytes())
                .expect("chunk size");
            stream.write_all(first).expect("first pcm");
            stream.write_all(b"\r\n").expect("chunk end");
            stream.flush().expect("flush first");
            first_written.send(()).expect("signal first");
            go_rx.recv().expect("wait to finish");
            let rest = &body[48..];
            stream
                .write_all(format!("{:x}\r\n", rest.len()).as_bytes())
                .expect("rest size");
            stream.write_all(rest).expect("rest");
            stream.write_all(b"\r\n0\r\n\r\n").expect("last chunk");
            stream.flush().expect("flush rest");
        });
        let (audio_tx, audio_rx) = mpsc::channel();
        let mut tts = test_tts(endpoint);
        let worker = std::thread::spawn(move || {
            tts.synthesize_chunk_into(&token("Hello.", true), &Cancel::new(), &mut |audio| {
                let _ = audio_tx.send(audio.samples.len());
                Ok(())
            })
        });
        first_rx.recv().expect("server wrote first bytes");
        let first_len = audio_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("client heard first audio before the rest of the body");
        assert!(first_len > 0);
        go.send(()).expect("release remainder");
        worker.join().expect("client").expect("synthesize");
        let _ = handle.join();
    }

    #[test]
    fn barge_in_drops_the_in_flight_request() {
        let (started, started_rx) = mpsc::channel();
        let (endpoint, handle) = serve_one(move |stream, _| {
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 64\r\n\r\n")
                .expect("headers");
            stream.flush().expect("flush");
            started.send(()).expect("started");
            std::thread::sleep(Duration::from_millis(800));
        });
        let cancel = Cancel::new();
        let cancel_for_worker = cancel.clone();
        let mut tts = test_tts(endpoint);
        let worker = std::thread::spawn(move || {
            tts.synthesize_chunk(&token("Hello.", true), &cancel_for_worker)
        });
        started_rx.recv().expect("headers flushed");
        cancel.cancel_generation();
        let err = worker.join().expect("client").expect_err("barge-in");
        assert!(matches!(err, Error::Cancelled), "{err}");
        let _ = handle.join();
    }
}
