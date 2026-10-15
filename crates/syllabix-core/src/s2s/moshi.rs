//! Remote Moshi websocket client (`s2s-opus`).
//!
//! Connects to `{base_url}/api/chat`. The server speaks first with `MT=0`
//! (Python sends one byte; the Rust server sends protocol.md's 8-byte
//! payload). We reply with protocol version 0 and model version 0 only
//! when the server used the 9-byte Rust handshake.

use std::collections::VecDeque;
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tungstenite::client::IntoClientRequest;
use tungstenite::http::HeaderValue;
use tungstenite::protocol::WebSocket;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{client_tls_with_config, Connector, Message};
use zeroize::Zeroizing;

use super::codec::{
    MoshiMessage, OggOpusDecoder, OggOpusEncoder, MOSHI_CHANNELS, MOSHI_SAMPLE_RATE_HZ,
    PROTOCOL_VERSION,
};
use super::{DuplexS2s, S2sEvent};
use crate::audio::{f32_to_i16, i16_to_f32, PcmConverter, PcmFormat};
use crate::cancel::Cancel;
use crate::config::S2sConfig;
use crate::error::{Error, Result};
use crate::types::{AudioFrame, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ};

/// Websocket path on the Moshi HTTP origin (Python, MLX, and Rust servers).
pub const MOSHI_CHAT_PATH: &str = "/api/chat";

const IO_TICK: Duration = Duration::from_millis(20);
/// TCP connect and WebSocket upgrade. Same as [`crate::openai::CONNECT_TIMEOUT`].
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_VERSION: u32 = 0;

enum Wire {
    Audio(AudioFrame),
    Stop,
}

enum Socket {
    Plain(Box<WebSocket<TcpStream>>),
    Tls(Box<WebSocket<MaybeTlsStream<TcpStream>>>),
}

impl Socket {
    fn read(&mut self) -> Result<Option<Message>> {
        let result = match self {
            Self::Plain(ws) => ws.read(),
            Self::Tls(ws) => ws.read(),
        };
        match result {
            Ok(message) => Ok(Some(message)),
            Err(err) if is_would_block(&err) => Ok(None),
            Err(err) => Err(ws_err(err)),
        }
    }

    fn send(&mut self, message: Message) -> Result<()> {
        match self {
            Self::Plain(ws) => ws.send(message),
            Self::Tls(ws) => ws.send(message),
        }
        .map_err(ws_err)
    }

    fn close(&mut self) {
        let _ = match self {
            Self::Plain(ws) => ws.close(None),
            Self::Tls(ws) => ws.close(None),
        };
    }

    fn set_read_timeout(&mut self, timeout: Duration) -> Result<()> {
        match self {
            Self::Plain(ws) => set_tcp_timeouts(ws.get_mut(), timeout, CONNECT_TIMEOUT),
            Self::Tls(ws) => set_maybe_tls_timeouts(ws.get_mut(), timeout, CONNECT_TIMEOUT),
        }
    }
}

struct Shared {
    events: Mutex<VecDeque<S2sEvent>>,
    ready: Condvar,
}

/// Remote Moshi duplex backend: 16 kHz pipeline PCM over 24 kHz Ogg/Opus.
pub struct MoshiWsS2s {
    outgoing: Mutex<Option<std::sync::mpsc::Sender<Wire>>>,
    shared: Arc<Shared>,
}

impl MoshiWsS2s {
    /// Open `{base_url}/api/chat`, complete the handshake, and start I/O.
    pub fn connect(
        config: &S2sConfig,
        api_key: Option<Zeroizing<String>>,
        cancel: &Cancel,
    ) -> Result<Self> {
        if config.tls_insecure && !is_loopback_base_url(&config.base_url) {
            return Err(Error::Config {
                field: "pipeline.s2s.tls_insecure".into(),
                message: "is only allowed when pipeline.s2s.base_url is loopback".into(),
            });
        }
        let url = moshi_ws_url(&config.base_url)?;
        let mut socket = open_socket(
            &url,
            config.tls_insecure,
            api_key.as_ref().map(|key| key.as_str()),
            cancel,
        )?;
        if wait_handshake(&mut socket, cancel)? {
            socket.send(Message::Binary(
                MoshiMessage::Handshake {
                    protocol_version: PROTOCOL_VERSION,
                    model_version: MODEL_VERSION,
                }
                .encode(),
            ))?;
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let shared = Arc::new(Shared {
            events: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
        });
        let io_shared = Arc::clone(&shared);
        let io_cancel = cancel.clone();
        thread::Builder::new()
            .name("syllabix-moshi-ws".into())
            .spawn(move || io_loop(socket, rx, io_shared, io_cancel))
            .map_err(|err| moshi_err(format!("spawn websocket worker: {err}")))?;

        Ok(Self {
            outgoing: Mutex::new(Some(tx)),
            shared,
        })
    }
}

impl Drop for MoshiWsS2s {
    fn drop(&mut self) {
        if let Some(tx) = self.outgoing.lock().expect("moshi send").take() {
            let _ = tx.send(Wire::Stop);
        }
    }
}

impl DuplexS2s for MoshiWsS2s {
    fn name(&self) -> &'static str {
        "moshi"
    }

    fn push(&self, frame: AudioFrame, cancel: &Cancel) -> Result<()> {
        frame.validate()?;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let tx = self.outgoing.lock().expect("moshi send");
        let Some(tx) = tx.as_ref() else {
            return Err(moshi_err("websocket is closed"));
        };
        tx.send(Wire::Audio(frame))
            .map_err(|_| moshi_err("websocket worker exited"))
    }

    fn poll(&self, cancel: &Cancel) -> Result<S2sEvent> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let mut queue = self.shared.events.lock().expect("moshi events");
            if let Some(event) = queue.pop_front() {
                return Ok(event);
            }
            let (guard, _) = self
                .shared
                .ready
                .wait_timeout(queue, Duration::from_millis(10))
                .expect("moshi events");
            drop(guard);
        }
    }
}

/// True when `raw` is an HTTP(S) URL whose host is loopback.
///
/// Used to refuse `tls_insecure` off loopback.
fn is_loopback_base_url(raw: &str) -> bool {
    let Ok(url) = url::Url::parse(raw.trim()) else {
        return false;
    };
    match url.host() {
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        None => false,
    }
}

fn moshi_ws_url(base_url: &str) -> Result<url::Url> {
    let mut url = url::Url::parse(base_url.trim()).map_err(|err| Error::Config {
        field: "pipeline.s2s.base_url".into(),
        message: format!("is not a valid URL ({err})"),
    })?;
    match url.scheme() {
        "http" => {
            url.set_scheme("ws")
                .map_err(|_| moshi_err("could not switch http to ws"))?;
        }
        "https" => {
            url.set_scheme("wss")
                .map_err(|_| moshi_err("could not switch https to wss"))?;
        }
        "ws" | "wss" => {}
        other => {
            return Err(Error::Config {
                field: "pipeline.s2s.base_url".into(),
                message: format!("scheme {other:?} is not supported; use http or https"),
            });
        }
    }
    url.set_path(MOSHI_CHAT_PATH);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn open_socket(
    url: &url::Url,
    tls_insecure: bool,
    api_key: Option<&str>,
    cancel: &Cancel,
) -> Result<Socket> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    let host = url
        .host_str()
        .ok_or_else(|| moshi_err("websocket URL has no host"))?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| moshi_err("websocket URL has no port"))?;
    let stream = tcp_connect(host, port, deadline, cancel)?;
    stream
        .set_nodelay(true)
        .map_err(|err| moshi_err(format!("tcp nodelay: {err}")))?;
    let remaining = remaining_until(deadline);
    set_tcp_timeouts(&stream, remaining, remaining)?;

    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|err| moshi_err(format!("websocket request: {err}")))?;
    if let Some(key) = api_key.map(str::trim).filter(|key| !key.is_empty()) {
        let value = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|err| moshi_err(format!("authorization header: {err}")))?;
        request.headers_mut().insert("Authorization", value);
    }

    let mut socket = match url.scheme() {
        "wss" => {
            let connector = if tls_insecure {
                Some(Connector::Rustls(insecure_tls_config()))
            } else {
                None
            };
            let (ws, _) = complete_ws_handshake(
                client_tls_with_config(request, stream, None, connector),
                deadline,
                cancel,
            )?;
            Socket::Tls(Box::new(ws))
        }
        _ => {
            let (ws, _) =
                complete_ws_handshake(tungstenite::client(request, stream), deadline, cancel)?;
            Socket::Plain(Box::new(ws))
        }
    };
    socket.set_read_timeout(IO_TICK)?;
    Ok(socket)
}

fn remaining_until(deadline: Instant) -> Duration {
    deadline
        .saturating_duration_since(Instant::now())
        .max(Duration::from_millis(1))
}

type ClientHandshakeErr<S> =
    tungstenite::handshake::HandshakeError<tungstenite::handshake::client::ClientHandshake<S>>;
type ClientHandshakeOk<S> = (WebSocket<S>, tungstenite::http::Response<Option<Vec<u8>>>);

fn complete_ws_handshake<S>(
    mut step: std::result::Result<ClientHandshakeOk<S>, ClientHandshakeErr<S>>,
    deadline: Instant,
    cancel: &Cancel,
) -> Result<ClientHandshakeOk<S>>
where
    S: std::io::Read + std::io::Write,
{
    use tungstenite::handshake::HandshakeError;
    loop {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(upgrade_busy_error());
        }
        match step {
            Ok(ok) => return Ok(ok),
            Err(HandshakeError::Failure(err)) => {
                return Err(moshi_err(format!("websocket handshake: {err}")));
            }
            Err(HandshakeError::Interrupted(mid)) => {
                step = mid.handshake();
            }
        }
    }
}

fn upgrade_busy_error() -> Error {
    moshi_err("timed out waiting for websocket upgrade (server busy)")
}

fn tcp_connect(host: &str, port: u16, deadline: Instant, cancel: &Cancel) -> Result<TcpStream> {
    let addrs: Vec<_> = (host, port)
        .to_socket_addrs()
        .map_err(|err| moshi_err(format!("resolve {host}:{port}: {err}")))?
        .collect();
    if addrs.is_empty() {
        return Err(moshi_err(format!("no addresses for {host}:{port}")));
    }
    let mut last_err = None;
    for addr in addrs {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tcp_connect_one(addr, remaining, cancel) {
            Ok(stream) => return Ok(stream),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(err) => last_err = Some(err),
        }
    }
    Err(last_err.unwrap_or_else(|| moshi_err(format!("timed out connecting to {host}:{port}"))))
}

fn tcp_connect_one(
    addr: std::net::SocketAddr,
    timeout: Duration,
    cancel: &Cancel,
) -> Result<TcpStream> {
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(TcpStream::connect_timeout(&addr, timeout));
    });
    loop {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        match rx.try_recv() {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(err)) => return Err(moshi_err(format!("connect {addr}: {err}"))),
            Err(std::sync::mpsc::TryRecvError::Empty) => thread::sleep(Duration::from_millis(10)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(moshi_err(format!("connect {addr}: worker exited")));
            }
        }
    }
}

fn set_tcp_timeouts(stream: &TcpStream, read: Duration, write: Duration) -> Result<()> {
    stream
        .set_read_timeout(Some(read))
        .map_err(|err| moshi_err(format!("tcp read timeout: {err}")))?;
    stream
        .set_write_timeout(Some(write))
        .map_err(|err| moshi_err(format!("tcp write timeout: {err}")))
}

fn set_maybe_tls_timeouts(
    stream: &mut MaybeTlsStream<TcpStream>,
    read: Duration,
    write: Duration,
) -> Result<()> {
    let tcp = match stream {
        MaybeTlsStream::Plain(tcp) => tcp,
        MaybeTlsStream::Rustls(tls) => &mut tls.sock,
        other => {
            return Err(moshi_err(format!("unsupported tls stream {other:?}")));
        }
    };
    set_tcp_timeouts(tcp, read, write)
}

/// Returns `true` when the client must send the 9-byte Rust handshake reply.
fn wait_handshake(socket: &mut Socket, cancel: &Cancel) -> Result<bool> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    loop {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(moshi_err("timed out waiting for handshake"));
        }
        match socket.read()? {
            Some(Message::Binary(bytes)) => match incoming_handshake(&bytes)? {
                Some(reply) => return Ok(reply),
                None => {
                    if let Some(MoshiMessage::Error(message)) = MoshiMessage::decode(&bytes)? {
                        return Err(moshi_err(message));
                    }
                }
            },
            Some(Message::Ping(payload)) => {
                socket.send(Message::Pong(payload))?;
            }
            Some(Message::Pong(_) | Message::Frame(_)) | None => {}
            Some(Message::Text(text)) => {
                return Err(moshi_err(format!("unexpected text frame: {text}")));
            }
            Some(Message::Close(_)) => return Err(moshi_err("closed before handshake")),
        }
    }
}

fn incoming_handshake(bytes: &[u8]) -> Result<Option<bool>> {
    let Some((&ty, payload)) = bytes.split_first() else {
        return Err(moshi_err("empty websocket message"));
    };
    if ty != 0 {
        return Ok(None);
    }
    match payload.len() {
        0 => Ok(Some(false)),
        8 => {
            let protocol = u32::from_le_bytes(payload[0..4].try_into().expect("4"));
            if protocol != PROTOCOL_VERSION {
                return Err(moshi_err(format!(
                    "unsupported protocol version {protocol}"
                )));
            }
            Ok(Some(true))
        }
        n => Err(moshi_err(format!(
            "handshake payload is {n} bytes, expected 0 or 8"
        ))),
    }
}

fn io_loop(
    mut socket: Socket,
    incoming: std::sync::mpsc::Receiver<Wire>,
    shared: Arc<Shared>,
    cancel: Cancel,
) {
    let result = io_loop_inner(&mut socket, &incoming, &shared, &cancel);
    socket.close();
    if let Err(err) = result {
        if !cancel.is_shutdown() {
            push_event(
                &shared,
                S2sEvent::Error {
                    message: err.to_string(),
                },
            );
        }
    }
}

fn io_loop_inner(
    socket: &mut Socket,
    incoming: &std::sync::mpsc::Receiver<Wire>,
    shared: &Shared,
    cancel: &Cancel,
) -> Result<()> {
    let mut encoder = OggOpusEncoder::new()?;
    let mut decoder = OggOpusDecoder::new()?;
    let moshi_pcm = PcmFormat {
        sample_rate_hz: MOSHI_SAMPLE_RATE_HZ,
        channels: MOSHI_CHANNELS,
    };
    let mut up = PcmConverter::new(PcmFormat::v0(), moshi_pcm)?;
    let mut down = PcmConverter::new(moshi_pcm, PcmFormat::v0())?;
    let mut seq = 0_u64;
    loop {
        if cancel.is_shutdown() {
            return Ok(());
        }
        match incoming.try_recv() {
            Ok(Wire::Stop) => return Ok(()),
            Ok(Wire::Audio(frame)) => {
                let pcm24 = f32_to_i16(&up.push(&i16_to_f32(&frame.samples)));
                let pages = encoder.encode(&pcm24)?;
                if !pages.is_empty() {
                    socket.send(Message::Binary(MoshiMessage::Audio(pages).encode()))?;
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return Ok(()),
        }
        match socket.read()? {
            Some(Message::Binary(bytes)) => {
                handle_binary(&bytes, &mut decoder, &mut down, &mut seq, shared)?;
            }
            Some(Message::Ping(payload)) => socket.send(Message::Pong(payload))?,
            Some(Message::Pong(_) | Message::Frame(_)) | None => {}
            Some(Message::Text(text)) => {
                return Err(moshi_err(format!("unexpected text frame: {text}")));
            }
            Some(Message::Close(_)) => return Err(moshi_err("server closed the websocket")),
        }
    }
}

fn handle_binary(
    bytes: &[u8],
    decoder: &mut OggOpusDecoder,
    down: &mut PcmConverter,
    seq: &mut u64,
    shared: &Shared,
) -> Result<()> {
    let Some(msg) = MoshiMessage::decode(bytes)? else {
        return Ok(());
    };
    match msg {
        MoshiMessage::Audio(pages) => {
            let pcm24 = decoder.decode(&pages)?;
            let pcm16 = f32_to_i16(&down.push(&i16_to_f32(&pcm24)));
            if pcm16.is_empty() {
                return Ok(());
            }
            let frame = AudioFrame::new(*seq, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, pcm16)?;
            *seq += 1;
            push_event(shared, S2sEvent::Audio(frame));
        }
        MoshiMessage::Text(text) => push_event(shared, S2sEvent::Text(text)),
        MoshiMessage::Error(message) => push_event(shared, S2sEvent::Error { message }),
        MoshiMessage::Handshake { .. }
        | MoshiMessage::Control(_)
        | MoshiMessage::Metadata(_)
        | MoshiMessage::Ping => {}
    }
    Ok(())
}

fn push_event(shared: &Shared, event: S2sEvent) {
    shared.events.lock().expect("moshi events").push_back(event);
    shared.ready.notify_one();
}

fn insecure_tls_config() -> Arc<rustls::ClientConfig> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    Arc::new(
        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
            .with_no_client_auth(),
    )
}

#[derive(Debug)]
struct NoCertVerifier;

impl rustls::client::danger::ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn is_would_block(err: &tungstenite::Error) -> bool {
    matches!(
        err,
        tungstenite::Error::Io(io) if io.kind() == std::io::ErrorKind::WouldBlock
            || io.kind() == std::io::ErrorKind::TimedOut
    )
}

fn ws_err(err: tungstenite::Error) -> Error {
    moshi_err(err.to_string())
}

fn moshi_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "moshi",
        message: message.into(),
    }
}

/// Optional S2S key from the environment. Missing or blank is `None`.
pub fn s2s_api_key_from_env() -> Option<Zeroizing<String>> {
    match std::env::var(crate::config::S2S_API_KEY_ENV) {
        Ok(key) if !key.trim().is_empty() => Some(Zeroizing::new(key)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "device-audio")]
    use crate::config::S2S_API_KEY_ENV;
    #[cfg(feature = "device-audio")]
    use crate::pipeline::LoopMode;
    #[cfg(feature = "device-audio")]
    use crate::providers::AudioSink;
    #[cfg(feature = "device-audio")]
    use crate::s2s::DuplexS2s;
    #[cfg(feature = "device-audio")]
    use crate::testing::{scripted_frames, CollectingSink};
    #[cfg(feature = "device-audio")]
    use crate::types::SynthesizedAudio;
    #[cfg(feature = "device-audio")]
    use crate::Session;
    use std::net::TcpListener;
    #[cfg(feature = "device-audio")]
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[cfg(feature = "device-audio")]
    use std::sync::Mutex as StdMutex;
    #[cfg(feature = "device-audio")]
    use tungstenite::handshake::server::{Request, Response};

    #[cfg(feature = "device-audio")]
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    #[cfg(feature = "device-audio")]
    #[derive(Clone, Copy)]
    enum FakeMode {
        Echo,
        TextThenEcho,
        ErrorAfterHandshake,
    }

    #[cfg(feature = "device-audio")]
    struct FakeMoshi {
        base_url: String,
        seen_handshake: Arc<AtomicBool>,
        auth: Arc<Mutex<Option<String>>>,
        text_sent: Arc<AtomicBool>,
    }

    #[cfg(feature = "device-audio")]
    fn start_fake(mode: FakeMode) -> FakeMoshi {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen_handshake = Arc::new(AtomicBool::new(false));
        let auth = Arc::new(Mutex::new(None));
        let text_sent = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&seen_handshake);
        let auth_h = Arc::clone(&auth);
        let text_h = Arc::clone(&text_sent);
        thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            let callback = |req: &Request, response: Response| {
                if req.uri().path() != MOSHI_CHAT_PATH {
                    return Err(Response::builder().status(404).body(None).expect("404"));
                }
                if let Some(value) = req.headers().get("Authorization") {
                    *auth_h.lock().expect("auth") =
                        Some(String::from_utf8_lossy(value.as_bytes()).into_owned());
                }
                Ok(response)
            };
            let mut ws = tungstenite::accept_hdr(stream, callback).expect("ws accept");
            ws.send(Message::Binary(
                MoshiMessage::Handshake {
                    protocol_version: PROTOCOL_VERSION,
                    model_version: MODEL_VERSION,
                }
                .encode(),
            ))
            .expect("server handshake");
            if matches!(mode, FakeMode::ErrorAfterHandshake) {
                ws.send(Message::Binary(
                    MoshiMessage::Error(String::from("moshi boom")).encode(),
                ))
                .expect("error");
            }
            loop {
                match ws.read() {
                    Ok(Message::Binary(bytes)) => match MoshiMessage::decode(&bytes) {
                        Ok(Some(MoshiMessage::Handshake { .. })) => {
                            seen.store(true, Ordering::SeqCst);
                        }
                        Ok(Some(MoshiMessage::Audio(pages))) => {
                            if matches!(mode, FakeMode::TextThenEcho)
                                && !text_h.swap(true, Ordering::SeqCst)
                            {
                                ws.send(Message::Binary(
                                    MoshiMessage::Text(String::from("hello from moshi")).encode(),
                                ))
                                .expect("text");
                            }
                            if !matches!(mode, FakeMode::ErrorAfterHandshake) {
                                ws.send(Message::Binary(MoshiMessage::Audio(pages).encode()))
                                    .expect("echo");
                            }
                        }
                        _ => {}
                    },
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        });
        FakeMoshi {
            base_url: format!("http://{addr}"),
            seen_handshake,
            auth,
            text_sent,
        }
    }

    #[cfg(feature = "device-audio")]
    fn s2s_config(base_url: &str) -> crate::config::AgentConfig {
        let mut config = crate::config::AgentConfig::v0();
        config.mode = crate::defaults::PipelineMode::S2s;
        config.s2s = Some(S2sConfig {
            base_url: base_url.into(),
            tls_insecure: false,
        });
        config
    }

    #[cfg(feature = "device-audio")]
    #[derive(Clone, Default)]
    struct SharedSink {
        chunks: Arc<Mutex<Vec<SynthesizedAudio>>>,
        interrupted: Arc<AtomicUsize>,
    }

    #[cfg(feature = "device-audio")]
    impl AudioSink for SharedSink {
        fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            self.chunks.lock().expect("chunks").push(audio);
            Ok(())
        }

        fn interrupt(&mut self) {
            self.interrupted.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[cfg(feature = "device-audio")]
    fn run_session(
        fake: &FakeMoshi,
        sink: SharedSink,
        mode: LoopMode,
    ) -> crate::error::Result<crate::pipeline::LoopReport> {
        let (tx, _rx) = std::sync::mpsc::channel();
        Session::from_config(&s2s_config(&fake.base_url))
            .with_events(Some(tx))
            .with_mode(mode)
            .with_sink(sink)
            .with_frames(scripted_frames(1, 24, 0))
            .run(Cancel::new())
    }

    #[test]
    fn chat_url_is_api_chat_on_the_origin() {
        let url = moshi_ws_url("http://127.0.0.1:8998").expect("url");
        assert_eq!(url.as_str(), "ws://127.0.0.1:8998/api/chat");
        let url = moshi_ws_url("https://127.0.0.1:8998/ignored").expect("url");
        assert_eq!(url.as_str(), "wss://127.0.0.1:8998/api/chat");
    }

    #[test]
    fn tls_insecure_is_rejected_off_loopback() {
        let err = match MoshiWsS2s::connect(
            &S2sConfig {
                base_url: "https://moshi.example.test".into(),
                tls_insecure: true,
            },
            None,
            &Cancel::new(),
        ) {
            Ok(_) => panic!("remote insecure should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("tls_insecure"), "{err}");
        assert!(err.to_string().contains("loopback"), "{err}");
    }

    #[test]
    fn connect_timeout_matches_the_crate() {
        assert_eq!(CONNECT_TIMEOUT, crate::openai::CONNECT_TIMEOUT);
        assert!(CONNECT_TIMEOUT > IO_TICK);
    }

    #[test]
    fn delayed_websocket_upgrade_survives_rtt_above_io_tick() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            thread::sleep(IO_TICK + Duration::from_millis(40));
            let mut ws = tungstenite::accept(stream).expect("ws");
            ws.send(Message::Binary(
                MoshiMessage::Handshake {
                    protocol_version: PROTOCOL_VERSION,
                    model_version: MODEL_VERSION,
                }
                .encode(),
            ))
            .expect("handshake");
            let _ = ws.read();
        });
        MoshiWsS2s::connect(
            &S2sConfig {
                base_url: format!("http://{addr}"),
                tls_insecure: false,
            },
            None,
            &Cancel::new(),
        )
        .expect("upgrade must not use the 20 ms I/O tick");
    }

    #[test]
    fn python_and_rust_handshakes_are_accepted() {
        assert!(!incoming_handshake(&[0]).expect("python").expect("ok"));
        assert!(incoming_handshake(&[0, 0, 0, 0, 0, 0, 0, 0, 0])
            .expect("rust")
            .expect("ok"));
        assert_eq!(incoming_handshake(&[1, 2]).expect("audio"), None);
        let err = incoming_handshake(&[0, 1]).expect_err("short");
        assert!(err.to_string().contains("handshake"), "{err}");
    }

    #[test]
    fn upgrade_timeout_names_server_busy() {
        let err = upgrade_busy_error();
        let text = err.to_string();
        assert!(text.contains("timed out"), "{err}");
        assert!(text.contains("server busy"), "{err}");
    }

    #[test]
    fn python_handshake_gets_no_client_reply() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let replies = Arc::new(AtomicUsize::new(0));
        let replies_h = Arc::clone(&replies);
        thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            let mut ws = tungstenite::accept(stream).expect("ws");
            ws.send(Message::Binary(vec![0])).expect("python handshake");
            let _ = ws
                .get_mut()
                .set_read_timeout(Some(Duration::from_millis(250)));
            loop {
                match ws.read() {
                    Ok(Message::Binary(bytes)) if bytes.first() == Some(&0) => {
                        replies_h.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        });
        let client = MoshiWsS2s::connect(
            &S2sConfig {
                base_url: format!("http://{addr}"),
                tls_insecure: false,
            },
            None,
            &Cancel::new(),
        )
        .expect("python handshake");
        thread::sleep(Duration::from_millis(150));
        drop(client);
        assert_eq!(
            replies.load(Ordering::SeqCst),
            0,
            "Python MT=0 must not get a 9-byte client reply"
        );
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn session_loopback_handshake_echo_text_and_error() {
        let _env = ENV_LOCK.lock().expect("env");
        std::env::remove_var(S2S_API_KEY_ENV);

        let echo = start_fake(FakeMode::Echo);
        std::env::set_var(S2S_API_KEY_ENV, "secret-s2s");
        let sink = SharedSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let report = run_session(&echo, sink, LoopMode::StopAfterTurns(1)).expect("echo session");
        assert_eq!(report.tasks_exited, 3);
        wait_flag(
            &echo.seen_handshake,
            true,
            "client must reply to the Rust handshake",
        );
        assert_eq!(
            echo.auth.lock().expect("auth").as_deref(),
            Some("Bearer secret-s2s")
        );
        let played = chunks.lock().expect("chunks").clone();
        assert!(
            played
                .iter()
                .any(|chunk| chunk.samples.iter().any(|s| *s != 0)),
            "echoed audio should reach the Session sink, got {played:?}"
        );
        std::env::remove_var(S2S_API_KEY_ENV);

        let text = start_fake(FakeMode::TextThenEcho);
        let sink = SharedSink::default();
        run_session(&text, sink, LoopMode::StopAfterTurns(1)).expect("text session");
        assert!(
            text.text_sent.load(Ordering::SeqCst),
            "fake server should emit a text frame"
        );

        let boom = start_fake(FakeMode::ErrorAfterHandshake);
        let err = Session::from_config(&s2s_config(&boom.base_url))
            .with_mode(LoopMode::UntilInputEnds)
            .with_sink(CollectingSink::default())
            .with_frames(scripted_frames(1, 4, 0))
            .run(Cancel::new())
            .expect_err("error session");
        assert!(err.to_string().contains("moshi boom"), "{err}");
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn connect_name_is_moshi() {
        let _env = ENV_LOCK.lock().expect("env");
        std::env::remove_var(S2S_API_KEY_ENV);
        let fake = start_fake(FakeMode::Echo);
        let client = MoshiWsS2s::connect(
            &S2sConfig {
                base_url: fake.base_url.clone(),
                tls_insecure: false,
            },
            None,
            &Cancel::new(),
        )
        .expect("connect");
        assert_eq!(DuplexS2s::name(&client), "moshi");
        wait_flag(
            &fake.seen_handshake,
            true,
            "client must reply to the Rust handshake",
        );
        drop(client);
    }

    #[cfg(feature = "device-audio")]
    fn wait_flag(flag: &AtomicBool, expected: bool, msg: &str) {
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if flag.load(Ordering::SeqCst) == expected {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(flag.load(Ordering::SeqCst), expected, "{msg}");
    }
}
