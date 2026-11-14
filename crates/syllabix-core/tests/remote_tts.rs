//! Session loopback for the remote TTS client (needs `Session` / testing fakes).

#![cfg(feature = "cloud")]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use syllabix_core::openai::OpenAiTimeouts;
use syllabix_core::remote_tts::{RemoteTts, RemoteTtsSettings, DEFAULT_VOICE};
use syllabix_core::testing::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeVad};
use syllabix_core::{Cancel, Session};
use zeroize::Zeroizing;

const TEST_KEY: &str = "sk-tts-test";

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

fn openai_pcm_bytes() -> Vec<u8> {
    (1..=48)
        .map(|i| i * 200)
        .flat_map(|s: i16| s.to_le_bytes())
        .collect()
}

#[test]
fn session_loopback_speaks_one_scripted_turn_without_a_mic() {
    let (endpoint, handle) = serve_one(|stream, request| {
        let text = String::from_utf8_lossy(request);
        assert!(text.contains("POST /v1/audio/speech"), "{text}");
        assert!(text.contains("\"stream_format\":\"audio\""), "{text}");
        write_bytes(
            stream,
            "HTTP/1.1 200 OK",
            &[("Content-Type", "application/octet-stream")],
            &openai_pcm_bytes(),
        )
        .expect("write");
    });
    let sink = CollectingSink::default();
    let report = Session::v0()
        .with_vad(FakeVad::new())
        .with_stt(FakeStt)
        .with_llm(FakeLlm::new())
        .with_tts(test_tts(endpoint))
        .with_sink(sink)
        .with_frames(scripted_frames(1, 2, 1))
        .run(Cancel::new())
        .expect("session loopback");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.skipped_turns, 0);
    let _ = handle.join();
}
