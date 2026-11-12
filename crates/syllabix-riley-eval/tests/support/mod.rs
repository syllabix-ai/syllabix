//! Local stand-in for the three OpenAI-compatible endpoints the cascade
//! calls (`/audio/transcriptions`, `/chat/completions`, `/audio/speech`).
//! CI never calls a paid API: the real `RemoteStt`, `RileyLlm`, and
//! `RemoteTts` clients talk to this server over loopback HTTP.
//!
//! STT: caller turn `i` is injected as constant samples `1000 + i`; the
//! server reads the first non-zero sample of the uploaded WAV and returns
//! that caller line. LLM: replies are popped from a script (HTTP 500 when
//! the script is empty). TTS: every request returns a 16 kHz mono WAV.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use serde_json::{json, Value};
use tiny_http::{Header, Response, Server};

/// Sample value for caller turn `index`.
pub fn caller_level(index: usize) -> i16 {
    1000 + i16::try_from(index).expect("small index")
}

#[derive(Default)]
pub struct MockState {
    pub caller_lines: Vec<String>,
    pub llm_script: VecDeque<Value>,
    /// Parsed chat-completions request bodies, in order.
    pub llm_requests: Vec<Value>,
    /// `(path, Authorization header)` for every request.
    pub auth: Vec<(String, String)>,
    /// `input` of every speech request.
    pub tts_inputs: Vec<String>,
    /// Samples per speech response.
    pub tts_samples: usize,
}

pub struct MockApi {
    pub base: String,
    pub state: Arc<Mutex<MockState>>,
    server: Arc<Server>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for MockApi {
    fn drop(&mut self) {
        self.server.unblock();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Assistant message that only calls tools: `(id, name, arguments)`.
pub fn tool_calls(calls: &[(&str, &str, Value)]) -> Value {
    let calls: Vec<Value> = calls
        .iter()
        .map(|(id, name, args)| {
            json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": args.to_string()},
            })
        })
        .collect();
    json!({"role": "assistant", "content": null, "tool_calls": calls})
}

/// Assistant message with spoken text.
pub fn say(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

fn wav(samples: usize, level: i16) -> Vec<u8> {
    let data_len = u32::try_from(samples * 2).expect("small wav");
    let mut out = Vec::with_capacity(44 + samples * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for _ in 0..samples {
        out.extend_from_slice(&level.to_le_bytes());
    }
    out
}

/// First non-zero sample in the `data` chunk of the first WAV in `body`.
fn first_level(body: &[u8]) -> Option<i16> {
    let riff = body.windows(4).position(|w| w == b"RIFF")?;
    let data = riff + body[riff..].windows(4).position(|w| w == b"data")? + 8;
    body[data..]
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .find(|sample| *sample != 0)
}

fn handle(state: &Mutex<MockState>, mut request: tiny_http::Request) {
    let path = request.url().to_string();
    let auth = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.to_string())
        .unwrap_or_default();
    let mut body = Vec::new();
    let _ = request.as_reader().read_to_end(&mut body);
    let mut state = state.lock().expect("mock state");
    state.auth.push((path.clone(), auth));
    let json_header = Header::from_bytes("Content-Type", "application/json").expect("header");
    let response = if path.ends_with("/audio/transcriptions") {
        let text = first_level(&body)
            .and_then(|level| usize::try_from(level - 1000).ok())
            .and_then(|index| state.caller_lines.get(index).cloned())
            .unwrap_or_default();
        Response::from_data(json!({ "text": text }).to_string().into_bytes())
            .with_header(json_header)
    } else if path.ends_with("/chat/completions") {
        let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        state.llm_requests.push(parsed);
        match state.llm_script.pop_front() {
            Some(message) => Response::from_data(
                json!({"id": "mock", "choices": [{"index": 0, "message": message}]})
                    .to_string()
                    .into_bytes(),
            )
            .with_header(json_header),
            None => {
                Response::from_data(b"{\"error\":\"script empty\"}".to_vec()).with_status_code(500)
            }
        }
    } else if path.ends_with("/audio/speech") {
        let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let input = parsed["input"].as_str().unwrap_or_default().to_string();
        state.tts_inputs.push(input);
        Response::from_data(wav(state.tts_samples, 500))
            .with_header(Header::from_bytes("Content-Type", "audio/wav").expect("header"))
    } else {
        Response::from_data(Vec::new()).with_status_code(404)
    };
    drop(state);
    let _ = request.respond(response);
}

impl MockApi {
    pub fn start(caller_lines: &[&str], llm_script: Vec<Value>, tts_samples: usize) -> Self {
        let server = Arc::new(Server::http("127.0.0.1:0").expect("bind mock api"));
        let port = server.server_addr().to_ip().expect("ip addr").port();
        let state = Arc::new(Mutex::new(MockState {
            caller_lines: caller_lines.iter().map(|line| line.to_string()).collect(),
            llm_script: llm_script.into(),
            tts_samples,
            ..MockState::default()
        }));
        let thread = {
            let server = Arc::clone(&server);
            let state = Arc::clone(&state);
            std::thread::spawn(move || {
                for request in server.incoming_requests() {
                    handle(&state, request);
                }
            })
        };
        Self {
            base: format!("http://127.0.0.1:{port}/v1"),
            state,
            server,
            thread: Some(thread),
        }
    }
}
