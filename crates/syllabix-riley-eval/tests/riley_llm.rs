//! `RileyLlm` against the local mock endpoint, with an in-memory tool
//! executor (no Postgres): tool rounds, rejected calls, and every failure
//! path. Failures are `Error::Provider`, so the pipeline skips the turn and
//! the session fails closed.

// Linux-gated per #329 A2 (the weekly Windows/macOS matrix skips this file).
#![cfg(target_os = "linux")]

mod support;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use support::{say, tool_calls, MockApi};
use syllabix_core::types::{ToolResult, Transcript, TurnId};
use syllabix_core::{Cancel, Error, Llm, TokenChunk};
use syllabix_riley_eval::llm::{RileyLlm, RileyLlmSettings, PROVIDER_NAME};
use zeroize::Zeroizing;

fn llm(settings: RileyLlmSettings, seen: Arc<Mutex<Vec<String>>>) -> RileyLlm {
    RileyLlm::new(
        settings,
        Zeroizing::new("llm-test-key".into()),
        Box::new(move |call| {
            seen.lock().unwrap().push(call.name.clone());
            ToolResult {
                tool_call_id: call.id.clone(),
                content: json!({"id": "c_alice_debit"}).to_string(),
                ok: true,
            }
        }),
    )
}

fn user(text: &str) -> Transcript {
    Transcript {
        turn: TurnId(4),
        text: text.into(),
        language: "en".into(),
    }
}

fn generate(llm: &mut RileyLlm, cancel: &Cancel) -> (Result<(), Error>, Vec<TokenChunk>) {
    let mut tokens = Vec::new();
    let result = llm.generate(&[], &user("hi"), cancel, &mut |token| {
        tokens.push(token);
        Ok(())
    });
    (result, tokens)
}

fn provider_message(result: Result<(), Error>) -> String {
    match result {
        Err(Error::Provider { provider, message }) => {
            assert_eq!(provider, PROVIDER_NAME);
            message
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
}

/// Serve one raw HTTP response, then close.
fn one_shot(response: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 65536];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(response.as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}/v1")
}

#[test]
fn tool_rounds_run_until_the_model_speaks() {
    let api = MockApi::start(
        &[],
        vec![
            tool_calls(&[
                ("bad", "display_card_info_by_last4", json!("not an object")),
                (
                    "call_1",
                    "display_card_info_by_last4",
                    json!({"last4": "1111"}),
                ),
            ]),
            say("  Found it.  "),
        ],
        0,
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut llm = llm(RileyLlmSettings::new(&api.base, "m"), Arc::clone(&seen));
    let (result, tokens) = generate(&mut llm, &Cancel::new());
    result.unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].text, "Found it.");
    assert!(tokens[0].is_last);
    assert_eq!(tokens[0].turn, TurnId(4));
    assert_eq!(*seen.lock().unwrap(), ["display_card_info_by_last4"]);
    let kinds: Vec<(String, String)> = llm
        .take_tool_events()
        .into_iter()
        .map(|event| (event.kind, event.name))
        .collect();
    assert_eq!(
        kinds,
        [
            ("rejected".to_string(), String::new()),
            ("call".to_string(), "display_card_info_by_last4".to_string()),
            (
                "result".to_string(),
                "display_card_info_by_last4".to_string()
            ),
        ]
    );
    assert!(llm.take_tool_events().is_empty(), "events are drained");
    let state = api.state.lock().unwrap();
    let second: &Vec<Value> = state.llm_requests[1]["messages"].as_array().unwrap();
    assert_eq!(second[3]["tool_call_id"], "bad");
    assert_eq!(
        second[3]["content"],
        "{\"error\":\"tool call arguments must be a JSON object\"}"
    );
    assert_eq!(second[4]["tool_call_id"], "call_1");
}

#[test]
fn http_errors_bad_bodies_and_empty_replies_are_provider_errors() {
    let api = MockApi::start(&[], vec![json!("not a message"), say("   ")], 0);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut llm = llm(RileyLlmSettings::new(&api.base, "m"), seen);
    let cancel = Cancel::new();
    assert_eq!(
        provider_message(generate(&mut llm, &cancel).0),
        "response has no choices[0].message"
    );
    assert_eq!(
        provider_message(generate(&mut llm, &cancel).0),
        "model returned an empty reply"
    );
    // Script is empty now: HTTP 500.
    assert_eq!(
        provider_message(generate(&mut llm, &cancel).0),
        "chat completions returned HTTP 500"
    );
    // Failed turns stay in the log so the next request is well formed.
    let state = api.state.lock().unwrap();
    let last = state.llm_requests.last().unwrap()["messages"]
        .as_array()
        .unwrap();
    assert_eq!(last.len(), 4, "system + three user turns");
}

#[test]
fn invalid_json_and_connection_failures_never_leak_the_key() {
    let base = one_shot("HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nnot");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut bad_json = llm(RileyLlmSettings::new(&base, "m"), Arc::clone(&seen));
    let message = provider_message(generate(&mut bad_json, &Cancel::new()).0);
    assert_eq!(message, "chat completions returned invalid JSON");

    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let mut refused = llm(
        RileyLlmSettings::new(&format!("http://127.0.0.1:{port}/v1"), "m"),
        seen,
    );
    let message = provider_message(generate(&mut refused, &Cancel::new()).0);
    assert!(
        message.starts_with("chat completions request failed: "),
        "{message}"
    );
    assert!(!message.contains("llm-test-key"));
}

#[test]
fn round_limit_and_cancellation_end_the_turn() {
    let api = MockApi::start(
        &[],
        vec![
            tool_calls(&[("a", "display_user_info", json!({"user_id": "u"}))]),
            tool_calls(&[("b", "display_user_info", json!({"user_id": "u"}))]),
        ],
        0,
    );
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut settings = RileyLlmSettings::new(&api.base, "m");
    settings.max_tool_rounds = 2;
    let mut limited = llm(settings, Arc::clone(&seen));
    assert_eq!(
        provider_message(generate(&mut limited, &Cancel::new()).0),
        "no reply after 2 tool rounds"
    );
    assert_eq!(limited.take_tool_events().last().unwrap().kind, "limit");

    let shut = Cancel::new();
    shut.shutdown();
    assert!(matches!(
        generate(&mut limited, &shut).0,
        Err(Error::Cancelled)
    ));

    // A server that never answers: barge-in (generation cancel) wins.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        silent.local_addr().unwrap().port()
    );
    let mut hanging = llm(RileyLlmSettings::new(&base, "m"), Arc::clone(&seen));
    let cancel = Cancel::new();
    let barge = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        barge.cancel_generation();
    });
    assert!(matches!(
        generate(&mut hanging, &cancel).0,
        Err(Error::Cancelled)
    ));

    // Same server with a short budget: the turn times out.
    let mut settings = RileyLlmSettings::new(&base, "m");
    settings.request_timeout = Duration::from_millis(200);
    let mut slow = llm(settings, seen);
    let message = provider_message(generate(&mut slow, &Cancel::new()).0);
    assert!(
        message == "chat completions request timed out"
            || message.starts_with("chat completions request failed: "),
        "{message}"
    );
    drop(silent);
}
