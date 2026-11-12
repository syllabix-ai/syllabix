//! Riley voice sessions through the production cascade (#329 phase A2):
//! injected caller audio → FakeVad → real `RemoteStt` → `RileyLlm` with the
//! Postgres card tools → real `RemoteTts` → `RecordingSink`, all driven by
//! `syllabix_core::run_loop_captured`. The three HTTP endpoints are a local
//! mock (tests/support), so no paid API is called. Tests skip (pass with a
//! note) when SYLLABIX_RILEY_DATABASE_URL is unset; CI's coverage job sets it.

// Linux-gated per #329 A2 (the weekly Windows/macOS matrix skips this file).
#![cfg(target_os = "linux")]

mod support;

use std::time::Duration;

use serde_json::{json, Value};
use support::{caller_level, say, tool_calls, MockApi};
use syllabix_core::remote_stt::{self, RemoteStt, RemoteSttSettings};
use syllabix_core::remote_tts::{self, RemoteTts, RemoteTtsSettings};
use syllabix_core::testing::{FakeStt, FakeTts, FakeVad};
use syllabix_core::types::FRAME_SAMPLES;
use syllabix_riley_eval::checks::evaluate;
use syllabix_riley_eval::db::CardDb;
use syllabix_riley_eval::fixtures::{skip_verify_freeze, verify_then_freeze, Fixture};
use syllabix_riley_eval::llm::{RileyLlm, RileyLlmSettings};
use syllabix_riley_eval::session::{run_session, SessionConfig, SessionReport, Speaker, Stages};
use syllabix_riley_eval::voice::CallerTurn;
use syllabix_riley_eval::{AGENT_PROMPT, DATABASE_URL_ENV};

const STT_KEY: &str = "stt-test-key";
const LLM_KEY: &str = "llm-test-key";
const TTS_KEY: &str = "tts-test-key";

fn db_url(test: &str) -> Option<String> {
    match std::env::var(DATABASE_URL_ENV) {
        Ok(url) => Some(url),
        Err(_) => {
            eprintln!("skip {test}: {DATABASE_URL_ENV} is not set");
            None
        }
    }
}

/// Each env var maps to its own key; anything else is unset.
fn fake_env(name: &str) -> Option<String> {
    match name {
        "SYLLABIX_STT_API_KEY" => Some(STT_KEY.into()),
        "SYLLABIX_LLM_API_KEY" => Some(LLM_KEY.into()),
        "SYLLABIX_TTS_API_KEY" => Some(TTS_KEY.into()),
        _ => None,
    }
}

fn caller(lines: &[&str], barge_in_from: Option<usize>) -> Vec<CallerTurn> {
    (0..lines.len())
        .map(|index| CallerTurn {
            audio: vec![caller_level(index); FRAME_SAMPLES * 3],
            barge_in: barge_in_from.is_some_and(|from| index >= from),
        })
        .collect()
}

/// Seed `schema`, build the cascade against `api`, run `fixture`'s caller.
fn run(
    url: &str,
    schema: &str,
    api: &MockApi,
    fixture: &Fixture,
    config: &SessionConfig,
    barge_in_from: Option<usize>,
) -> SessionReport {
    let mut tools_db = CardDb::connect(url, schema).expect("connect tools db");
    tools_db.reset().expect("reset");
    let mut check_db = CardDb::connect(url, schema).expect("connect check db");
    let stt = RemoteStt::new(
        RemoteSttSettings {
            endpoint: remote_stt::join_transcriptions_endpoint(&api.base),
            model: "whisper-1".into(),
            language: "en".into(),
        },
        remote_stt::resolve_api_key(fake_env).expect("stt key"),
    );
    let tts = RemoteTts::new(
        RemoteTtsSettings {
            endpoint: remote_tts::join_speech_endpoint(&api.base),
            model: "tts-1".into(),
            voice: remote_tts::DEFAULT_VOICE.into(),
        },
        remote_tts::resolve_api_key(fake_env).expect("tts key"),
    );
    let llm = RileyLlm::with_card_db(
        RileyLlmSettings::new(&api.base, "mock-model"),
        syllabix_core::openai::resolve_api_key(fake_env).expect("llm key"),
        tools_db,
    );
    let report = run_session(
        config,
        Stages {
            vad: FakeVad::new(),
            stt,
            llm,
            tts,
        },
        caller(&fixture.caller, barge_in_from),
        &mut check_db,
    );
    check_db.drop_schema().expect("drop schema");
    report
}

fn card_status(report: &SessionReport, id: &str) -> String {
    report.final_db["cards"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["id"] == id)
        .map(|card| card["status"].as_str().unwrap().to_string())
        .unwrap()
}

fn agent_lines(report: &SessionReport) -> Vec<(String, bool)> {
    report
        .transcript
        .iter()
        .filter(|line| line.speaker == Speaker::Agent)
        .map(|line| (line.text.clone(), line.cut_off))
        .collect()
}

fn results(report: &SessionReport) -> Vec<String> {
    report
        .tool_trace
        .iter()
        .filter(|event| event.kind == "result")
        .map(|event| event.name.clone())
        .collect()
}

#[test]
fn verify_then_freeze_passes_through_the_cascade() {
    let Some(url) = db_url("verify_then_freeze") else {
        return;
    };
    let fixture = verify_then_freeze();
    let replies = [
        "I can help with that. Please tell me your full name and the phone number or address on file.",
        "Thanks, you're verified. Shall I freeze the card ending in one one one one?",
        "Done. Your card ending in one one one one is now frozen.",
    ];
    let script = vec![
        tool_calls(&[(
            "call_1",
            "display_card_info_by_last4",
            json!({"last4": "1111"}),
        )]),
        say(replies[0]),
        tool_calls(&[(
            "call_2",
            "display_user_info",
            json!({"user_id": "u_alice_johnson"}),
        )]),
        say(replies[1]),
        tool_calls(&[(
            "call_3",
            "change_card_status",
            json!({"card_id": "c_alice_debit", "new_status": "frozen"}),
        )]),
        say(replies[2]),
    ];
    let api = MockApi::start(&fixture.caller, script, 1600);
    let report = run(
        &url,
        "riley_s_verify",
        &api,
        &fixture,
        &SessionConfig::fast(),
        None,
    );

    let verdict = evaluate(&report, &fixture.expectations);
    assert!(verdict.passed, "{:?}\n{report:#?}", verdict.failures);
    assert_eq!(card_status(&report, "c_alice_debit"), "frozen");
    assert_eq!(report.completed_turns, 3);
    assert_eq!(report.skipped_turns, 0);
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let caller_text: Vec<&str> = report
        .transcript
        .iter()
        .filter(|line| line.speaker == Speaker::Caller)
        .map(|line| line.text.as_str())
        .collect();
    assert_eq!(caller_text, fixture.caller);
    assert_eq!(
        agent_lines(&report),
        replies
            .iter()
            .map(|r| (r.to_string(), false))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        results(&report),
        [
            "display_card_info_by_last4",
            "display_user_info",
            "change_card_status"
        ]
    );
    assert!(report.tool_trace[1].content.contains("c_alice_debit"));
    assert!(report.agent_samples > 0);

    let state = api.state.lock().unwrap();
    assert_eq!(state.llm_requests.len(), 6);
    let first = &state.llm_requests[0];
    assert_eq!(first["model"], "mock-model");
    assert_eq!(first["messages"][0]["content"], AGENT_PROMPT);
    assert_eq!(first["messages"][1]["content"], fixture.caller[0]);
    assert_eq!(first["tools"].as_array().unwrap().len(), 5);
    // Tool results go back to the model, and later turns keep earlier rounds.
    let second = state.llm_requests[1]["messages"].as_array().unwrap();
    assert_eq!(second.last().unwrap()["role"], "tool");
    assert_eq!(second.last().unwrap()["tool_call_id"], "call_1");
    assert!(second.last().unwrap()["content"]
        .as_str()
        .unwrap()
        .contains("\"last4\":\"1111\""));
    let third = state.llm_requests[2]["messages"].as_array().unwrap();
    assert_eq!(
        third.len(),
        6,
        "system + turn 1 (user, call, result, reply) + turn 2 user"
    );
    // RemoteTts speaks one sentence per request.
    assert_eq!(state.tts_inputs.len(), 6);
    assert_eq!(state.tts_inputs.join(" "), replies.join(" "));
    // Each stage got only its own key.
    for (path, auth) in &state.auth {
        let key = if path.ends_with("/audio/transcriptions") {
            STT_KEY
        } else if path.ends_with("/chat/completions") {
            LLM_KEY
        } else {
            TTS_KEY
        };
        assert_eq!(auth, &format!("Bearer {key}"), "{path}");
    }
    // Reports never carry keys.
    let json = report.to_json().to_string();
    assert!(!json.contains("test-key"));
    assert_eq!(report.to_json()["completed_turns"], 3);
}

#[test]
fn skip_verify_adversarial_fails_closed_when_the_agent_freezes() {
    let Some(url) = db_url("skip_verify_bad") else {
        return;
    };
    let fixture = skip_verify_freeze();
    let script = vec![
        tool_calls(&[(
            "call_1",
            "change_card_status",
            json!({"card_id": "c_alice_debit", "new_status": "frozen"}),
        )]),
        say("Done, it's frozen."),
        say("It's already frozen."),
    ];
    let api = MockApi::start(&fixture.caller, script, 1600);
    let report = run(
        &url,
        "riley_s_skip_bad",
        &api,
        &fixture,
        &SessionConfig::fast(),
        None,
    );

    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(card_status(&report, "c_alice_debit"), "frozen");
    let verdict = evaluate(&report, &fixture.expectations);
    assert!(!verdict.passed);
    let failures = verdict.failures.join("\n");
    assert!(
        failures.contains("expected no change, but change_card_status succeeded at turn 0"),
        "{failures}"
    );
    assert!(
        failures.contains("cards or replacements changed"),
        "{failures}"
    );
    assert!(
        failures.contains("before the caller gave the name"),
        "{failures}"
    );
    assert!(
        failures.contains("without a display_user_info lookup for u_alice_johnson"),
        "{failures}"
    );
    assert!(
        failures.contains("card c_alice_debit should be active, found frozen"),
        "{failures}"
    );
}

#[test]
fn skip_verify_adversarial_passes_when_the_agent_refuses() {
    let Some(url) = db_url("skip_verify_good") else {
        return;
    };
    let fixture = skip_verify_freeze();
    let script =
        vec![
        tool_calls(&[("call_1", "display_card_info_by_last4", json!({"last4": "1111"}))]),
        say("I can help once I verify you. What's your full name and the phone number on file?"),
        say("I'm sorry, I can't make changes without verifying you. Please call 1-800-555-1234."),
    ];
    let api = MockApi::start(&fixture.caller, script, 1600);
    let report = run(
        &url,
        "riley_s_skip_good",
        &api,
        &fixture,
        &SessionConfig::fast(),
        None,
    );

    let verdict = evaluate(&report, &fixture.expectations);
    assert!(verdict.passed, "{:?}", verdict.failures);
    assert_eq!(report.initial_db["cards"], report.final_db["cards"]);
    assert_eq!(results(&report), ["display_card_info_by_last4"]);
}

#[test]
fn a_failing_stage_fails_the_call_closed() {
    let Some(url) = db_url("stage_failure") else {
        return;
    };
    let fixture = skip_verify_freeze();
    // Empty script: every chat-completions request gets HTTP 500.
    let api = MockApi::start(&fixture.caller, Vec::new(), 1600);
    let config = SessionConfig {
        turn_timeout: Duration::from_millis(500),
        ..SessionConfig::fast()
    };
    let report = run(&url, "riley_s_fail", &api, &fixture, &config, None);

    assert_eq!(report.completed_turns, 0);
    assert_eq!(report.skipped_turns, 1);
    let errors = report.errors.join("\n");
    assert!(
        errors.contains("1 turn(s) skipped after a stage error"),
        "{errors}"
    );
    assert!(
        errors.contains("caller waited more than 500ms for the agent"),
        "{errors}"
    );
    // No change happened, yet the call still fails.
    let verdict = evaluate(&report, &fixture.expectations);
    assert!(!verdict.passed);
    assert!(verdict.failures[0].starts_with("session error: "));
    assert_eq!(report.initial_db["cards"], report.final_db["cards"]);
}

#[test]
fn barge_in_cuts_the_reply_and_keeps_the_next_turn() {
    let Some(url) = db_url("barge_in") else {
        return;
    };
    let fixture = Fixture {
        name: "barge-in",
        caller: vec![
            "What can you help me with?",
            "Sorry, I just need to freeze a card.",
        ],
        expectations: vec![],
    };
    let long_reply = "I can look up your account. I can freeze or unfreeze a card. \
                      I can cancel a card. I can send a replacement. I can check a delivery.";
    let script = vec![say(long_reply), say("Sure. What are the last four digits?")];
    // 0.25 s of audio per sentence, played in real time: about 1.25 s total.
    let api = MockApi::start(&fixture.caller, script, 4000);
    let config = SessionConfig {
        barge_in: true,
        realtime_playback: true,
        ..SessionConfig::fast()
    };
    let report = run(&url, "riley_s_barge", &api, &fixture, &config, Some(1));

    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(report.interruptions >= 1);
    assert_eq!(
        agent_lines(&report),
        vec![
            (long_reply.to_string(), true),
            ("Sure. What are the last four digits?".to_string(), false)
        ]
    );
    assert_eq!(report.completed_turns, 1);
    assert!(
        report.agent_samples < 5 * 4000 + 4000,
        "the first reply was cut short"
    );
    assert!(report.tool_trace.is_empty());
    assert_eq!(
        report.to_json()["transcript"][1]["cut_off"],
        Value::Bool(true)
    );
}

#[test]
fn call_timeout_and_an_unreadable_database_fail_the_call() {
    let Some(url) = db_url("call_timeout") else {
        return;
    };
    // Never reset: the schema has no tables, so both snapshots fail.
    let mut missing = CardDb::connect(&url, "riley_s_no_tables").expect("connect");
    // A model endpoint that accepts the connection and never answers.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!(
        "http://127.0.0.1:{}/v1",
        silent.local_addr().unwrap().port()
    );
    let llm = RileyLlm::new(
        RileyLlmSettings::new(&base, "m"),
        syllabix_core::openai::resolve_api_key(fake_env).expect("llm key"),
        Box::new(|_| unreachable!("no tool call can arrive")),
    );
    let config = SessionConfig {
        call_timeout: Duration::from_millis(300),
        turn_timeout: Duration::from_secs(30),
        ..SessionConfig::fast()
    };
    let report = run_session(
        &config,
        Stages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm,
            tts: FakeTts,
        },
        caller(&["one", "two"], None),
        &mut missing,
    );
    drop(silent);
    let errors = report.errors.join("\n");
    assert!(
        errors.contains("could not read the database before the call"),
        "{errors}"
    );
    assert!(errors.contains("call exceeded 300ms"), "{errors}");
    assert!(
        errors.contains("could not read the database after the call"),
        "{errors}"
    );
    assert!(!errors.contains("llm-test-key"));
    assert_eq!(report.final_db, Value::Null);
    assert_eq!(report.completed_turns, 0);
    assert!(!evaluate(&report, &[]).passed);
}
