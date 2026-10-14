//! Process tests for the Goose ACP client. A tiny Python sidecar speaks ACP.

use super::*;
use crate::cancel::Cancel;
use crate::providers::Llm;
use crate::types::TurnId;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use zeroize::Zeroizing;

const FAKE_ACP: &str = r#"#!/usr/bin/env python3
import json, os, sys, time

root = os.environ.get("GOOSE_PATH_ROOT", "")
if root:
    os.makedirs(root, exist_ok=True)
    env = {k: os.environ.get(k, "") for k in (
        "GOOSE_PROVIDER", "GOOSE_MODEL", "OPENAI_HOST", "OPENAI_BASE_PATH",
        "OPENAI_API_KEY", "GOOSE_PATH_ROOT", "GOOSE_DISABLE_KEYRING", "GOOSE_MODE",
    )}
    env["argv"] = sys.argv
    with open(os.path.join(root, "spawn-env.json"), "w") as f:
        json.dump(env, f)

state = {"session_id": "sess-1", "cwd": os.getcwd()}

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def rpc_result(msg_id, result):
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})

def handle(msg):
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    if method == "initialize":
        caps = (params.get("clientCapabilities") or {})
        assert caps.get("terminal") is True, caps
        fs = caps.get("fs") or {}
        assert not fs.get("readTextFile"), fs
        assert not fs.get("writeTextFile"), fs
        rpc_result(msg_id, {"protocolVersion": 1, "agentCapabilities": {}})
    elif method == "initialized":
        pass
    elif method == "session/new":
        state["cwd"] = params.get("cwd") or os.getcwd()
        meta = params.get("_meta") or {}
        ext = meta.get("enabledExtensions") or []
        assert ext and ext[0].get("name") == "developer", ext
        assert ext[0].get("available_tools") == ["shell"], ext
        assert "availableTools" not in ext[0], ext
        rpc_result(msg_id, {"sessionId": state["session_id"]})
    elif method == "_goose/unstable/session/system-prompt/set":
        assert params.get("mode") == "set"
        assert params.get("key") == "syllabix"
        rpc_result(msg_id, {})
    elif method == "session/prompt":
        text = ""
        for block in params.get("prompt") or []:
            text += block.get("text") or ""
        if "escape-cwd" in text:
            send({
                "jsonrpc": "2.0",
                "id": 9101,
                "method": "terminal/create",
                "params": {
                    "sessionId": state["session_id"],
                    "command": "echo escaped",
                    "cwd": "/tmp",
                },
            })
            created = json.loads(sys.stdin.readline())
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "refused-cwd"},
                    },
                },
            })
            if root:
                with open(os.path.join(root, "cwd-refuse.json"), "w") as f:
                    json.dump(created, f)
        elif "use-shell" in text or "slow-shell" in text:
            call_id = "call-shell-1"
            cmd = "sleep 1; echo goose-shell" if "slow-shell" in text else "echo goose-shell"
            reply = "ran-slow-shell" if "slow-shell" in text else "ran-shell"
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": call_id,
                        "title": "shell · " + cmd,
                        "status": "pending",
                    },
                },
            })
            send({
                "jsonrpc": "2.0",
                "id": 9001,
                "method": "terminal/create",
                "params": {
                    "sessionId": state["session_id"],
                    "command": cmd,
                    "cwd": state["cwd"],
                    "outputByteLimit": 100000,
                },
            })
            created = json.loads(sys.stdin.readline())
            tid = (created.get("result") or {}).get("terminalId")
            send({
                "jsonrpc": "2.0",
                "id": 9002,
                "method": "terminal/wait_for_exit",
                "params": {"sessionId": state["session_id"], "terminalId": tid},
            })
            sys.stdin.readline()
            send({
                "jsonrpc": "2.0",
                "id": 9003,
                "method": "terminal/output",
                "params": {"sessionId": state["session_id"], "terminalId": tid},
            })
            out = json.loads(sys.stdin.readline())
            send({
                "jsonrpc": "2.0",
                "id": 9004,
                "method": "terminal/release",
                "params": {"sessionId": state["session_id"], "terminalId": tid},
            })
            sys.stdin.readline()
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": call_id,
                        "title": "shell · " + cmd,
                        "status": "completed",
                    },
                },
            })
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": reply},
                    },
                },
            })
            if root:
                with open(os.path.join(root, "terminal-output.json"), "w") as f:
                    json.dump(out, f)
        elif "slow-reply" in text:
            time.sleep(2.0)
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "too-late"},
                    },
                },
            })
        else:
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": state["session_id"],
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "hello-from-goose"},
                    },
                },
            })
        rpc_result(msg_id, {"stopReason": "end_turn"})
    elif method:
        if msg_id is not None:
            send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": method}})

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    handle(json.loads(raw))
"#;

fn scratch() -> PathBuf {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir()
        .join("syllabix-goose-acp")
        .join(format!(
            "{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst)
        ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn fake_goose_bin() -> PathBuf {
    let dir = scratch();
    let path = dir.join("goose");
    fs::write(&path, FAKE_ACP).unwrap();
    let mut perms = fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).unwrap();
    path
}

fn online_config() -> crate::AgentConfig {
    let mut config = crate::AgentConfig::v0();
    config.agent = crate::defaults::AgentBackend::Goose;
    config.llm = crate::defaults::LlmProvider::Online;
    config.llm_model = "gpt-4o-mini".into();
    config
}

fn start_fake() -> (GooseAgent, PathBuf) {
    start_fake_with(online_config())
}

fn start_fake_with(config: crate::AgentConfig) -> (GooseAgent, PathBuf) {
    let binary = fake_goose_bin();
    let home = scratch().join("goose-home");
    fs::create_dir_all(&home).unwrap();
    let workspace = scratch().join("ws");
    fs::create_dir_all(&workspace).unwrap();
    let agent = GooseAgent::start(GooseStart {
        binary,
        cache: home.clone(),
        config,
        api_key: Zeroizing::new("sk-x".into()),
        workspace,
        skill_list: String::new(),
    })
    .unwrap_or_else(|err| {
        let stderr = fs::read_to_string(home.join("goose.stderr")).unwrap_or_default();
        panic!("fake ACP handshake: {err}; stderr={stderr}");
    });
    (agent, home)
}

#[test]
fn acp_handshake_writes_permission_yaml_and_spawn_env() {
    let (agent, home) = start_fake();
    let yaml = fs::read_to_string(home.join("config/permission.yaml")).unwrap();
    assert!(yaml.contains("shell"), "{yaml}");
    assert!(yaml.contains("never_allow"), "{yaml}");
    let env: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(home.join("spawn-env.json")).unwrap()).unwrap();
    assert_eq!(env["GOOSE_PROVIDER"], "openai");
    assert_eq!(env["GOOSE_MODE"], "approve");
    assert_eq!(env["GOOSE_DISABLE_KEYRING"], "1");
    assert_eq!(env["OPENAI_API_KEY"], "sk-x");
    assert_eq!(env["argv"][1], "acp");
    drop(agent);
}

#[test]
fn generate_streams_tokens_without_sending_history() {
    let (mut agent, _) = start_fake();
    let transcript = Transcript {
        turn: TurnId(1),
        text: "hi".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    agent
        .generate(&[], &transcript, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk.text.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(chunks.last().map(String::as_str), Some("hello-from-goose"));
    assert!(agent.acp_tool_call_ids().is_empty());
    assert_eq!(agent.terminal_create_count(), 0);
    assert!(
        agent.take_acp_trace().is_empty(),
        "diagnostics off → no ACP trace"
    );
    assert!(agent
        .debug_meta()
        .unwrap()
        .endpoint
        .contains("chat/completions"));
    assert_eq!(agent.name(), "goose");
}

#[test]
fn generate_cancel_wakes_without_waiting_for_idle_timeout() {
    let (mut agent, _) = start_fake();
    let cancel = Cancel::new();
    let watcher = cancel.clone();
    let started = std::time::Instant::now();
    let join = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(150));
        watcher.cancel_generation();
    });
    let transcript = Transcript {
        turn: TurnId(4),
        text: "please slow-reply".into(),
        language: "en".into(),
    };
    let err = agent
        .generate(&[], &transcript, &cancel, &mut |_| Ok(()))
        .expect_err("cancel");
    join.join().unwrap();
    assert!(matches!(err, crate::error::Error::Cancelled), "{err}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "cancel must not wait on the 2s fake sleep / idle timeout"
    );
}

#[test]
fn diagnostics_records_acp_jsonl_lines_both_directions() {
    let mut config = online_config();
    config.diagnostics_timestamps = true;
    let dump = scratch().join("turn-debug");
    config.diagnostics_directory = dump.clone();
    let (mut agent, _) = start_fake_with(config);
    let transcript = Transcript {
        turn: TurnId(5),
        text: "hi".into(),
        language: "en".into(),
    };
    agent
        .generate(&[], &transcript, &Cancel::new(), &mut |_| Ok(()))
        .unwrap();
    let body = fs::read_to_string(dump.join("turn-005").join("acp.jsonl")).unwrap();
    assert!(body.contains("\"dir\":\"out\""), "{body}");
    assert!(body.contains("session/prompt"), "{body}");
    assert!(body.contains("\"dir\":\"in\""), "{body}");
    assert!(body.contains("hello-from-goose"), "{body}");
    assert!(
        !body.contains("sk-x"),
        "spawn-env secrets must not appear in the ACP dump: {body}"
    );
    assert!(agent.take_acp_trace().is_empty(), "flush drains the buffer");
}

#[test]
fn generate_idle_waits_out_a_long_shell() {
    // Without shell_inflight, a 400 ms idle limit would fail during `sleep 1`.
    let _idle = GooseIdleGuard::set(Duration::from_millis(400));
    let (mut agent, _) = start_fake();
    let transcript = Transcript {
        turn: TurnId(6),
        text: "please slow-shell".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    agent
        .generate(&[], &transcript, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk.text.clone());
            Ok(())
        })
        .expect("shell work must not count as idle silence");
    assert_eq!(chunks.last().map(String::as_str), Some("ran-slow-shell"));
}

#[test]
fn terminal_create_runs_through_execute_with_permissions() {
    let (mut agent, home) = start_fake();
    let transcript = Transcript {
        turn: TurnId(2),
        text: "please use-shell".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    agent
        .generate(&[], &transcript, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk.text.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(chunks.last().map(String::as_str), Some("ran-shell"));
    let events = agent.take_tool_events();
    assert!(
        events
            .iter()
            .any(|event| event.kind == "call" && event.name == "shell"),
        "{events:?}"
    );
    assert!(
        events.iter().all(|event| event.name == "shell"),
        "{events:?}"
    );
    assert_eq!(agent.acp_tool_call_ids(), ["call-shell-1"]);
    assert_eq!(
        agent.acp_tool_call_ids().len() as u64,
        agent.terminal_create_count()
    );
    let out = fs::read_to_string(home.join("terminal-output.json")).unwrap();
    assert!(out.contains("goose-shell") || out.contains("exit"), "{out}");
}

#[test]
fn terminal_create_refuses_cwd_outside_workspace() {
    let (mut agent, home) = start_fake();
    let transcript = Transcript {
        turn: TurnId(3),
        text: "please escape-cwd".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    agent
        .generate(&[], &transcript, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk.text.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(chunks.last().map(String::as_str), Some("refused-cwd"));
    let reply: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(home.join("cwd-refuse.json")).unwrap()).unwrap();
    assert!(reply.get("error").is_some(), "{reply}");
    assert!(reply.get("result").is_none(), "{reply}");
    assert!(agent.take_tool_events().is_empty());
    assert_eq!(agent.terminal_create_count(), 0);
}

#[test]
fn goose_crash_is_a_provider_error() {
    let dir = scratch();
    let path = dir.join("goose");
    fs::write(&path, "#!/bin/sh\nexit 1\n").unwrap();
    let mut perms = fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&path, perms).unwrap();
    let err = GooseAgent::start(GooseStart {
        binary: path,
        cache: scratch().join("home"),
        config: online_config(),
        api_key: Zeroizing::new("sk-x".into()),
        workspace: scratch().join("ws"),
        skill_list: String::new(),
    })
    .err()
    .expect("dead goose");
    assert!(err.to_string().contains("goose"), "{err}");
}
