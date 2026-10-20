//! Process tests for the Goose ACP client. A tiny Python sidecar speaks ACP.

use super::*;
use crate::cancel::Cancel;
use crate::error::Error;
use crate::providers::Llm;
use crate::types::TurnId;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const FAKE_ACP: &str = r#"#!/usr/bin/env python3
import json, os, select, sys, time

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

state = {"session_id": "sess-1", "cwd": os.getcwd(), "prompt_id": None}

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def rpc_result(msg_id, result):
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})

def note_cancel(msg):
    if root:
        with open(os.path.join(root, "got-cancel.json"), "w") as f:
            json.dump(msg, f)

def wait_line(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    raw = sys.stdin.readline()
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return wait_line(timeout)
    return json.loads(raw)

def tool_update(call_id, title, status):
    kind = "tool_call" if status == "pending" else "tool_call_update"
    send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": state["session_id"],
            "update": {
                "sessionUpdate": kind,
                "toolCallId": call_id,
                "title": title,
                "status": status,
            },
        },
    })

def run_terminal(command, call_id="call-shell-1"):
    tool_update(call_id, "shell · " + command, "pending")
    send({
        "jsonrpc": "2.0",
        "id": 9001,
        "method": "terminal/create",
        "params": {
            "sessionId": state["session_id"],
            "command": command,
            "cwd": state["cwd"],
            "outputByteLimit": 100000,
        },
    })
    created = wait_line(30)
    if created is None:
        return None
    if created.get("method") == "session/cancel":
        note_cancel(created)
        created = wait_line(30)
        if created is None:
            return None
    tid = (created.get("result") or {}).get("terminalId")
    send({
        "jsonrpc": "2.0",
        "id": 9002,
        "method": "terminal/wait_for_exit",
        "params": {"sessionId": state["session_id"], "terminalId": tid},
    })
    waited = wait_line(30)
    if waited and waited.get("method") == "session/cancel":
        note_cancel(waited)
        wait_line(30)
    send({
        "jsonrpc": "2.0",
        "id": 9003,
        "method": "terminal/output",
        "params": {"sessionId": state["session_id"], "terminalId": tid},
    })
    out = wait_line(30)
    if out and out.get("method") == "session/cancel":
        note_cancel(out)
        out = wait_line(30)
    send({
        "jsonrpc": "2.0",
        "id": 9004,
        "method": "terminal/release",
        "params": {"sessionId": state["session_id"], "terminalId": tid},
    })
    released = wait_line(30)
    if released and released.get("method") == "session/cancel":
        note_cancel(released)
        wait_line(30)
    tool_update(call_id, "shell · " + command, "completed")
    return out

def chunk(text):
    send({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": state["session_id"],
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    })

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
        assert len(ext) == 2 and ext[1].get("type") == "mcp", ext
        assert ext[1].get("server", {}).get("name") == "syllabix", ext
        assert ext[1].get("available_tools") == ["web_search", "web_fetch"], ext
        servers = params.get("mcpServers") or []
        assert len(servers) == 1 and servers[0].get("name") == "syllabix", servers
        assert "mcp" in (servers[0].get("args") or []), servers
        if root:
            with open(os.path.join(root, "session-new.json"), "w") as f:
                json.dump(params, f)
        rpc_result(msg_id, {"sessionId": state["session_id"]})
    elif method == "_goose/unstable/session/system-prompt/set":
        assert params.get("mode") == "set"
        assert params.get("key") == "syllabix"
        rpc_result(msg_id, {})
    elif method == "session/cancel":
        note_cancel(msg)
        if state["prompt_id"] is not None:
            rpc_result(state["prompt_id"], {"stopReason": "cancelled"})
            state["prompt_id"] = None
    elif method == "session/prompt":
        text = ""
        for block in params.get("prompt") or []:
            text += block.get("text") or ""
        state["prompt_id"] = msg_id
        if "crash-now" in text:
            os._exit(1)
        if "prompt-error" in text:
            send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": "prompt failed"}})
            state["prompt_id"] = None
            return
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
            created = wait_line(30)
            chunk("refused-cwd")
            if root and created is not None:
                with open(os.path.join(root, "cwd-refuse.json"), "w") as f:
                    json.dump(created, f)
            rpc_result(msg_id, {"stopReason": "end_turn"})
            state["prompt_id"] = None
            return
        if "sleep-shell" in text:
            if root:
                with open(os.path.join(root, "waiting.json"), "w") as f:
                    f.write("1")
            out = run_terminal("sleep 30")
            if root and out is not None:
                with open(os.path.join(root, "terminal-output.json"), "w") as f:
                    json.dump(out, f)
            if state["prompt_id"] is not None:
                rpc_result(msg_id, {"stopReason": "end_turn"})
                state["prompt_id"] = None
            return
        if "use-shell" in text or "slow-shell" in text:
            cmd = "sleep 1; echo goose-shell" if "slow-shell" in text else "echo goose-shell"
            reply = "ran-slow-shell" if "slow-shell" in text else "ran-shell"
            out = run_terminal(cmd)
            chunk(reply)
            if root and out is not None:
                with open(os.path.join(root, "terminal-output.json"), "w") as f:
                    json.dump(out, f)
            rpc_result(msg_id, {"stopReason": "end_turn"})
            state["prompt_id"] = None
            return
        if "ignore-cancel" in text:
            chunk("partial")
            if root:
                with open(os.path.join(root, "waiting.json"), "w") as f:
                    f.write("1")
            time.sleep(30)
            rpc_result(msg_id, {"stopReason": "end_turn"})
            state["prompt_id"] = None
            return
        if "hold-for-cancel" in text:
            chunk("partial")
            if root:
                with open(os.path.join(root, "waiting.json"), "w") as f:
                    f.write("1")
            nxt = wait_line(10)
            if nxt and nxt.get("method") == "session/cancel":
                note_cancel(nxt)
                chunk("should-drop")
                rpc_result(msg_id, {"stopReason": "cancelled"})
                state["prompt_id"] = None
            elif nxt:
                handle(nxt)
            return
        if "create-after-cancel" in text:
            if root:
                with open(os.path.join(root, "waiting.json"), "w") as f:
                    f.write("1")
            nxt = wait_line(10)
            if nxt and nxt.get("method") == "session/cancel":
                note_cancel(nxt)
                # After cancel, Goose may still request a shell; it must not live on.
                out = run_terminal("sleep 30")
                if root and out is not None:
                    with open(os.path.join(root, "terminal-output.json"), "w") as f:
                        json.dump(out, f)
                if state["prompt_id"] is not None:
                    rpc_result(msg_id, {"stopReason": "cancelled"})
                    state["prompt_id"] = None
            elif nxt:
                handle(nxt)
            return
        if "slow-reply" in text:
            nxt = wait_line(10)
            if nxt and nxt.get("method") == "session/cancel":
                note_cancel(nxt)
                rpc_result(msg_id, {"stopReason": "cancelled"})
                state["prompt_id"] = None
                return
            chunk("too-late")
            rpc_result(msg_id, {"stopReason": "end_turn"})
            state["prompt_id"] = None
            return
        chunk("hello-from-goose")
        rpc_result(msg_id, {"stopReason": "end_turn"})
        state["prompt_id"] = None
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

fn goose_config() -> crate::AgentConfig {
    let mut config = crate::AgentConfig::v0();
    config.agent = crate::defaults::AgentBackend::Goose;
    config
}

fn write_goose_config(home: &std::path::Path) {
    fs::create_dir_all(home.join("config")).unwrap();
    fs::write(
        home.join("config/config.yaml"),
        "active_provider: local\nproviders:\n  local:\n    enabled: true\n    model: lfm2.5-2.6b\n",
    )
    .unwrap();
}

fn start_fake() -> (GooseAgent, PathBuf) {
    start_fake_with(goose_config())
}

fn start_fake_with(config: crate::AgentConfig) -> (GooseAgent, PathBuf) {
    let binary = fake_goose_bin();
    let home = scratch().join("goose-home");
    fs::create_dir_all(&home).unwrap();
    write_goose_config(&home);
    let workspace = scratch().join("ws");
    fs::create_dir_all(&workspace).unwrap();
    let agent = GooseAgent::start(GooseStart {
        binary,
        cache: home.clone(),
        config,
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
    assert!(yaml.contains("syllabix__web_search"), "{yaml}");
    assert!(yaml.contains("syllabix__web_fetch"), "{yaml}");
    assert!(yaml.contains("never_allow"), "{yaml}");
    let env: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(home.join("spawn-env.json")).unwrap()).unwrap();
    assert_eq!(env["GOOSE_PROVIDER"], "");
    assert_eq!(env["GOOSE_MODEL"], "");
    assert_eq!(env["GOOSE_MODE"], "approve");
    assert_eq!(env["GOOSE_DISABLE_KEYRING"], "1");
    assert_eq!(env["OPENAI_API_KEY"], "");
    assert_eq!(env["argv"][1], "acp");
    let session: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(home.join("session-new.json")).unwrap()).unwrap();
    assert_eq!(session["mcpServers"][0]["name"], "syllabix");
    assert!(
        session["mcpServers"][0]["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|arg| arg == "mcp"),
        "{session}"
    );
    assert_eq!(session["_meta"]["enabledExtensions"][1]["type"], "mcp");
    drop(agent);
}

#[test]
fn take_tool_events_merges_mcp_jsonl() {
    let (mut agent, home) = start_fake();
    let events_path = home.join("mcp-tool-events.jsonl");
    fs::write(
        &events_path,
        concat!(
            r#"{"kind":"call","name":"web_search","call_id":"mcp-1","arguments":"{\"query\":\"rust\"}","content":""}"#,
            "\n",
            r#"{"kind":"result","name":"web_search","call_id":"mcp-1","arguments":"{\"query\":\"rust\"}","content":"[]"}"#,
            "\n",
        ),
    )
    .unwrap();
    let events = agent.take_tool_events();
    assert!(
        events
            .iter()
            .any(|event| event.kind == "call" && event.name == "web_search"),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| event.kind == "result" && event.name == "web_search"),
        "{events:?}"
    );
    assert!(agent.take_tool_events().is_empty());
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
    let meta = agent.debug_meta().unwrap();
    assert_eq!(meta.endpoint, "");
    assert_eq!(meta.model, "lfm2.5-2.6b");
    assert_eq!(agent.name(), "goose");
}

#[test]
fn generate_cancel_wakes_without_waiting_for_idle_timeout() {
    let (mut agent, _) = start_fake();
    let cancel = Cancel::new();
    let watcher = cancel.clone();
    let started = Instant::now();
    let join = thread::spawn(move || {
        thread::sleep(Duration::from_millis(150));
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
    assert!(matches!(err, Error::Cancelled), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "cancel must not wait on idle timeout"
    );
}

#[test]
fn diagnostics_records_acp_jsonl_lines_both_directions() {
    let mut config = goose_config();
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
    let home = scratch().join("home");
    write_goose_config(&home);
    let err = GooseAgent::start(GooseStart {
        binary: path,
        cache: home,
        config: goose_config(),
        workspace: scratch().join("ws"),
        skill_list: String::new(),
    })
    .err()
    .expect("dead goose");
    assert!(err.to_string().contains("goose"), "{err}");
}

fn wait_file(path: &std::path::Path, timeout: Duration) {
    let start = Instant::now();
    while !path.exists() {
        assert!(
            start.elapsed() < timeout,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn barge_in_sends_session_cancel_and_drops_later_tokens() {
    let (mut agent, home) = start_fake();
    let cancel = Cancel::new();
    let transcript = Transcript {
        turn: TurnId(6),
        text: "hold-for-cancel".into(),
        language: "en".into(),
    };
    let cancel_thread = cancel.clone();
    let worker = thread::spawn(move || {
        let mut chunks = Vec::new();
        let result = agent.generate(&[], &transcript, &cancel_thread, &mut |chunk| {
            chunks.push(chunk.text.clone());
            Ok(())
        });
        (result, chunks)
    });
    wait_file(&home.join("waiting.json"), Duration::from_secs(5));
    cancel.cancel_generation();
    let (result, chunks) = worker.join().expect("generate thread");
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "expected cancelled, got {result:?}"
    );
    assert!(
        !chunks.iter().any(|chunk| chunk.contains("should-drop")),
        "{chunks:?}"
    );
    let got = fs::read_to_string(home.join("got-cancel.json")).unwrap();
    assert!(got.contains("session/cancel"), "{got}");
}

#[test]
fn cancel_deadline_returns_without_waiting_for_goose() {
    let (mut agent, home) = start_fake();
    let cancel = Cancel::new();
    let transcript = Transcript {
        turn: TurnId(7),
        text: "ignore-cancel".into(),
        language: "en".into(),
    };
    let cancel_thread = cancel.clone();
    let started = Instant::now();
    let worker =
        thread::spawn(move || agent.generate(&[], &transcript, &cancel_thread, &mut |_| Ok(())));
    wait_file(&home.join("waiting.json"), Duration::from_secs(5));
    cancel.cancel_generation();
    let result = worker.join().expect("generate thread");
    let elapsed = started.elapsed();
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "expected cancelled, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(8),
        "cancel deadline too slow: {elapsed:?}"
    );
}

#[test]
fn prompt_rpc_error_is_a_provider_error() {
    let (mut agent, _) = start_fake();
    let transcript = Transcript {
        turn: TurnId(8),
        text: "prompt-error".into(),
        language: "en".into(),
    };
    let err = agent
        .generate(&[], &transcript, &Cancel::new(), &mut |_| Ok(()))
        .expect_err("prompt rpc error");
    match err {
        Error::Provider { provider, message } => {
            assert_eq!(provider, "goose");
            assert!(message.contains("prompt failed"), "{message}");
        }
        other => panic!("expected provider error, got {other}"),
    }
}

#[test]
fn mid_turn_crash_is_a_provider_error() {
    let (mut agent, _) = start_fake();
    let transcript = Transcript {
        turn: TurnId(9),
        text: "crash-now".into(),
        language: "en".into(),
    };
    let err = agent
        .generate(&[], &transcript, &Cancel::new(), &mut |_| Ok(()))
        .expect_err("dead goose mid-turn");
    match err {
        Error::Provider { provider, .. } => assert_eq!(provider, "goose"),
        other => panic!("expected provider error, got {other}"),
    }
}

#[test]
fn barge_in_kills_running_shell() {
    let (mut agent, home) = start_fake();
    let cancel = Cancel::new();
    let transcript = Transcript {
        turn: TurnId(10),
        text: "sleep-shell".into(),
        language: "en".into(),
    };
    let cancel_thread = cancel.clone();
    let started = Instant::now();
    let worker =
        thread::spawn(move || agent.generate(&[], &transcript, &cancel_thread, &mut |_| Ok(())));
    wait_file(&home.join("waiting.json"), Duration::from_secs(5));
    thread::sleep(Duration::from_millis(150));
    cancel.cancel_generation();
    let result = worker.join().expect("generate thread");
    let elapsed = started.elapsed();
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "expected cancelled, got {result:?}"
    );
    // The fake sidecar writes this only after wait_for_exit returns. The 2 s
    // cancel deadline alone cannot create it while sleep 30 is still running.
    wait_file(&home.join("terminal-output.json"), Duration::from_secs(3));
    assert!(
        elapsed < Duration::from_secs(8),
        "running shell was not killed: {elapsed:?}"
    );
}

#[test]
fn terminal_create_after_cancel_does_not_leave_a_live_shell() {
    let (mut agent, home) = start_fake();
    let cancel = Cancel::new();
    let transcript = Transcript {
        turn: TurnId(11),
        text: "create-after-cancel".into(),
        language: "en".into(),
    };
    let cancel_thread = cancel.clone();
    let started = Instant::now();
    let worker =
        thread::spawn(move || agent.generate(&[], &transcript, &cancel_thread, &mut |_| Ok(())));
    wait_file(&home.join("waiting.json"), Duration::from_secs(5));
    cancel.cancel_generation();
    let result = worker.join().expect("generate thread");
    assert!(
        matches!(result, Err(Error::Cancelled)),
        "expected cancelled, got {result:?}"
    );
    // Sidecar only writes this after wait_for_exit. A live sleep 30 would not.
    wait_file(&home.join("terminal-output.json"), Duration::from_secs(3));
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "post-cancel shell was left running"
    );
}
