//! Eval-only MCP stdio server for the five Riley tools
//! (`syllabix-riley-mcp`). Same newline-delimited JSON-RPC shape as
//! `syllabix_core::mcp`; Goose can attach it as a stdio extension. Not part
//! of `syllabix mcp`, so the default tool set is unchanged.

use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::db::CardDb;
use crate::tools;
use crate::{DATABASE_URL_ENV, DEFAULT_SCHEMA, SCHEMA_ENV};

/// MCP server name.
pub const SERVER_NAME: &str = "syllabix-riley";

/// Serve JSON-RPC lines until input ends. `dispatch` runs one tool call.
pub fn serve(
    lines: impl Iterator<Item = std::io::Result<String>>,
    out: &mut impl Write,
    dispatch: &mut dyn FnMut(&str, &Value) -> Result<Value, String>,
) -> std::io::Result<()> {
    for line in lines {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(response) = handle_message(&msg, dispatch) {
            writeln!(out, "{response}")?;
            out.flush()?;
        }
    }
    Ok(())
}

fn handle_message(
    msg: &Value,
    dispatch: &mut dyn FnMut(&str, &Value) -> Result<Value, String>,
) -> Option<String> {
    let method = msg.get("method").and_then(Value::as_str)?;
    // Notifications have no id and need no reply.
    let id = msg.get("id").cloned()?;
    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
        }),
        "ping" => json!({}),
        "tools/list" => json!({ "tools": tools::mcp_tool_list() }),
        "tools/call" => call_tool(msg.get("params").unwrap_or(&Value::Null), dispatch),
        other => {
            return Some(
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("method not found: {other}") }
                })
                .to_string(),
            );
        }
    };
    Some(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string())
}

fn call_tool(
    params: &Value,
    dispatch: &mut dyn FnMut(&str, &Value) -> Result<Value, String>,
) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let empty = json!({});
    let arguments = params.get("arguments").unwrap_or(&empty);
    let (text, is_error) = match dispatch(name, arguments) {
        Ok(value) => (value.to_string(), false),
        Err(message) => (json!({ "error": message }).to_string(), true),
    };
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error,
    })
}

/// Entry point for `syllabix-riley-mcp`: read the URL from the environment,
/// reset the seed, then serve stdio. Exit codes: 0 ok, 1 database or I/O
/// error, 2 URL not set. Never writes the URL anywhere.
pub fn run_from_env(
    getenv: impl Fn(&str) -> Option<String>,
    input: impl BufRead,
    out: &mut impl Write,
    err: &mut impl Write,
) -> u8 {
    let Some(url) = getenv(DATABASE_URL_ENV) else {
        let _ = writeln!(err, "{DATABASE_URL_ENV} is not set");
        return 2;
    };
    let schema = getenv(SCHEMA_ENV).unwrap_or_else(|| DEFAULT_SCHEMA.to_string());
    let mut db = match CardDb::connect(&url, &schema) {
        Ok(db) => db,
        Err(message) => {
            let _ = writeln!(err, "riley eval: {message}");
            return 1;
        }
    };
    if let Err(message) = db.reset() {
        let _ = writeln!(err, "riley eval: {message}");
        return 1;
    }
    let mut dispatch = |name: &str, args: &Value| tools::dispatch(&mut db, name, args);
    match serve(input.lines(), out, &mut dispatch) {
        Ok(()) => 0,
        Err(io) => {
            let _ = writeln!(err, "riley eval: {io}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn roundtrip(requests: &[&str]) -> (Vec<Value>, Vec<(String, Value)>) {
        let input = requests.join("\n") + "\n";
        let mut out = Vec::new();
        let mut seen = Vec::new();
        let mut fake = |name: &str, args: &Value| {
            seen.push((name.to_string(), args.clone()));
            if name == "display_user_info" {
                Ok(json!({"id": "u_test"}))
            } else {
                Err(format!("unknown tool: {name}"))
            }
        };
        serve(Cursor::new(input).lines(), &mut out, &mut fake).unwrap();
        let replies = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        (replies, seen)
    }

    #[test]
    fn initialize_and_list_advertise_the_five_riley_tools() {
        let (replies, _) = roundtrip(&[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"ping"}"#,
        ]);
        assert_eq!(replies.len(), 3, "notification gets no reply");
        assert_eq!(replies[0]["result"]["serverInfo"]["name"], SERVER_NAME);
        let names: Vec<&str> = replies[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, tools::TOOL_NAMES);
        assert!(!names.contains(&"shell"));
        assert_eq!(replies[2]["result"], json!({}));
    }

    #[test]
    fn tools_call_returns_text_and_error_flags() {
        let (replies, seen) = roundtrip(&[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"display_user_info","arguments":{"user_id":"u_test"}}}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"shell"}}"#,
            "not json",
            r#"{"jsonrpc":"2.0","id":3,"method":"resources/list"}"#,
        ]);
        assert_eq!(replies.len(), 3, "bad JSON line is skipped");
        assert_eq!(replies[0]["result"]["isError"], false);
        assert_eq!(
            replies[0]["result"]["content"][0]["text"],
            r#"{"id":"u_test"}"#
        );
        assert_eq!(replies[1]["result"]["isError"], true);
        assert_eq!(
            replies[1]["result"]["content"][0]["text"],
            r#"{"error":"unknown tool: shell"}"#
        );
        assert_eq!(replies[2]["error"]["code"], -32601);
        assert_eq!(seen[1], ("shell".to_string(), json!({})));
    }

    #[test]
    fn run_from_env_without_url_exits_2() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run_from_env(|_| None, Cursor::new(""), &mut out, &mut err);
        assert_eq!(code, 2);
        assert!(out.is_empty());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "SYLLABIX_RILEY_DATABASE_URL is not set\n"
        );
    }

    #[test]
    fn run_from_env_bad_schema_exits_1_without_leaking_the_url() {
        let getenv = |key: &str| match key {
            DATABASE_URL_ENV => Some("postgres://u:hunter2@127.0.0.1:9/x".to_string()),
            SCHEMA_ENV => Some("Bad-Schema".to_string()),
            _ => None,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = run_from_env(getenv, Cursor::new(""), &mut out, &mut err);
        assert_eq!(code, 1);
        let err = String::from_utf8(err).unwrap();
        assert!(err.contains("invalid schema name"), "{err}");
        assert!(!err.contains("hunter2"), "{err}");
    }
}
