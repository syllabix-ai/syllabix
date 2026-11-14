//! `syllabix-riley-mcp` process tests: exit codes, stdio round trip, and no
//! URL (or password) on stderr.

use std::io::Write;
use std::process::{Command, Stdio};

use syllabix_riley_eval::{DATABASE_URL_ENV, SCHEMA_ENV};

const BIN: &str = env!("CARGO_BIN_EXE_syllabix-riley-mcp");

#[test]
fn binary_without_url_exits_2() {
    let output = Command::new(BIN)
        .env_remove(DATABASE_URL_ENV)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "SYLLABIX_RILEY_DATABASE_URL is not set\n"
    );
}

#[test]
fn binary_connection_error_exits_1_without_printing_the_url() {
    let output = Command::new(BIN)
        .env(
            DATABASE_URL_ENV,
            "postgres://riley:not-a-real-secret@127.0.0.1:9/riley?connect_timeout=2",
        )
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.starts_with("riley eval: database error"), "{stderr}");
    assert!(!stderr.contains("not-a-real-secret"), "{stderr}");
    assert!(!stderr.contains("127.0.0.1:9"), "{stderr}");
}

#[test]
fn binary_serves_the_five_tools_over_stdio() {
    let Ok(url) = std::env::var(DATABASE_URL_ENV) else {
        eprintln!("skip binary_serves_the_five_tools_over_stdio: {DATABASE_URL_ENV} is not set");
        return;
    };
    let mut child = Command::new(BIN)
        .env(DATABASE_URL_ENV, url)
        .env(SCHEMA_ENV, "riley_t_bin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let requests = concat!(
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        "\n",
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"change_card_status","arguments":{"card_id":"c_bob_debit","new_status":"frozen"}}}"#,
        "\n",
    );
    child
        .stdin
        .take()
        .unwrap()
        .write_all(requests.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let replies: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies[0]["result"]["tools"].as_array().unwrap().len(), 5);
    assert_eq!(replies[1]["result"]["isError"], false);
    let text = replies[1]["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains(r#""status":"frozen""#), "{text}");
}
