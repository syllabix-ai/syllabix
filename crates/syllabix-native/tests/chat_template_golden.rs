//! Template-only apply/parse goldens: no GGUF weights. Fails CI if common chat
//! stops loading tool PEG parsers or parsing native tool dialects.
//!
//! Skipped under `cfg(coverage)` because the native stubs do not implement common chat.

#![cfg(not(coverage))]

use std::path::PathBuf;

use syllabix_native::{ChatTemplateSession, ParsedChatOutput};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chat_templates")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

const TOOLS_JSON: &str = r#"[
  {"type":"function","function":{
    "name":"shell",
    "description":"run a command",
    "parameters":{"type":"object","required":["command"],"properties":{"command":{"type":"string"}}}
  }}
]"#;

const USER_TURN_JSON: &str =
    r#"[{"role":"user","content":"Run date in the shell and reply with nothing else."}]"#;

fn shell_command(output: &ParsedChatOutput) -> String {
    assert_eq!(
        output.tool_calls.len(),
        1,
        "expected one tool call: {output:?}"
    );
    assert_eq!(output.tool_calls[0].name, "shell");
    let args: serde_json::Value =
        serde_json::from_str(&output.tool_calls[0].arguments).expect("arguments json");
    args.get("command")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("missing command in {args}"))
        .to_string()
}

fn apply_parse(template_file: &str, assistant_text: &str) -> ParsedChatOutput {
    let mut session =
        ChatTemplateSession::from_template(&fixture(template_file)).expect("template session");
    session
        .apply_prompt(USER_TURN_JSON, TOOLS_JSON, false)
        .expect("apply");
    session.parse_output(assistant_text, false).expect("parse")
}

#[test]
fn lfm_template_parses_pythonic_tool_call() {
    let text = concat!(
        "<|im_start|>assistant\n",
        "<|tool_call_start|>[shell(command=\"date\")]<|tool_call_end|>"
    );
    let out = apply_parse("LFM2.5-Instruct.jinja", text);
    assert_eq!(shell_command(&out), "date");
}

#[test]
fn qwen35_template_parses_xml_tool_call() {
    let text = concat!(
        "<|im_start|>assistant\n",
        "<tool_call>\n<function=shell>\n<parameter=command>\ndate\n",
        "</parameter>\n</function>\n</tool_call>"
    );
    let out = apply_parse("Qwen3.5-4B.jinja", text);
    assert_eq!(shell_command(&out), "date");
}

#[test]
fn gemma4_template_parses_call_prefix_tool_call() {
    let text = concat!(
        "<|turn>model\n",
        "<|tool_call>call:shell{command:<|\"|>date<|\"|>}<tool_call|>"
    );
    let out = apply_parse("google-gemma-4-31B-it.jinja", text);
    assert_eq!(shell_command(&out), "date");
}

#[test]
fn llama32_instruct_template_parses_via_autoparser() {
    let mut session =
        ChatTemplateSession::from_template(&fixture("meta-llama-Llama-3.2-3B-Instruct.jinja"))
            .expect("template session");
    session
        .apply_prompt(USER_TURN_JSON, TOOLS_JSON, false)
        .expect("apply");
    let text = r#"{"name":"shell","parameters":{"command":"date"}}"#;
    let out = session.parse_output(text, false).expect("parse");
    assert_eq!(shell_command(&out), "date");
}

#[test]
fn lfm_streaming_partial_tool_call_reports_shell_name() {
    let mut session =
        ChatTemplateSession::from_template(&fixture("LFM2.5-Instruct.jinja")).expect("session");
    session
        .apply_prompt(USER_TURN_JSON, TOOLS_JSON, false)
        .expect("apply");
    let full = concat!(
        "<|im_start|>assistant\n",
        "<|tool_call_start|>[shell(command=\"date\")]<|tool_call_end|>"
    );
    let split = full.find("[shell(command=").expect("marker") + "[shell(command=\"da".len(); // mid-argument stream
    let partial = &full[..split];
    let out = session.parse_output(partial, true).expect("partial parse");
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].name, "shell");
    assert!(
        out.tool_calls[0].arguments.contains("da"),
        "partial arguments should include the streamed prefix, got {:?}",
        out.tool_calls[0].arguments
    );
    assert_eq!(
        shell_command(&session.parse_output(full, false).expect("full parse")),
        "date"
    );
}
