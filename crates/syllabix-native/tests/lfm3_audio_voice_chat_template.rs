//! Voice-chat turn render from the LFM3-Audio default chat template (no weights).

#![cfg(feature = "cpp")]
#![cfg(not(coverage))]

use std::path::PathBuf;

use syllabix_core::s2s::LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT;
use syllabix_native::ChatTemplateSession;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chat_templates")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn lfm3_audio_voice_chat_template_renders_turn() {
    let messages = format!(
        r#"[{{"role":"system","content":{system}}},{{"role":"user","content":"<|audio|>"}}]"#,
        system = serde_json::to_string(LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT).unwrap()
    );
    let mut session =
        ChatTemplateSession::from_template(&fixture("LFM3-Audio-2B.jinja")).expect("template");
    let rendered = session
        .apply_prompt(&messages, "", false)
        .expect("apply voice chat turn");
    assert!(
        rendered.contains(LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT),
        "system prompt missing"
    );
    assert!(rendered.contains("<|audio|>"), "audio marker missing");
    assert!(
        rendered.contains("<|im_start|>assistant"),
        "assistant generation prompt missing: {rendered}"
    );
    assert!(
        !rendered.contains("<|audio_start|>"),
        "must not use audio_start prefill"
    );
}
