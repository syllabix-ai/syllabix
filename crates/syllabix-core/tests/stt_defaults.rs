//! `syllabix-core` defaults still turn on every speech-to-text engine after
//! the move to `syllabix-stt`: Whisper and Qwen3-ASR with `ggml`, the remote
//! client with `cloud`, and Moonshine always.

const CORE_TOML: &str = include_str!("../Cargo.toml");

/// The value of `key = ...` inside `[section]`, as one trimmed line.
fn entry(section: &str, key: &str) -> &'static str {
    let mut in_section = false;
    for line in CORE_TOML.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_section = line == section;
            continue;
        }
        if in_section {
            if let Some(rest) = line.strip_prefix(key) {
                if let Some(value) = rest.trim_start().strip_prefix('=') {
                    return value.trim();
                }
            }
        }
    }
    panic!("{section} {key} missing from syllabix-core/Cargo.toml");
}

#[test]
fn default_features_keep_every_stt_engine() {
    assert_eq!(entry("[features]", "default"), r#"["full"]"#);
    let full = entry("[features]", "full");
    assert!(
        full.contains(r#""ggml""#) && full.contains(r#""cloud""#),
        "{full}"
    );
    let ggml = entry("[features]", "ggml");
    assert!(ggml.contains(r#""syllabix-stt/whisper""#), "{ggml}");
    assert!(ggml.contains(r#""syllabix-stt/qwen3-asr""#), "{ggml}");
    let cloud = entry("[features]", "cloud");
    assert!(cloud.contains(r#""syllabix-stt/remote""#), "{cloud}");
    let dep = entry("[dependencies]", "syllabix-stt");
    assert!(dep.contains(r#""moonshine""#), "{dep}");
}

#[test]
fn stt_paths_still_compile() {
    // Old paths keep working in every feature set.
    let _: Option<syllabix_core::stt::SttModel> =
        syllabix_core::defaults::SttModel::parse("whisper-small");
    let _ = syllabix_core::moonshine::ENCODER_ASSET;
    let _ = syllabix_core::qwen_asr::ASR_ASSET;
    let _ = syllabix_core::stt::STT_LANGUAGE;
    fn takes_stt(_: &dyn syllabix_core::Stt) {}
    let _ = takes_stt;
    #[cfg(feature = "cloud")]
    let _ = syllabix_core::remote_stt::STT_API_KEY_ENV;
}
