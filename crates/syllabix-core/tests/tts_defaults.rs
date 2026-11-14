//! `syllabix-core` defaults still turn on every text-to-speech engine after
//! the move to `syllabix-tts`: Qwen3-TTS with `ggml`, the remote client with
//! `cloud`, and Kokoro and Pocket TTS always.

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
fn default_features_keep_every_tts_engine() {
    assert_eq!(entry("[features]", "default"), r#"["full"]"#);
    let full = entry("[features]", "full");
    assert!(
        full.contains(r#""ggml""#) && full.contains(r#""cloud""#),
        "{full}"
    );
    let ggml = entry("[features]", "ggml");
    assert!(ggml.contains(r#""syllabix-tts/qwen3""#), "{ggml}");
    let cloud = entry("[features]", "cloud");
    assert!(cloud.contains(r#""syllabix-tts/remote""#), "{cloud}");
    let dep = entry("[dependencies]", "syllabix-tts");
    assert!(
        dep.contains(r#""kokoro""#) && dep.contains(r#""pocket""#),
        "{dep}"
    );
}

#[test]
fn tts_paths_still_compile() {
    // Old paths keep working in every feature set.
    let _: Option<syllabix_core::tts::TtsModel> =
        syllabix_core::defaults::TtsModel::parse("pocket-tts");
    let _ = syllabix_core::tts::KOKORO_ASSET;
    let _ = syllabix_core::tts::QwenTts::voice_anchor_engaged;
    let _ = syllabix_core::tts::KokoroTts::from_cache;
    let _ = syllabix_core::pocket_tts::POCKET_TTS_TEXT_CONDITIONER_ASSET;
    let _ = syllabix_core::g2p::KOKORO_MAX_PHONEME_TOKENS;
    let _ = syllabix_core::speech_text::speak_text_for_tts;
    fn takes_tts(_: &dyn syllabix_core::Tts) {}
    let _ = takes_tts;
    #[cfg(feature = "cloud")]
    let _ = syllabix_core::remote_tts::TTS_API_KEY_ENV;
}
