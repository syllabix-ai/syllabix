//! Minimal `syllabix.yaml`. Missing file means [`crate::BuiltinDefaults`].

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_yaml::Value;

use crate::defaults::{
    BuiltinDefaults, LlmProvider, SttModel, SttProvider, TtsProvider, VadProvider,
};
use crate::error::{Error, Result};
use crate::language::is_supported as is_supported_language;
use crate::vad::{VadSettings, END_SILENCE, MIN_SPEECH, SPEECH_THRESHOLD, WHISPER_PREROLL};

/// File name written by `init` and optionally read by `run`.
pub const CONFIG_FILE_NAME: &str = "syllabix.yaml";

/// Validated v0 agent config. One provider per layer; language is a single STT code.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Agent name (`demo-agent` by default).
    pub name: String,
    /// VAD provider.
    pub vad: VadProvider,
    /// Silero speech threshold `(0, 1]`.
    pub vad_threshold: f32,
    /// Contiguous speech to open a turn, milliseconds.
    pub vad_min_speech_ms: u32,
    /// Hangover silence to close a turn, milliseconds.
    pub vad_end_silence_ms: u32,
    /// Post-AEC Whisper preroll, milliseconds.
    pub vad_preroll_ms: u32,
    /// STT provider.
    pub stt: SttProvider,
    /// STT model id.
    pub stt_model: SttModel,
    /// STT language code: a whisper-supported ISO code or `auto`.
    pub language: String,
    /// LLM provider.
    pub llm: LlmProvider,
    /// LLM model id (`llama-3.2-1b`, `qwen3.5-0.8b`, or `qwen3.5-2b`).
    pub llm_model: String,
    /// Qwen thinking. Default false; yaml `thinking: true` enables it.
    pub thinking: bool,
    /// TTS provider.
    pub tts: TtsProvider,
}

impl AgentConfig {
    /// Built-in zero-config stack.
    pub fn v0() -> Self {
        Self::from_defaults(&BuiltinDefaults::v0())
    }

    /// Copy names from [`BuiltinDefaults`].
    pub fn from_defaults(defaults: &BuiltinDefaults) -> Self {
        Self {
            name: defaults.name.to_string(),
            vad: defaults.vad,
            vad_threshold: SPEECH_THRESHOLD,
            vad_min_speech_ms: MIN_SPEECH.as_millis() as u32,
            vad_end_silence_ms: END_SILENCE.as_millis() as u32,
            vad_preroll_ms: WHISPER_PREROLL.as_millis() as u32,
            stt: defaults.stt,
            stt_model: defaults.stt_model,
            language: defaults.language.to_string(),
            llm: defaults.llm,
            llm_model: defaults.llm_model.to_string(),
            thinking: defaults.llm_thinking,
            tts: defaults.tts,
        }
    }

    /// Parse and validate a yaml document.
    pub fn parse_yaml(text: &str) -> Result<Self> {
        let value: Value = serde_yaml::from_str(text).map_err(|err| Error::Config {
            field: ".".into(),
            message: err.to_string(),
        })?;
        parse_value(&value)
    }

    /// Load `path` as yaml.
    pub fn load_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::parse_yaml(&text).map_err(|err| match err {
            Error::Config { field, message } => Error::Config {
                field: format!("{}: {field}", path.display()),
                message,
            },
            other => other,
        })
    }

    /// `dir/syllabix.yaml` when the file exists; otherwise built-ins.
    pub fn resolve_for_run(dir: &Path) -> Result<Self> {
        let path = dir.join(CONFIG_FILE_NAME);
        if path.is_file() {
            Self::load_path(&path)
        } else {
            Ok(Self::v0())
        }
    }

    /// Canonical yaml matching `V0_LAUNCH.md` plus `language`.
    pub fn to_yaml(&self) -> String {
        format!(
            "\
name: {name}
pipeline:
  vad:
    provider: {vad}
    threshold: {vad_threshold}
    min_speech_ms: {vad_min_speech_ms}
    end_silence_ms: {vad_end_silence_ms}
    preroll_ms: {vad_preroll_ms}
  stt:
    provider: {stt}
    model: {stt_model}
    language: {language}
  llm:
    provider: {llm}
    model: {llm_model}
    thinking: {thinking}
  tts:
    provider: {tts}
",
            name = self.name,
            vad = self.vad.as_str(),
            vad_threshold = self.vad_threshold,
            vad_min_speech_ms = self.vad_min_speech_ms,
            vad_end_silence_ms = self.vad_end_silence_ms,
            vad_preroll_ms = self.vad_preroll_ms,
            stt = self.stt.as_str(),
            stt_model = self.stt_model.as_str(),
            language = self.language,
            llm = self.llm.as_str(),
            llm_model = self.llm_model,
            thinking = self.thinking,
            tts = self.tts.as_str(),
        )
    }

    /// Write [`CONFIG_FILE_NAME`] under `dir`. Creates `dir` when missing.
    pub fn write_init(dir: &Path) -> Result<PathBuf> {
        fs::create_dir_all(dir)?;
        let path = dir.join(CONFIG_FILE_NAME);
        if path.exists() {
            return Err(Error::Config {
                field: path.display().to_string(),
                message: "already exists".into(),
            });
        }
        fs::write(&path, Self::v0().to_yaml())?;
        Ok(path)
    }

    /// Silero turn policy for this config.
    pub fn vad_settings(&self) -> VadSettings {
        VadSettings {
            speech_threshold: self.vad_threshold,
            min_speech: Duration::from_millis(u64::from(self.vad_min_speech_ms)),
            end_silence: Duration::from_millis(u64::from(self.vad_end_silence_ms)),
            preroll: Duration::from_millis(u64::from(self.vad_preroll_ms)),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self::v0()
    }
}

fn parse_value(value: &Value) -> Result<AgentConfig> {
    let root = mapping(value, ".")?;
    deny_unknown(root, ".", &["name", "pipeline"])?;
    let name = required_string(root, "name", "name")?;
    let pipeline = mapping(
        root.get("pipeline").ok_or_else(|| missing("pipeline"))?,
        "pipeline",
    )?;
    deny_unknown(pipeline, "pipeline", &["vad", "stt", "llm", "tts"])?;

    let vad = mapping(required(pipeline, "pipeline.vad", "vad")?, "pipeline.vad")?;
    deny_unknown(
        vad,
        "pipeline.vad",
        &[
            "provider",
            "threshold",
            "min_speech_ms",
            "end_silence_ms",
            "preroll_ms",
        ],
    )?;
    let vad_provider = parse_vad(required_string(vad, "pipeline.vad.provider", "provider")?)?;
    let vad_threshold = optional_f32(vad, "pipeline.vad", "threshold", SPEECH_THRESHOLD)?;
    let vad_min_speech_ms = optional_u32(
        vad,
        "pipeline.vad",
        "min_speech_ms",
        MIN_SPEECH.as_millis() as u32,
    )?;
    let vad_end_silence_ms = optional_u32(
        vad,
        "pipeline.vad",
        "end_silence_ms",
        END_SILENCE.as_millis() as u32,
    )?;
    let vad_preroll_ms = optional_u32(
        vad,
        "pipeline.vad",
        "preroll_ms",
        WHISPER_PREROLL.as_millis() as u32,
    )?;
    validate_vad(
        vad_threshold,
        vad_min_speech_ms,
        vad_end_silence_ms,
        vad_preroll_ms,
    )?;

    let stt = mapping(required(pipeline, "pipeline.stt", "stt")?, "pipeline.stt")?;
    deny_unknown(stt, "pipeline.stt", &["provider", "model", "language"])?;
    let stt_provider = parse_stt(required_string(stt, "pipeline.stt.provider", "provider")?)?;
    let stt_model = parse_stt_model(required_string(stt, "pipeline.stt.model", "model")?)?;
    let language = parse_language(required_string(stt, "pipeline.stt.language", "language")?)?;

    let llm = mapping(required(pipeline, "pipeline.llm", "llm")?, "pipeline.llm")?;
    deny_unknown(llm, "pipeline.llm", &["provider", "model", "thinking"])?;
    let llm_provider = parse_llm(required_string(llm, "pipeline.llm.provider", "provider")?)?;
    let llm_model = parse_llm_model(required_string(llm, "pipeline.llm.model", "model")?)?;
    let thinking = optional_bool(llm, "pipeline.llm", "thinking", false)?;

    let tts = mapping(required(pipeline, "pipeline.tts", "tts")?, "pipeline.tts")?;
    deny_unknown(tts, "pipeline.tts", &["provider"])?;
    let tts_provider = parse_tts(required_string(tts, "pipeline.tts.provider", "provider")?)?;

    Ok(AgentConfig {
        name: name.to_string(),
        vad: vad_provider,
        vad_threshold,
        vad_min_speech_ms,
        vad_end_silence_ms,
        vad_preroll_ms,
        stt: stt_provider,
        stt_model,
        language,
        llm: llm_provider,
        llm_model: llm_model.to_string(),
        thinking,
        tts: tts_provider,
    })
}

fn mapping<'a>(value: &'a Value, field: &str) -> Result<&'a serde_yaml::Mapping> {
    value.as_mapping().ok_or_else(|| Error::Config {
        field: field.into(),
        message: "must be a mapping".into(),
    })
}

fn deny_unknown(map: &serde_yaml::Mapping, prefix: &str, allowed: &[&str]) -> Result<()> {
    for key in map.keys() {
        let Some(name) = key.as_str() else {
            return Err(Error::Config {
                field: prefix.into(),
                message: "keys must be strings".into(),
            });
        };
        if !allowed.contains(&name) {
            let field = if prefix == "." {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            };
            return Err(Error::Config {
                field,
                message: format!("unknown field (allowed: {})", allowed.join(", ")),
            });
        }
    }
    Ok(())
}

fn required<'a>(map: &'a serde_yaml::Mapping, field: &str, key: &str) -> Result<&'a Value> {
    map.get(key).ok_or_else(|| missing(field))
}

fn required_string<'a>(map: &'a serde_yaml::Mapping, field: &str, key: &str) -> Result<&'a str> {
    let value = required(map, field, key)?;
    value.as_str().ok_or_else(|| Error::Config {
        field: field.into(),
        message: "must be a string".into(),
    })
}

fn optional_u32(map: &serde_yaml::Mapping, prefix: &str, key: &str, default: u32) -> Result<u32> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    let field = format!("{prefix}.{key}");
    let n = if let Some(n) = value.as_u64() {
        n
    } else if let Some(n) = value.as_i64() {
        u64::try_from(n).map_err(|_| Error::Config {
            field: field.clone(),
            message: "must be a positive integer".into(),
        })?
    } else {
        return Err(Error::Config {
            field,
            message: "must be a positive integer".into(),
        });
    };
    u32::try_from(n).map_err(|_| Error::Config {
        field,
        message: "must be a positive integer".into(),
    })
}

fn optional_f32(map: &serde_yaml::Mapping, prefix: &str, key: &str, default: f32) -> Result<f32> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    let field = format!("{prefix}.{key}");
    if let Some(n) = value.as_f64() {
        return Ok(n as f32);
    }
    if let Some(n) = value.as_i64() {
        return Ok(n as f32);
    }
    Err(Error::Config {
        field,
        message: "must be a number".into(),
    })
}

fn validate_vad(
    threshold: f32,
    min_speech_ms: u32,
    end_silence_ms: u32,
    preroll_ms: u32,
) -> Result<()> {
    if !(threshold > 0.0 && threshold <= 1.0) {
        return Err(Error::Config {
            field: "pipeline.vad.threshold".into(),
            message: "must be in (0, 1]".into(),
        });
    }
    for (field, ms) in [
        ("pipeline.vad.min_speech_ms", min_speech_ms),
        ("pipeline.vad.end_silence_ms", end_silence_ms),
        ("pipeline.vad.preroll_ms", preroll_ms),
    ] {
        if ms == 0 {
            return Err(Error::Config {
                field: field.into(),
                message: "must be at least 1".into(),
            });
        }
    }
    Ok(())
}

fn optional_bool(
    map: &serde_yaml::Mapping,
    prefix: &str,
    key: &str,
    default: bool,
) -> Result<bool> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    value.as_bool().ok_or_else(|| Error::Config {
        field: format!("{prefix}.{key}"),
        message: "must be a boolean".into(),
    })
}

fn missing(field: &str) -> Error {
    Error::Config {
        field: field.into(),
        message: "missing field".into(),
    }
}

fn parse_vad(value: &str) -> Result<VadProvider> {
    match value {
        "silero" => Ok(VadProvider::Silero),
        other => Err(unsupported("pipeline.vad.provider", other, "silero")),
    }
}

fn parse_stt(value: &str) -> Result<SttProvider> {
    match value {
        "whisper.cpp" => Ok(SttProvider::WhisperCpp),
        other => Err(unsupported("pipeline.stt.provider", other, "whisper.cpp")),
    }
}

fn parse_stt_model(value: &str) -> Result<SttModel> {
    SttModel::parse(value).ok_or_else(|| {
        unsupported(
            "pipeline.stt.model",
            value,
            "small, medium, large-v3-turbo, medium-q5_0, large-v3-turbo-q5_0",
        )
    })
}

fn parse_language(value: &str) -> Result<String> {
    if is_supported_language(value) {
        Ok(value.to_string())
    } else {
        Err(unsupported(
            "pipeline.stt.language",
            value,
            "ISO code or \"auto\"",
        ))
    }
}

fn parse_llm(value: &str) -> Result<LlmProvider> {
    match value {
        "llama.cpp" => Ok(LlmProvider::LlamaCpp),
        other => Err(unsupported("pipeline.llm.provider", other, "llama.cpp")),
    }
}

fn parse_llm_model(value: &str) -> Result<&str> {
    match value {
        crate::llm::QWEN35_08B_ASSET
        | crate::llm::QWEN35_2B_ASSET
        | crate::llm::LLAMA_32_1B_ASSET => Ok(value),
        other => Err(unsupported(
            "pipeline.llm.model",
            other,
            "llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b",
        )),
    }
}

fn parse_tts(value: &str) -> Result<TtsProvider> {
    match value {
        "kokoro" => Ok(TtsProvider::Kokoro),
        other => Err(unsupported("pipeline.tts.provider", other, "kokoro")),
    }
}

fn unsupported(field: &str, got: &str, allowed: &str) -> Error {
    Error::Config {
        field: field.into(),
        message: format!("unsupported value {got:?} (allowed: {allowed})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::LANGUAGE_AUTO;

    fn tmp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "syllabix-config-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn generated_yaml_round_trips() {
        let yaml = AgentConfig::v0().to_yaml();
        assert!(yaml.contains("language: en"));
        assert!(yaml.contains("model: llama-3.2-1b"));
        assert!(yaml.contains("thinking: false"));
        assert!(yaml.contains("threshold: 0.5"));
        assert!(yaml.contains("min_speech_ms: 100"));
        assert!(yaml.contains("end_silence_ms: 350"));
        assert!(yaml.contains("preroll_ms: 200"));
        assert_eq!(AgentConfig::parse_yaml(&yaml).unwrap(), AgentConfig::v0());
    }

    #[test]
    fn example_config_parses_to_launch_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/demo-agent.yaml");
        let text = fs::read_to_string(&path).expect("examples/demo-agent.yaml exists");
        assert!(text.contains("#"), "example documents its keys in comments");
        let config = AgentConfig::parse_yaml(&text).expect("example parses");
        assert_eq!(config, AgentConfig::v0());
        assert!(!config.thinking, "thinking stays off by default");
        // The example is the minimal shape: no VAD tunables spelled out.
        assert!(
            !text.contains("threshold:") && !text.contains("min_speech_ms:"),
            "example stays minimal; launch defaults are omitted"
        );
    }

    #[test]
    fn example_config_loads_from_path_with_field_context() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/demo-agent.yaml");
        let config = AgentConfig::load_path(&path).expect("load example");
        assert_eq!(config.name, "demo-agent");
        assert_eq!(config.llm_model, "llama-3.2-1b");
        assert_eq!(config.language, "en");
    }

    #[test]
    fn yaml_vad_tunables_override_defaults() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("threshold: 0.5", "threshold: 0.7")
            .replace("min_speech_ms: 100", "min_speech_ms: 200");
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert!((cfg.vad_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.vad_min_speech_ms, 200);
        assert_eq!(cfg.vad_end_silence_ms, 350);
        assert_eq!(cfg.vad_settings().min_speech, Duration::from_millis(200));
    }

    #[test]
    fn vad_unknown_field_is_named() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("preroll_ms: 200", "preroll_ms: 200\n    sr: 16000");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.vad.sr"), "{err}");
    }

    #[test]
    fn vad_threshold_out_of_range_is_field_level() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("threshold: 0.5", "threshold: 1.5");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.vad.threshold"), "{err}");
    }

    #[test]
    fn launch_shape_with_language_parses() {
        let yaml = r#"
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: whisper.cpp
    model: small
    language: en
  llm:
    provider: llama.cpp
    model: llama-3.2-1b
  tts:
    provider: kokoro
"#;
        assert_eq!(AgentConfig::parse_yaml(yaml).unwrap(), AgentConfig::v0());
    }

    #[test]
    fn yaml_qwen_sizes_and_thinking_on() {
        let two = AgentConfig::v0()
            .to_yaml()
            .replace("model: llama-3.2-1b", "model: qwen3.5-2b")
            .replace("thinking: false", "thinking: true");
        let cfg = AgentConfig::parse_yaml(&two).unwrap();
        assert_eq!(cfg.llm_model, "qwen3.5-2b");
        assert!(cfg.thinking);
        let small = AgentConfig::v0()
            .to_yaml()
            .replace("model: llama-3.2-1b", "model: qwen3.5-0.8b");
        let cfg = AgentConfig::parse_yaml(&small).unwrap();
        assert_eq!(cfg.llm_model, "qwen3.5-0.8b");
        assert!(!cfg.thinking);
    }

    #[test]
    fn unknown_llm_model_is_field_level() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("model: llama-3.2-1b", "model: huge");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.model"), "{err}");
        assert!(err.to_string().contains("huge"), "{err}");
    }

    #[test]
    fn thinking_must_be_a_boolean() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("thinking: false", "thinking: nope");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.thinking"), "{err}");
        assert!(err.to_string().contains("boolean"), "{err}");
    }

    #[test]
    fn missing_language_is_field_level() {
        let yaml = r#"
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: whisper.cpp
    model: small
  llm:
    provider: llama.cpp
    model: qwen3.5-2b
  tts:
    provider: kokoro
"#;
        let err = AgentConfig::parse_yaml(yaml).unwrap_err();
        assert_eq!(err.to_string(), "pipeline.stt.language: missing field");
    }

    #[test]
    fn unsupported_language_is_field_level() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("language: en", "language: klingon");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.stt.language"), "{err}");
        assert!(err.to_string().contains("klingon"), "{err}");
    }

    #[test]
    fn stt_model_menu_parses_and_rejects_unknown_ids() {
        for model in SttModel::ALL {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: small", &format!("model: {}", model.as_str()));
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.stt_model, model);
        }
        for bad in ["tiny", "base", "large", "huge", "small.en"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: small", &format!("model: {bad}"));
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.stt.model"),
                "{bad}: {err}"
            );
            assert!(err.to_string().contains(bad), "{bad}: {err}");
        }
        assert_eq!(AgentConfig::v0().stt_model, SttModel::Small);
    }

    #[test]
    fn stt_language_menu_and_auto_parse() {
        for code in ["en", "fr", "de", "es", "ja", "zh", "yue", "haw"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("language: en", &format!("language: {code}"));
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.language, code);
        }
        let auto = AgentConfig::v0()
            .to_yaml()
            .replace("language: en", "language: auto");
        assert_eq!(
            AgentConfig::parse_yaml(&auto).unwrap().language,
            LANGUAGE_AUTO
        );
    }

    #[test]
    fn unknown_field_is_named() {
        let err = AgentConfig::parse_yaml("name: x\nfallback: true\n").unwrap_err();
        assert!(err.to_string().starts_with("fallback:"), "{err}");
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn wrong_provider_is_named() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("provider: kokoro", "provider: qwen");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.tts.provider"), "{err}");
        assert!(err.to_string().contains("qwen"), "{err}");
    }

    #[test]
    fn resolve_run_uses_builtins_when_missing() {
        let dir = tmp_dir("missing");
        assert_eq!(
            AgentConfig::resolve_for_run(&dir).unwrap(),
            AgentConfig::v0()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn init_writes_yaml_and_refuses_overwrite() {
        let dir = tmp_dir("init");
        let path = AgentConfig::write_init(&dir).unwrap();
        assert_eq!(path, dir.join(CONFIG_FILE_NAME));
        let loaded = AgentConfig::load_path(&path).unwrap();
        assert_eq!(loaded, AgentConfig::v0());
        let err = AgentConfig::write_init(&dir).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn field_errors_cover_each_layer() {
        let mut extra = AgentConfig::v0().to_yaml();
        extra.push_str("  extra: true\n");
        let cases: Vec<(String, &str)> = vec![
            ("name: 1\n".into(), "name:"),
            ("name: x\npipeline: 1\n".into(), "pipeline:"),
            (
                "name: x\npipeline:\n  vad: {provider: silero}\n".into(),
                "pipeline.stt:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("provider: silero", "provider: webrtc"),
                "pipeline.vad.provider:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("provider: whisper.cpp", "provider: deepgram"),
                "pipeline.stt.provider:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("model: small", "model: large"),
                "pipeline.stt.model:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("provider: llama.cpp", "provider: ollama"),
                "pipeline.llm.provider:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("model: llama-3.2-1b", "model: huge"),
                "pipeline.llm.model:",
            ),
            (extra, "pipeline.extra:"),
            ("name: x\n1: true\n".into(), "."),
        ];
        for (yaml, prefix) in cases {
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().starts_with(prefix) || err.to_string().contains(prefix),
                "yaml {yaml:?} -> {err} (want {prefix})"
            );
        }
    }

    #[test]
    fn resolve_run_loads_existing_yaml() {
        let dir = tmp_dir("resolve");
        AgentConfig::write_init(&dir).unwrap();
        let loaded = AgentConfig::resolve_for_run(&dir).unwrap();
        assert_eq!(loaded, AgentConfig::v0());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn non_string_provider_is_field_level() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("provider: kokoro", "provider: 1");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.tts.provider"), "{err}");
        assert!(err.to_string().contains("must be a string"), "{err}");
    }

    #[test]
    fn load_path_prefixes_the_filename() {
        let dir = tmp_dir("bad");
        let path = dir.join(CONFIG_FILE_NAME);
        fs::write(&path, "name: x\n").unwrap();
        let err = AgentConfig::load_path(&path).unwrap_err();
        assert!(err.to_string().contains(CONFIG_FILE_NAME), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
