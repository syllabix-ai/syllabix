//! Minimal `syllabix.yaml`. Missing file means [`crate::BuiltinDefaults`].

use std::fs;
use std::path::{Path, PathBuf};

use serde_yaml::Value;

use crate::defaults::{
    BuiltinDefaults, LlmProvider, SttModel, SttProvider, TtsProvider, VadProvider,
};
use crate::error::{Error, Result};
use crate::stt::STT_LANGUAGE;

/// File name written by `init` and optionally read by `run`.
pub const CONFIG_FILE_NAME: &str = "syllabix.yaml";

/// Validated v0 agent config. One provider per layer; language is a single STT code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Agent name (`demo-agent` by default).
    pub name: String,
    /// VAD provider.
    pub vad: VadProvider,
    /// STT provider.
    pub stt: SttProvider,
    /// STT model id.
    pub stt_model: SttModel,
    /// STT language code. v0 allows `en` only.
    pub language: String,
    /// LLM provider.
    pub llm: LlmProvider,
    /// LLM model id (`llama-3.2-1b`).
    pub llm_model: String,
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
            stt: defaults.stt,
            stt_model: defaults.stt_model,
            language: defaults.language.to_string(),
            llm: defaults.llm,
            llm_model: defaults.llm_model.to_string(),
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
  stt:
    provider: {stt}
    model: {stt_model}
    language: {language}
  llm:
    provider: {llm}
    model: {llm_model}
  tts:
    provider: {tts}
",
            name = self.name,
            vad = self.vad.as_str(),
            stt = self.stt.as_str(),
            stt_model = self.stt_model.as_str(),
            language = self.language,
            llm = self.llm.as_str(),
            llm_model = self.llm_model,
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
    deny_unknown(vad, "pipeline.vad", &["provider"])?;
    let vad_provider = parse_vad(required_string(vad, "pipeline.vad.provider", "provider")?)?;

    let stt = mapping(required(pipeline, "pipeline.stt", "stt")?, "pipeline.stt")?;
    deny_unknown(stt, "pipeline.stt", &["provider", "model", "language"])?;
    let stt_provider = parse_stt(required_string(stt, "pipeline.stt.provider", "provider")?)?;
    let stt_model = parse_stt_model(required_string(stt, "pipeline.stt.model", "model")?)?;
    let language = parse_language(required_string(stt, "pipeline.stt.language", "language")?)?;

    let llm = mapping(required(pipeline, "pipeline.llm", "llm")?, "pipeline.llm")?;
    deny_unknown(llm, "pipeline.llm", &["provider", "model"])?;
    let llm_provider = parse_llm(required_string(llm, "pipeline.llm.provider", "provider")?)?;
    let llm_model = parse_llm_model(required_string(llm, "pipeline.llm.model", "model")?)?;

    let tts = mapping(required(pipeline, "pipeline.tts", "tts")?, "pipeline.tts")?;
    deny_unknown(tts, "pipeline.tts", &["provider"])?;
    let tts_provider = parse_tts(required_string(tts, "pipeline.tts.provider", "provider")?)?;

    Ok(AgentConfig {
        name: name.to_string(),
        vad: vad_provider,
        stt: stt_provider,
        stt_model,
        language: language.to_string(),
        llm: llm_provider,
        llm_model: llm_model.to_string(),
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
    match value {
        "small" => Ok(SttModel::Small),
        other => Err(unsupported("pipeline.stt.model", other, "small")),
    }
}

fn parse_language(value: &str) -> Result<&str> {
    if value == STT_LANGUAGE {
        Ok(STT_LANGUAGE)
    } else {
        Err(unsupported("pipeline.stt.language", value, STT_LANGUAGE))
    }
}

fn parse_llm(value: &str) -> Result<LlmProvider> {
    match value {
        "llama.cpp" => Ok(LlmProvider::LlamaCpp),
        other => Err(unsupported("pipeline.llm.provider", other, "llama.cpp")),
    }
}

fn parse_llm_model(value: &str) -> Result<&str> {
    let expected = BuiltinDefaults::v0().llm_model;
    if value == expected {
        Ok(expected)
    } else {
        Err(unsupported("pipeline.llm.model", value, expected))
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
        assert_eq!(AgentConfig::parse_yaml(&yaml).unwrap(), AgentConfig::v0());
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
    model: llama-3.2-1b
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
            .replace("language: en", "language: es");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.stt.language"), "{err}");
        assert!(err.to_string().contains("es"), "{err}");
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
