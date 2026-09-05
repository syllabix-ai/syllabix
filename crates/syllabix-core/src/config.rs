//! Minimal `syllabix.yaml`. Missing file means [`crate::BuiltinDefaults`].

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_yaml::Value;

use crate::defaults::{
    BuiltinDefaults, LlmProvider, SttModel, SttProvider, TtsModel, TtsProvider, VadProvider,
};
use crate::error::{Error, Result};
use crate::language::is_supported as is_supported_language;
use crate::turn_debug::DEFAULT_TURN_DEBUG_DIR;
use crate::vad::{VadSettings, END_SILENCE, MIN_SPEECH, SPEECH_THRESHOLD, WHISPER_PREROLL};

/// File name written by `init` and optionally read by `run`.
pub const CONFIG_FILE_NAME: &str = "syllabix.yaml";

/// Default idle duration before the mic auto-mutes (`auto-timeout.mic_mute_ms`).
pub const DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS: u32 = 180_000;
/// Default idle duration before `run` exits (`auto-timeout.exit_ms`).
pub const DEFAULT_AUTO_TIMEOUT_EXIT_MS: u32 = 600_000;

/// Validated agent configuration with one provider selected for each pipeline stage.
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
    /// LLM model id (`lfm2.5-2.6b`, `llama-3.2-1b`, `qwen3.5-0.8b`, or `qwen3.5-2b`).
    pub llm_model: String,
    /// Qwen thinking. Default false; yaml `thinking: true` enables it on
    /// `qwen3.5-2b` only and is rejected for every other model.
    pub thinking: bool,
    /// LLM system prompt template (`pipeline.llm.system_prompt`). `{language}`
    /// is replaced at generate time with the STT language's English name.
    /// Omitted YAML uses the built-in prompt template.
    pub system_prompt: String,
    /// OpenAI-compatible endpoint (`pipeline.llm.base_url`). Required for
    /// `provider: online`, forbidden for `provider: local`; the API key never
    /// lives here.
    pub llm_base_url: Option<String>,
    /// Explicit developer-only opt-in for the API tool loop. It is
    /// never enabled by defaults or written by `init`.
    pub llm_developer_harness: bool,
    /// TTS provider (`local`; `online` is reserved and rejected).
    pub tts: TtsProvider,
    /// TTS model id (`kokoro`, `qwen3-0.6`, `qwen3-1.7`, or `pocket-tts`).
    pub tts_model: TtsModel,
    /// TTS language code (`en`). Qwen3-TTS speaks this language; Kokoro
    /// ignores it (the ONNX voice is fixed).
    pub tts_language: String,
    /// Diagnostics: write per-turn sidecars with the monotonic turn timeline
    /// (`diagnostics.timestamps`). `diagnostics.audio` implies this.
    pub diagnostics_timestamps: bool,
    /// Diagnostics: additionally write the turn WAVs (`diagnostics.audio`).
    /// Implies `diagnostics_timestamps`.
    pub diagnostics_audio: bool,
    /// Diagnostics output directory (`diagnostics.directory`).
    pub diagnostics_directory: PathBuf,
    /// Idle milliseconds before auto mic-mute (`auto-timeout.mic_mute_ms`).
    /// `0` disables. Default 180_000 (3 minutes).
    pub auto_timeout_mic_mute_ms: u32,
    /// Idle milliseconds before process exit (`auto-timeout.exit_ms`).
    /// `0` disables. Default 600_000 (10 minutes).
    pub auto_timeout_exit_ms: u32,
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
            system_prompt: crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
            llm_base_url: None,
            llm_developer_harness: false,
            tts: defaults.tts,
            tts_model: defaults.tts_model,
            tts_language: "en".to_string(),
            diagnostics_timestamps: false,
            diagnostics_audio: false,
            diagnostics_directory: PathBuf::from(DEFAULT_TURN_DEBUG_DIR),
            auto_timeout_mic_mute_ms: DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS,
            auto_timeout_exit_ms: DEFAULT_AUTO_TIMEOUT_EXIT_MS,
        }
    }

    /// Diagnostics recording is on (sidecars and/or turn WAVs).
    pub fn diagnostics_enabled(&self) -> bool {
        self.diagnostics_timestamps || self.diagnostics_audio
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
    ///
    /// The `diagnostics` block is emitted only when non-default, matching how
    /// `base_url` is omitted for the zero-config local LLM.
    pub fn to_yaml(&self) -> String {
        let diagnostics_block = self.render_diagnostics_block();
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
    system_prompt: {system_prompt}
{base_url_line}  tts:
    provider: {tts}
    model: {tts_model}
    language: {tts_language}
{diagnostics_block}",
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
            system_prompt = yaml_double_quoted(&self.system_prompt),
            base_url_line = self
                .llm_base_url
                .as_deref()
                .map(|url| format!("    base_url: {url}\n"))
                .unwrap_or_default(),
            tts = self.tts.as_str(),
            tts_model = self.tts_model.as_str(),
            tts_language = self.tts_language,
            diagnostics_block = diagnostics_block,
        )
    }

    /// `diagnostics:` yaml block; empty string when everything is default.
    fn render_diagnostics_block(&self) -> String {
        let directory_default = PathBuf::from(DEFAULT_TURN_DEBUG_DIR);
        if !self.diagnostics_enabled() && self.diagnostics_directory == directory_default {
            return String::new();
        }
        let directory_line = if self.diagnostics_directory == directory_default {
            String::new()
        } else {
            format!("  directory: {}\n", self.diagnostics_directory.display())
        };
        format!(
            "diagnostics:\n  timestamps: {}\n  audio: {}\n{}",
            self.diagnostics_timestamps, self.diagnostics_audio, directory_line
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
    deny_unknown(
        root,
        ".",
        &["name", "pipeline", "diagnostics", "auto-timeout"],
    )?;
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
    // Moonshine is English-only: any other language (including `auto`) would
    // mistranslate the `{language}` prompt pin, so reject it here where the
    // field is named.
    if stt_model.is_moonshine() && language != crate::moonshine::LANGUAGE {
        return Err(Error::Config {
            field: "pipeline.stt.language".into(),
            message: format!("unsupported value {language:?} (allowed: \"en\" with this model)"),
        });
    }

    let llm = mapping(required(pipeline, "pipeline.llm", "llm")?, "pipeline.llm")?;
    deny_unknown(
        llm,
        "pipeline.llm",
        &[
            "provider",
            "model",
            "thinking",
            "system_prompt",
            "base_url",
            "developer_harness",
        ],
    )?;
    let llm_provider = parse_llm(required_string(llm, "pipeline.llm.provider", "provider")?)?;
    let llm_model = parse_llm_model(
        llm_provider,
        required_string(llm, "pipeline.llm.model", "model")?,
    )?;
    let thinking = optional_bool(llm, "pipeline.llm", "thinking", false)?;
    // Thinking is a `qwen3.5-2b` capability only: the smaller Qwen and Llama
    // never enable chain-of-thought, so yaml must not ask them to.
    if thinking && !(llm_provider == LlmProvider::Local && llm_model == crate::llm::QWEN35_2B_ASSET)
    {
        return Err(Error::Config {
            field: "pipeline.llm.thinking".into(),
            message: "is only valid with pipeline.llm.model: qwen3.5-2b".into(),
        });
    }
    let system_prompt = match optional_string(llm, "pipeline.llm", "system_prompt")? {
        Some(value) => parse_system_prompt(&value)?,
        None => crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE.to_string(),
    };
    // The endpoint is explicit for `online` providers such as OpenAI, Groq, Ollama,
    // vLLM, llama-server); the key may not live here either way —
    // `SYLLABIX_LLM_API_KEY` env only, never yaml, never `.env`.
    let llm_base_url = resolve_llm_base_url(llm, llm_provider)?;
    let llm_developer_harness = optional_bool(llm, "pipeline.llm", "developer_harness", false)?;
    let local_lfm_harness =
        llm_provider == LlmProvider::Local && llm_model == crate::llm::LFM25_26B_ASSET;
    if llm_developer_harness && llm_provider != LlmProvider::Online && !local_lfm_harness {
        return Err(Error::Config {
            field: "pipeline.llm.developer_harness".into(),
            message:
                "is only valid when pipeline.llm.provider is online, or for local model lfm2.5-2.6b"
                    .into(),
        });
    }

    let tts = mapping(required(pipeline, "pipeline.tts", "tts")?, "pipeline.tts")?;
    deny_unknown(tts, "pipeline.tts", &["provider", "model", "language"])?;
    // `local` runs weights in-process. `online` is reserved and rejected until
    // an online TTS implementation exists.
    let tts_provider = parse_tts(required_string(tts, "pipeline.tts.provider", "provider")?)?;
    let tts_model = parse_tts_model(
        tts_provider,
        required_string(tts, "pipeline.tts.model", "model")?,
    )?;
    // Qwen3-TTS speaks this language; Kokoro ignores it. `auto` is an
    // STT concept and stays rejected here.
    let tts_language = match optional_string(tts, "pipeline.tts", "language")? {
        Some(value) => parse_tts_language(&value)?,
        None => "en".to_string(),
    };

    // Diagnostics are configured here instead of with a CLI flag. `audio: true`
    // implies `timestamps: true` — WAVs always ship with their sidecar.
    let (diagnostics_timestamps, diagnostics_audio, diagnostics_directory) =
        parse_diagnostics(root)?;
    let (auto_timeout_mic_mute_ms, auto_timeout_exit_ms) = parse_auto_timeout(root)?;

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
        system_prompt,
        llm_base_url,
        llm_developer_harness,
        tts: tts_provider,
        tts_model,
        tts_language,
        diagnostics_timestamps,
        diagnostics_audio,
        diagnostics_directory,
        auto_timeout_mic_mute_ms,
        auto_timeout_exit_ms,
    })
}

/// `auto-timeout: {mic_mute_ms, exit_ms}` — optional; defaults 180000 / 600000.
/// `0` disables that timer. When both are positive, `exit_ms` must be greater
/// than `mic_mute_ms`.
fn parse_auto_timeout(root: &serde_yaml::Mapping) -> Result<(u32, u32)> {
    let Some(value) = root.get("auto-timeout") else {
        return Ok((
            DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS,
            DEFAULT_AUTO_TIMEOUT_EXIT_MS,
        ));
    };
    let block = mapping(value, "auto-timeout")?;
    deny_unknown(block, "auto-timeout", &["mic_mute_ms", "exit_ms"])?;
    let mic_mute_ms = optional_timeout_ms(
        block,
        "auto-timeout",
        "mic_mute_ms",
        DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS,
    )?;
    let exit_ms = optional_timeout_ms(
        block,
        "auto-timeout",
        "exit_ms",
        DEFAULT_AUTO_TIMEOUT_EXIT_MS,
    )?;
    if mic_mute_ms > 0 && exit_ms > 0 && exit_ms <= mic_mute_ms {
        return Err(Error::Config {
            field: "auto-timeout.exit_ms".into(),
            message: format!("must be greater than auto-timeout.mic_mute_ms ({mic_mute_ms})"),
        });
    }
    Ok((mic_mute_ms, exit_ms))
}

/// `diagnostics: {timestamps, audio, directory}` — all optional, all default
/// off/`target/turn-debug`. `audio: true` forces `timestamps: true`.
fn parse_diagnostics(root: &serde_yaml::Mapping) -> Result<(bool, bool, PathBuf)> {
    let Some(value) = root.get("diagnostics") else {
        return Ok((false, false, PathBuf::from(DEFAULT_TURN_DEBUG_DIR)));
    };
    let diag = mapping(value, "diagnostics")?;
    deny_unknown(diag, "diagnostics", &["timestamps", "audio", "directory"])?;
    let timestamps = optional_bool(diag, "diagnostics", "timestamps", false)?;
    let audio = optional_bool(diag, "diagnostics", "audio", false)?;
    let directory = match optional_string(diag, "diagnostics", "directory")? {
        Some(dir) if !dir.trim().is_empty() => PathBuf::from(dir),
        Some(_) => {
            return Err(Error::Config {
                field: "diagnostics.directory".into(),
                message: "must be a non-empty string".into(),
            });
        }
        None => PathBuf::from(DEFAULT_TURN_DEBUG_DIR),
    };
    Ok((timestamps || audio, audio, directory))
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

fn optional_timeout_ms(
    map: &serde_yaml::Mapping,
    prefix: &str,
    key: &str,
    default: u32,
) -> Result<u32> {
    let Some(value) = map.get(key) else {
        return Ok(default);
    };
    let field = format!("{prefix}.{key}");
    let n = if let Some(n) = value.as_u64() {
        n
    } else if let Some(n) = value.as_i64() {
        u64::try_from(n).map_err(|_| Error::Config {
            field: field.clone(),
            message: "must be a non-negative integer".into(),
        })?
    } else {
        return Err(Error::Config {
            field,
            message: "must be a non-negative integer".into(),
        });
    };
    u32::try_from(n).map_err(|_| Error::Config {
        field,
        message: "must be a non-negative integer".into(),
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

fn optional_string(map: &serde_yaml::Mapping, prefix: &str, key: &str) -> Result<Option<String>> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|s| s.to_string())
        .map(Some)
        .ok_or_else(|| Error::Config {
            field: format!("{prefix}.{key}"),
            message: "must be a string".into(),
        })
}

fn parse_system_prompt(value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::Config {
            field: "pipeline.llm.system_prompt".into(),
            message: "must be a non-empty string".into(),
        });
    }
    Ok(trimmed.to_string())
}

/// Double-quoted yaml scalar so colons and `{language}` stay one value.
fn yaml_double_quoted(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
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
        "local" => Ok(SttProvider::Local),
        other => Err(unsupported("pipeline.stt.provider", other, "local")),
    }
}

fn parse_stt_model(value: &str) -> Result<SttModel> {
    SttModel::parse(value).ok_or_else(|| {
        unsupported(
            "pipeline.stt.model",
            value,
            "whisper-small, whisper-medium, whisper-large-v3-turbo, whisper-medium-q5_0, whisper-large-v3-turbo-q5_0, moonshine-streaming-small, moonshine-streaming-medium",
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
        "local" => Ok(LlmProvider::Local),
        "online" => Ok(LlmProvider::Online),
        other => Err(unsupported("pipeline.llm.provider", other, "local, online")),
    }
}

/// `pipeline.llm.model`: the four binary-supported GGUF ids for `local`; any
/// non-empty id for `online` (the endpoint decides what it serves).
fn parse_llm_model(provider: LlmProvider, value: &str) -> Result<String> {
    match provider {
        LlmProvider::Local => match value {
            crate::llm::QWEN35_08B_ASSET
            | crate::llm::QWEN35_2B_ASSET
            | crate::llm::LLAMA_32_1B_ASSET
            | crate::llm::LFM25_26B_ASSET => Ok(value.to_string()),
            other => Err(unsupported(
                "pipeline.llm.model",
                other,
                "lfm2.5-2.6b, llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b",
            )),
        },
        LlmProvider::Online => {
            if value.trim().is_empty() {
                Err(Error::Config {
                    field: "pipeline.llm.model".into(),
                    message: "must be a non-empty model id".into(),
                })
            } else {
                Ok(value.to_string())
            }
        }
    }
}

/// `pipeline.llm.base_url`: required (non-empty) for `provider: online`,
/// forbidden for `provider: local` — the local engine has nothing to point at.
fn resolve_llm_base_url(
    map: &serde_yaml::Mapping,
    provider: LlmProvider,
) -> Result<Option<String>> {
    match provider {
        LlmProvider::Local => {
            if map.get("base_url").is_some() {
                return Err(Error::Config {
                    field: "pipeline.llm.base_url".into(),
                    message: "is only valid when pipeline.llm.provider is online".into(),
                });
            }
            Ok(None)
        }
        LlmProvider::Online => {
            let value = map.get("base_url").ok_or_else(|| Error::Config {
                field: "pipeline.llm.base_url".into(),
                message: "is required when pipeline.llm.provider is online".into(),
            })?;
            let raw = value.as_str().ok_or_else(|| Error::Config {
                field: "pipeline.llm.base_url".into(),
                message: "must be a string".into(),
            })?;
            crate::openai::validate_base_url(raw).map_err(|message| Error::Config {
                field: "pipeline.llm.base_url".into(),
                message,
            })?;
            Ok(Some(raw.to_string()))
        }
    }
}

fn parse_tts(value: &str) -> Result<TtsProvider> {
    match value {
        "local" => Ok(TtsProvider::Local),
        "online" => Err(Error::Config {
            field: "pipeline.tts.provider".into(),
            message: "online TTS is not supported yet (weights never leave the machine today); \
                      use provider \"local\""
                .into(),
        }),
        other => Err(unsupported("pipeline.tts.provider", other, "local")),
    }
}

/// `pipeline.tts.model`: the local weight menu. `pocket-tts` is the default;
/// other model identifiers fetch their assets on first use only.
fn parse_tts_model(provider: TtsProvider, value: &str) -> Result<TtsModel> {
    match provider {
        TtsProvider::Local => TtsModel::parse(value).ok_or_else(|| {
            unsupported(
                "pipeline.tts.model",
                value,
                "kokoro, qwen3-0.6, qwen3-1.7, pocket-tts",
            )
        }),
        // parse_tts already rejected `online`; this arm keeps the type total.
        TtsProvider::Online => Err(Error::Config {
            field: "pipeline.tts.provider".into(),
            message: "online TTS is not supported yet; use provider \"local\"".into(),
        }),
    }
}

/// `pipeline.tts.language`: same supported-code vocabulary as STT, but `auto`
/// is an STT-only concept.
fn parse_tts_language(value: &str) -> Result<String> {
    if value == "auto" {
        return Err(unsupported(
            "pipeline.tts.language",
            value,
            "ISO code (\"auto\" is STT-only)",
        ));
    }
    if is_supported_language(value) {
        Ok(value.to_string())
    } else {
        Err(unsupported("pipeline.tts.language", value, "ISO code"))
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
        assert!(yaml.contains("model: lfm2.5-2.6b"));
        assert!(yaml.contains("thinking: false"));
        assert!(yaml.contains("system_prompt:"));
        assert!(yaml.contains("{language}"));
        assert!(yaml.contains("model: pocket-tts"));
        assert!(yaml.contains("threshold: 0.5"));
        assert!(yaml.contains("min_speech_ms: 100"));
        assert!(yaml.contains("end_silence_ms: 350"));
        assert!(yaml.contains("preroll_ms: 200"));
        assert_eq!(AgentConfig::parse_yaml(&yaml).unwrap(), AgentConfig::v0());
    }

    #[test]
    fn developer_harness_is_off_by_default_and_gated() {
        assert!(!AgentConfig::v0().llm_developer_harness);
        let enabled = AgentConfig::parse_yaml(
            r#"
name: harness
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: online, model: gpt-test, base_url: https://example.test/v1, developer_harness: true }
  tts: { provider: local, model: kokoro }
"#,
        )
        .expect("online developer harness parses");
        assert!(enabled.llm_developer_harness);
        let local = AgentConfig::parse_yaml(
            r#"
name: harness
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b, developer_harness: true }
  tts: { provider: local, model: kokoro }
"#,
        )
        .expect("admitted local LFM harness parses");
        assert!(local.llm_developer_harness);
        let err = AgentConfig::parse_yaml(
            r#"
name: harness
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: llama-3.2-1b, developer_harness: true }
  tts: { provider: local, model: kokoro }
"#,
        )
        .expect_err("non-admitted local harness is rejected");
        assert!(err.to_string().contains("developer_harness"));
    }

    #[test]
    fn auto_timeout_defaults_when_omitted() {
        let cfg = AgentConfig::v0();
        assert_eq!(
            cfg.auto_timeout_mic_mute_ms,
            DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS
        );
        assert_eq!(cfg.auto_timeout_exit_ms, DEFAULT_AUTO_TIMEOUT_EXIT_MS);
        let parsed = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
"#,
        )
        .unwrap();
        assert_eq!(parsed.auto_timeout_mic_mute_ms, 180_000);
        assert_eq!(parsed.auto_timeout_exit_ms, 600_000);
    }

    #[test]
    fn auto_timeout_zero_disables_each_timer() {
        let cfg = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
auto-timeout:
  mic_mute_ms: 0
  exit_ms: 0
"#,
        )
        .unwrap();
        assert_eq!(cfg.auto_timeout_mic_mute_ms, 0);
        assert_eq!(cfg.auto_timeout_exit_ms, 0);
    }

    #[test]
    fn auto_timeout_custom_values_parse() {
        let cfg = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
auto-timeout:
  mic_mute_ms: 60000
  exit_ms: 120000
"#,
        )
        .unwrap();
        assert_eq!(cfg.auto_timeout_mic_mute_ms, 60_000);
        assert_eq!(cfg.auto_timeout_exit_ms, 120_000);
    }

    #[test]
    fn auto_timeout_rejects_exit_not_after_mute() {
        let err = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
auto-timeout:
  mic_mute_ms: 100000
  exit_ms: 100000
"#,
        )
        .expect_err("exit must be > mic_mute when both set");
        assert!(err.to_string().contains("auto-timeout.exit_ms"), "{err}");
    }

    #[test]
    fn auto_timeout_rejects_unknown_keys_and_negatives() {
        let err = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
auto-timeout:
  idle_ms: 1
"#,
        )
        .expect_err("unknown key");
        assert!(err.to_string().contains("unknown field"), "{err}");
        let err = AgentConfig::parse_yaml(
            r#"
name: demo
pipeline:
  vad: { provider: silero }
  stt: { provider: local, model: whisper-small, language: en }
  llm: { provider: local, model: lfm2.5-2.6b }
  tts: { provider: local, model: pocket-tts }
auto-timeout:
  mic_mute_ms: -1
"#,
        )
        .expect_err("negative");
        assert!(
            err.to_string().contains("auto-timeout.mic_mute_ms"),
            "{err}"
        );
    }

    #[test]
    fn example_config_parses_to_launch_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/demo-agent.yaml");
        let text = fs::read_to_string(&path).expect("examples/demo-agent.yaml exists");
        assert!(text.contains("#"), "example documents its keys in comments");
        let config = AgentConfig::parse_yaml(&text).expect("example parses");
        assert_eq!(config, AgentConfig::v0());
        assert!(!config.thinking, "thinking stays off by default");
        assert_eq!(
            config.system_prompt,
            crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE
        );
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
        assert_eq!(config.llm_model, "lfm2.5-2.6b");
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
    provider: local
    model: whisper-small
    language: en
  llm:
    provider: local
    model: lfm2.5-2.6b
  tts:
    provider: local
    model: pocket-tts
"#;
        assert_eq!(AgentConfig::parse_yaml(yaml).unwrap(), AgentConfig::v0());
    }

    #[test]
    fn yaml_qwen_sizes_and_thinking_on() {
        let two = AgentConfig::v0()
            .to_yaml()
            .replace("model: lfm2.5-2.6b", "model: qwen3.5-2b")
            .replace("thinking: false", "thinking: true");
        let cfg = AgentConfig::parse_yaml(&two).unwrap();
        assert_eq!(cfg.llm_model, "qwen3.5-2b");
        assert!(cfg.thinking);
        let small = AgentConfig::v0()
            .to_yaml()
            .replace("model: lfm2.5-2.6b", "model: qwen3.5-0.8b");
        let cfg = AgentConfig::parse_yaml(&small).unwrap();
        assert_eq!(cfg.llm_model, "qwen3.5-0.8b");
        assert!(!cfg.thinking);

        let llama = AgentConfig::v0()
            .to_yaml()
            .replace("model: lfm2.5-2.6b", "model: llama-3.2-1b");
        let cfg = AgentConfig::parse_yaml(&llama).unwrap();
        assert_eq!(cfg.llm_model, "llama-3.2-1b");
        assert!(!cfg.thinking);
        let lfm_default = AgentConfig::v0();
        assert_eq!(lfm_default.llm_model, "lfm2.5-2.6b");
        assert!(!lfm_default.thinking);
    }

    #[test]
    fn thinking_true_is_rejected_except_on_the_2b_model() {
        for model in ["lfm2.5-2.6b", "llama-3.2-1b", "qwen3.5-0.8b"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: lfm2.5-2.6b", &format!("model: {model}"))
                .replace("thinking: false", "thinking: true");
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(err.to_string().contains("pipeline.llm.thinking"), "{err}");
        }
    }

    #[test]
    fn unknown_llm_model_is_field_level() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("model: lfm2.5-2.6b", "model: huge");
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
    fn yaml_system_prompt_overrides_the_launch_default() {
        let yaml = AgentConfig::v0().to_yaml().replace(
            &format!(
                "system_prompt: {}",
                super::yaml_double_quoted(crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE)
            ),
            "system_prompt: \"You are a cooking coach. Reply in spoken {language}.\"",
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert_eq!(
            cfg.system_prompt,
            "You are a cooking coach. Reply in spoken {language}."
        );
        let round = AgentConfig::parse_yaml(&cfg.to_yaml()).unwrap();
        assert_eq!(round.system_prompt, cfg.system_prompt);
    }

    #[test]
    fn yaml_system_prompt_literal_block_is_accepted() {
        let yaml = r#"
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: local
    model: whisper-small
    language: en
  llm:
    provider: local
    model: llama-3.2-1b
    system_prompt: |
      You are a cooking coach.
      Keep answers short enough to say aloud.
  tts:
    provider: local
    model: kokoro
"#;
        let cfg = AgentConfig::parse_yaml(yaml).unwrap();
        assert_eq!(
            cfg.system_prompt,
            "You are a cooking coach.\nKeep answers short enough to say aloud."
        );
    }

    #[test]
    fn empty_system_prompt_is_field_level() {
        let yaml = AgentConfig::v0().to_yaml().replace(
            &format!(
                "system_prompt: {}",
                super::yaml_double_quoted(crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE)
            ),
            "system_prompt: \"   \"",
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(
            err.to_string().contains("pipeline.llm.system_prompt"),
            "{err}"
        );
        assert!(err.to_string().contains("non-empty"), "{err}");
    }

    #[test]
    fn system_prompt_must_be_a_string() {
        let yaml = AgentConfig::v0().to_yaml().replace(
            &format!(
                "system_prompt: {}",
                super::yaml_double_quoted(crate::llm::VOICE_SYSTEM_PROMPT_TEMPLATE)
            ),
            "system_prompt: 1",
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(
            err.to_string().contains("pipeline.llm.system_prompt"),
            "{err}"
        );
        assert!(err.to_string().contains("string"), "{err}");
    }

    #[test]
    fn yaml_system_prompt_round_trips_quotes_and_whitespace() {
        let mut cfg = AgentConfig::v0();
        cfg.system_prompt = "Say \"hi\"\nthen a tab\there and a slash \\ and cr\r.".into();
        let yaml = cfg.to_yaml();
        assert!(yaml.contains("\\\""), "{yaml}");
        assert!(yaml.contains("\\n"), "{yaml}");
        assert!(yaml.contains("\\t"), "{yaml}");
        assert!(yaml.contains("\\\\"), "{yaml}");
        assert!(yaml.contains("\\r"), "{yaml}");
        let round = AgentConfig::parse_yaml(&yaml).unwrap();
        assert_eq!(round.system_prompt, cfg.system_prompt);
    }

    fn online_yaml(model: &str, base_url: &str) -> String {
        format!(
            r#"
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: local
    model: whisper-small
    language: en
  llm:
    provider: online
    model: "{model}"
    base_url: {base_url}
  tts:
    provider: local
    model: kokoro
"#
        )
    }

    #[test]
    fn online_provider_accepts_any_model_id() {
        let cfg = AgentConfig::parse_yaml(&online_yaml("gpt-4o-mini", "https://api.openai.com/v1"))
            .unwrap();
        assert_eq!(cfg.llm, LlmProvider::Online);
        assert_eq!(cfg.llm_model, "gpt-4o-mini");
        assert_eq!(
            cfg.llm_base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        // The endpoint decides what it serves; ids are not menu-restricted.
        assert_eq!(
            AgentConfig::parse_yaml(&online_yaml("llama3.2:1b", "http://127.0.0.1:11434/v1"))
                .unwrap()
                .llm_model,
            "llama3.2:1b"
        );
    }

    #[test]
    fn online_empty_model_is_field_level() {
        // A bare empty yaml value parses as null ("must be a string"); a
        // whitespace string must hit the non-empty rule.
        let err =
            AgentConfig::parse_yaml(&online_yaml("   ", "https://api.openai.com/v1")).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.model"), "{err}");
        assert!(err.to_string().contains("non-empty"), "{err}");
    }

    #[test]
    fn online_requires_a_base_url() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace(
                "  llm:\n    provider: local",
                "  llm:\n    provider: online",
            )
            .replace("model: lfm2.5-2.6b", "model: gpt-4o-mini");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.base_url"), "{err}");
        assert!(err.to_string().contains("required"), "{err}");
    }

    #[test]
    fn local_provider_still_enforces_the_gguf_menu() {
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("model: lfm2.5-2.6b", "model: gpt-4o-mini");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.model"), "{err}");
    }

    #[test]
    fn local_rejects_base_url() {
        // The in-process engine has nothing to point at; the field is
        // reserved for `online`.
        let yaml = AgentConfig::v0().to_yaml().replace(
            "    thinking: false",
            "    thinking: false\n    base_url: https://api.groq.com/openai/v1",
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.base_url"), "{err}");
        assert!(err.to_string().contains("online"), "{err}");
    }

    #[test]
    fn online_base_url_round_trips() {
        let url = "https://api.groq.com/openai/v1";
        let cfg = AgentConfig::parse_yaml(&online_yaml("gpt-4o-mini", url)).unwrap();
        assert_eq!(cfg.llm_base_url.as_deref(), Some(url));
        let round = AgentConfig::parse_yaml(&cfg.to_yaml()).unwrap();
        assert_eq!(round, cfg);
        assert!(cfg.to_yaml().contains(&format!("base_url: {url}")));
        // The default run yaml stays byte-identical: no key, no base_url line.
        assert!(!AgentConfig::v0().to_yaml().contains("base_url"));
        assert!(AgentConfig::v0().to_yaml().contains("provider: local"));
    }

    #[test]
    fn online_base_url_must_be_an_absolute_http_url() {
        for bad in [
            "api.example.com/v1",
            "ftp://api.example.com",
            "file:///tmp/sock",
            "https://user:pass@api.example.com",
            "",
        ] {
            let err = AgentConfig::parse_yaml(&online_yaml("gpt-4o-mini", bad)).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.llm.base_url"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn non_string_base_url_is_field_level() {
        let yaml = online_yaml("gpt-4o-mini", "https://api.openai.com/v1")
            .replace("base_url: https://api.openai.com/v1\n", "base_url: 7\n");
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.llm.base_url"), "{err}");
        assert!(err.to_string().contains("must be a string"), "{err}");
    }

    #[test]
    fn missing_language_is_field_level() {
        let yaml = r#"
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: local
    model: whisper-small
  llm:
    provider: local
    model: qwen3.5-2b
  tts:
    provider: local
    model: kokoro
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
            let yaml = AgentConfig::v0().to_yaml().replace(
                "model: whisper-small",
                &format!("model: {}", model.as_str()),
            );
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.stt_model, model);
        }
        for bad in ["tiny", "base", "large", "huge", "small.en"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: whisper-small", &format!("model: {bad}"));
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
    fn moonshine_model_requires_english_under_local_stt() {
        // Moonshine is English-only even though its provider is uniformly local.
        for (id, expected) in [
            (
                "moonshine-streaming-small",
                SttModel::MoonshineStreamingSmall,
            ),
            (
                "moonshine-streaming-medium",
                SttModel::MoonshineStreamingMedium,
            ),
        ] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: whisper-small", &format!("model: {id}"))
                .replace("language: en", "language: fr");
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.stt.language"),
                "{id}: {err}"
            );
            // The local Moonshine shape parses.
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: whisper-small", &format!("model: {id}"));
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.stt, SttProvider::Local);
            assert_eq!(cfg.stt_model, expected);
            assert_eq!(cfg.language, "en");
        }
    }

    #[test]
    fn stt_language_menu_and_auto_parse() {
        // Scoped to the STT block: `pipeline.tts.language` exists too.
        let stt_line = "  stt:\n    provider: local\n    model: whisper-small\n    language: en";
        for code in ["en", "fr", "de", "es", "ja", "zh", "yue", "haw"] {
            let swapped = stt_line.replace("language: en", &format!("language: {code}"));
            let yaml = AgentConfig::v0().to_yaml().replace(stt_line, &swapped);
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.language, code);
        }
        let swapped = stt_line.replace("language: en", "language: auto");
        let auto = AgentConfig::v0().to_yaml().replace(stt_line, &swapped);
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
        let yaml = AgentConfig::v0().to_yaml().replace(
            "  tts:\n    provider: local",
            "  tts:\n    provider: pansori",
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.tts.provider"), "{err}");
        assert!(err.to_string().contains("pansori"), "{err}");
        assert!(err.to_string().contains("local"), "{err}");
    }

    #[test]
    fn tts_model_menu_parses_and_rejects_unknown_ids() {
        for model in TtsModel::ALL {
            let yaml = AgentConfig::v0().to_yaml().replace(
                "model: pocket-tts\n    language",
                &format!("model: {}\n    language", model.as_str()),
            );
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.tts_model, model);
        }
        for bad in ["qwen", "neutts", "large"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: pocket-tts", &format!("model: {bad}"));
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.tts.model"),
                "{bad}: {err}"
            );
            assert!(err.to_string().contains(bad), "{bad}: {err}");
        }
        // Retired engine names are not provider values; the error
        // points at the new key.
        for legacy in ["kokoro", "qwen"] {
            let yaml = AgentConfig::v0().to_yaml().replace(
                "  tts:\n    provider: local",
                &format!("  tts:\n    provider: {legacy}"),
            );
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.tts.provider"),
                "{legacy}: {err}"
            );
            assert!(err.to_string().contains("local"), "{legacy}: {err}");
        }
    }

    #[test]
    fn online_tts_fails_fast_with_the_posture_hint() {
        let yaml = AgentConfig::v0().to_yaml().replace(
            "  tts:\n    provider: local",
            "  tts:\n    provider: online",
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("pipeline.tts.provider"), "{err}");
        assert!(err.to_string().contains("not supported yet"), "{err}");
        // Even a valid model id cannot sneak through an unsupported posture.
        let yaml = yaml.replace("model: pocket-tts", "model: qwen3-0.6");
        assert!(AgentConfig::parse_yaml(&yaml).is_err());
    }

    #[test]
    fn zero_config_uses_pocket_tts_by_construction() {
        let cfg = AgentConfig::v0();
        assert_eq!(cfg.tts, TtsProvider::Local);
        assert_eq!(cfg.tts_model, TtsModel::PocketTts);
    }

    #[test]
    fn qwen_tts_language_is_optional_iso_without_auto() {
        for code in ["en", "fr", "de", "es", "ja", "zh"] {
            let yaml = AgentConfig::v0()
                .to_yaml()
                .replace("model: pocket-tts", "model: qwen3-0.6")
                .replace("language: en\n", &format!("language: {code}\n"));
            let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
            assert_eq!(cfg.tts_model, TtsModel::Qwen06);
            assert_eq!(cfg.tts_language, code);
        }
        let yaml = AgentConfig::v0()
            .to_yaml()
            .replace("model: pocket-tts", "model: qwen3-1.7");
        assert!(
            yaml.contains("  tts:\n    provider: local\n    model: qwen3-1.7\n    language: en")
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert_eq!(cfg.tts_language, "en");

        for bad in ["auto", "tlh"] {
            let yaml = AgentConfig::v0().to_yaml().replace(
                "  tts:\n    provider: local\n    model: pocket-tts\n    language: en",
                &format!("  tts:\n    provider: local\n    model: pocket-tts\n    language: {bad}"),
            );
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains("pipeline.tts.language"),
                "{bad}: {err}"
            );
        }
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
    fn diagnostics_default_to_off_and_the_launch_directory() {
        let cfg = AgentConfig::v0();
        assert!(!cfg.diagnostics_timestamps);
        assert!(!cfg.diagnostics_audio);
        assert_eq!(
            cfg.diagnostics_directory,
            PathBuf::from(DEFAULT_TURN_DEBUG_DIR)
        );
        assert!(!cfg.diagnostics_enabled());
        // The zero-config yaml stays byte-identical: no diagnostics block.
        assert!(!AgentConfig::v0().to_yaml().contains("diagnostics"));
    }

    #[test]
    fn diagnostics_block_parses_every_key() {
        let yaml = format!(
            "{}\ndiagnostics:\n  timestamps: true\n  audio: true\n  directory: /tmp/turns\n",
            AgentConfig::v0().to_yaml()
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert!(cfg.diagnostics_timestamps);
        assert!(cfg.diagnostics_audio);
        assert_eq!(cfg.diagnostics_directory, PathBuf::from("/tmp/turns"));
        assert!(cfg.diagnostics_enabled());
    }

    #[test]
    fn diagnostics_audio_implies_timestamps() {
        let yaml = format!(
            "{}\ndiagnostics:\n  timestamps: false\n  audio: true\n",
            AgentConfig::v0().to_yaml()
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert!(cfg.diagnostics_audio);
        assert!(
            cfg.diagnostics_timestamps,
            "audio capture always ships with its sidecar"
        );
        // Round-trip keeps the implied timestamps explicit.
        let round = AgentConfig::parse_yaml(&cfg.to_yaml()).unwrap();
        assert_eq!(round, cfg);
        assert!(cfg.to_yaml().contains("timestamps: true"));
        assert!(cfg.to_yaml().contains("audio: true"));
    }

    #[test]
    fn diagnostics_timestamps_only_omits_wavs_flag_round_trip() {
        let yaml = format!(
            "{}\ndiagnostics:\n  timestamps: true\n",
            AgentConfig::v0().to_yaml()
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert!(cfg.diagnostics_timestamps);
        assert!(!cfg.diagnostics_audio);
        let round = AgentConfig::parse_yaml(&cfg.to_yaml()).unwrap();
        assert_eq!(round, cfg);
    }

    #[test]
    fn diagnostics_custom_directory_round_trips_without_flags() {
        let yaml = format!(
            "{}\ndiagnostics:\n  directory: /var/dumps\n",
            AgentConfig::v0().to_yaml()
        );
        let cfg = AgentConfig::parse_yaml(&yaml).unwrap();
        assert_eq!(cfg.diagnostics_directory, PathBuf::from("/var/dumps"));
        assert!(!cfg.diagnostics_enabled(), "flags stay off");
        let round = AgentConfig::parse_yaml(&cfg.to_yaml()).unwrap();
        assert_eq!(round, cfg, "inert custom directory still round-trips");
        assert!(cfg.to_yaml().contains("directory: /var/dumps"));
    }

    #[test]
    fn diagnostics_unknown_field_is_named() {
        let yaml = format!(
            "{}\ndiagnostics:\n  verbose: true\n",
            AgentConfig::v0().to_yaml()
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(err.to_string().contains("diagnostics.verbose"), "{err}");
    }

    #[test]
    fn diagnostics_flags_must_be_booleans() {
        for key in ["timestamps", "audio"] {
            let yaml = format!(
                "{}\ndiagnostics:\n  {key}: sure\n",
                AgentConfig::v0().to_yaml()
            );
            let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
            assert!(
                err.to_string().contains(&format!("diagnostics.{key}")),
                "{key}: {err}"
            );
            assert!(err.to_string().contains("boolean"), "{key}: {err}");
        }
    }

    #[test]
    fn diagnostics_empty_directory_is_field_level() {
        let yaml = format!(
            "{}\ndiagnostics:\n  directory: \"\"\n",
            AgentConfig::v0().to_yaml()
        );
        let err = AgentConfig::parse_yaml(&yaml).unwrap_err();
        assert!(
            err.to_string()
                .contains("diagnostics.directory: must be a non-empty string"),
            "{err}"
        );
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
                    .replace("provider: local", "provider: deepgram"),
                "pipeline.stt.provider:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("model: whisper-small", "model: large"),
                "pipeline.stt.model:",
            ),
            (
                AgentConfig::v0().to_yaml().replace(
                    "  llm:\n    provider: local",
                    "  llm:\n    provider: ollama",
                ),
                "pipeline.llm.provider:",
            ),
            (
                AgentConfig::v0()
                    .to_yaml()
                    .replace("model: lfm2.5-2.6b", "model: huge"),
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
            .replace("  tts:\n    provider: local", "  tts:\n    provider: 1");
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
