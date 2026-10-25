//! Goose backend for `developer_harness.agent: goose`.
//!
//! Starts the pinned Goose CLI and runs each reply through it. Goose keeps
//! conversation state; Syllabix history is not sent. `generate()` uses channels
//! and checks [`Cancel`] between events. The worker blocks until a command or
//! a line from Goose arrives (no idle poll).

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::defaults::LlmProvider;
use crate::error::{Error, Result};
use crate::executor::{self, MAX_OUTPUT_BYTES};
use crate::llm::render_system_prompt;
use crate::mcp;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::openai::API_KEY_ENV;
use crate::policy::DeveloperPermissions;
use crate::providers::Llm;
use crate::skills::SkillDiscovery;
use crate::turn_debug::{write_acp_jsonl, AcpTraceLine};
use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, ToolCall, ToolTurnEvent, Transcript};

const GOOSE_UNSTABLE_PREFIX: &str = "_goose/unstable/";
const SYSTEM_PROMPT_KEY: &str = "syllabix";
/// `generate()` `evt_rx` wait slice only (Goose worker uses blocking `AcpIn::recv`).
const CANCEL_POLL: Duration = Duration::from_millis(100);
const IDLE_TIMEOUT: Duration = Duration::from_secs(20);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// Barge-in waits this long for Goose `TurnEnd(cancelled)` after `session/cancel`.
const CANCEL_DEADLINE: Duration = Duration::from_secs(2);

/// How long `generate()` waits with no tokens before failing.
/// Shells in flight do not count as idle (see `shell_inflight`).
fn idle_limit() -> Duration {
    #[cfg(test)]
    {
        if let Some(limit) = *TEST_IDLE_TIMEOUT.lock().expect("goose idle-timeout mutex") {
            return limit;
        }
    }
    IDLE_TIMEOUT
}

/// Config name for diagnostics and [`Llm::name`].
pub const PROVIDER_NAME: &str = "goose";

/// Live Goose backend behind [`crate::real::LiveLlm`].
pub struct GooseAgent {
    model: String,
    endpoint: String,
    cmd_tx: Sender<AcpIn>,
    evt_rx: mpsc::Receiver<AcpEvt>,
    tool_events: Arc<Mutex<Vec<ToolTurnEvent>>>,
    /// JSONL written by the Syllabix MCP child (`web_search` / `web_fetch`).
    mcp_tool_events: PathBuf,
    /// Present only when `diagnostics:` is enabled; never logs spawn-env secrets.
    acp_trace: Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    /// `diagnostics.directory` when diagnostics are on; Goose writes `turn-*/acp.jsonl`.
    diagnostics_dir: Option<PathBuf>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    workspace: PathBuf,
    permissions: DeveloperPermissions,
    cancel: Mutex<Cancel>,
    terminals: Mutex<HashMap<String, StoredTerminal>>,
    next_terminal: AtomicU64,
    /// Shell commands in flight; `generate()` skips idle timeout while > 0.
    shell_inflight: AtomicU64,
    /// Distinct ACP `toolCallId`s. Not [`Llm::take_tool_events`].
    acp_tool_calls: Mutex<HashSet<String>>,
}

#[derive(Clone)]
struct StoredTerminal {
    cancel: Cancel,
    /// `(output, ok, truncated)` once the sandboxed shell finishes.
    finished: Arc<Mutex<Option<(String, bool, bool)>>>,
}

enum AcpCmd {
    Prompt(String),
    Cancel,
    Shutdown,
}

/// Multiplexed wakeup for the Goose worker: a command, stdout line, or shell done.
enum AcpIn {
    Cmd(AcpCmd),
    Line(std::result::Result<String, String>),
    /// Async `terminal/create` shell finished; drain pending `wait_for_exit`.
    TerminalDone,
}

#[derive(Debug, PartialEq, Eq)]
enum AcpEvt {
    Ready,
    Token(String),
    TurnEnd { cancelled: bool },
    Died(String),
    Fail(String),
}

enum Pending {
    Init,
    NewSession,
    Unstable,
    Prompt,
}

/// Arguments to start Goose after the pinned CLI is on disk.
pub struct GooseStart {
    /// Unpacked `goose` executable.
    pub binary: PathBuf,
    /// Cache whose root holds `goose/` (`GOOSE_PATH_ROOT`).
    pub cache: PathBuf,
    pub config: AgentConfig,
    pub workspace: PathBuf,
    pub skill_list: String,
    /// Model label for diagnostics after `pipeline.llm` is written to Goose config.
    pub resolved_model: String,
    /// Provider API keys and hosts for the Goose child (never written to yaml).
    pub provider_env: Vec<(String, String)>,
}

/// Result of mapping `pipeline.llm` onto Goose's config and child environment.
pub(crate) struct GoosePipelinePrepared {
    pub display_model: String,
    pub child_env: Vec<(String, String)>,
}

/// Path Goose uses for config/data/state (`GOOSE_PATH_ROOT`).
pub(crate) fn goose_path_root(cache: &ModelCache) -> PathBuf {
    let mut root = cache.models_dir();
    root.pop();
    root.pop();
    root.push("goose");
    root
}

/// Goose provider docs shown when no provider/model is configured.
pub(crate) const GOOSE_PROVIDER_DOCS: &str =
    "https://goose-docs.ai/docs/getting-started/providers/";

/// Goose's own config file under `{GOOSE_PATH_ROOT}/config`.
pub(crate) fn goose_config_path(path_root: &Path) -> PathBuf {
    path_root.join("config").join("config.yaml")
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn config_string(config: &serde_yaml::Value, key: &str) -> Option<String> {
    non_empty(config.get(key).and_then(serde_yaml::Value::as_str))
}

fn provider_model(config: &serde_yaml::Value, provider: &str) -> Option<String> {
    non_empty(
        config
            .get("providers")
            .and_then(|providers| providers.get(provider))
            .and_then(|entry| entry.get("model"))
            .and_then(serde_yaml::Value::as_str),
    )
}

/// Goose's system-wide config, merged before the scoped config.
fn system_goose_config_path() -> PathBuf {
    #[cfg(windows)]
    {
        std::env::var_os("PROGRAMDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
            .join("goose")
            .join("config.yaml")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from("/etc/goose/config.yaml")
    }
}

/// Config files Goose merges, lowest precedence first: system config, any
/// `GOOSE_ADDITIONAL_CONFIG_FILES` (only when the child inherits the session
/// environment), then the Syllabix-scoped user config.
fn goose_config_candidates(path_root: &Path, use_env: bool) -> Vec<PathBuf> {
    let mut paths = vec![system_goose_config_path()];
    if use_env {
        if let Some(extra) = std::env::var_os("GOOSE_ADDITIONAL_CONFIG_FILES") {
            paths.extend(std::env::split_paths(&extra).filter(|path| !path.as_os_str().is_empty()));
        }
    }
    paths.push(goose_config_path(path_root));
    paths
}

fn read_goose_config(path: &Path) -> Result<Option<serde_yaml::Value>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(Error::Config {
                field: "developer_harness.agent".into(),
                message: format!("could not read Goose config {}: {err}", path.display()),
            })
        }
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }
    serde_yaml::from_str(&raw)
        .map(Some)
        .map_err(|err| Error::Config {
            field: "developer_harness.agent".into(),
            message: format!("invalid Goose config {}: {err}", path.display()),
        })
}

/// Deep-merge mappings so a later file can override one provider key without
/// dropping the rest of the provider entry.
fn mapping_mut(value: &mut serde_yaml::Value) -> &mut serde_yaml::Mapping {
    match value {
        serde_yaml::Value::Mapping(map) => map,
        _ => panic!("expected yaml mapping"),
    }
}

fn set_provider_model(config: &mut serde_yaml::Value, provider: &str, model: &str) {
    let root = mapping_mut(config);
    root.insert(
        serde_yaml::Value::String("active_provider".into()),
        serde_yaml::Value::String(provider.into()),
    );
    let providers = root
        .entry(serde_yaml::Value::String("providers".into()))
        .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    let providers = mapping_mut(providers);
    let entry = providers
        .entry(serde_yaml::Value::String(provider.into()))
        .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    let entry = mapping_mut(entry);
    entry.insert(
        serde_yaml::Value::String("enabled".into()),
        serde_yaml::Value::Bool(true),
    );
    entry.insert(
        serde_yaml::Value::String("model".into()),
        serde_yaml::Value::String(model.into()),
    );
    entry.insert(
        serde_yaml::Value::String("configured".into()),
        serde_yaml::Value::Bool(true),
    );
}

fn openai_host_from_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if let Some(host) = trimmed.strip_suffix("/v1") {
        host.trim_end_matches('/').to_string()
    } else {
        trimmed.to_string()
    }
}

fn ollama_host_from_base_url(base_url: &str) -> String {
    openai_host_from_base_url(base_url)
}

fn goose_provider_names(pipeline: &LlmProvider) -> (&str, &str) {
    match pipeline {
        LlmProvider::Local => ("local", "local"),
        LlmProvider::OpenAiCompatible => ("openai", "openai-compatible"),
        LlmProvider::External(name) => (name.as_str(), name.as_str()),
    }
}

fn single_key_env_for_goose(goose_provider: &str) -> Option<&'static str> {
    match goose_provider {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "xai" => Some("XAI_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        "google" => Some("GOOGLE_API_KEY"),
        "local" | "ollama" => None,
        _ => None,
    }
}

fn passthrough_env_vars(goose_provider: &str) -> &'static [&'static str] {
    match goose_provider {
        "bedrock" => &[
            "AWS_PROFILE",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_DEFAULT_REGION",
        ],
        "azure_openai" => &[
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_AD_TOKEN",
        ],
        "azure_foundry" => &[
            "AZURE_FOUNDRY_ENDPOINT",
            "AZURE_FOUNDRY_API_KEY",
            "AZURE_FOUNDRY_AD_TOKEN",
            "AZURE_FOUNDRY_API_VERSION",
        ],
        "vertex_ai" | "gcp_vertex_ai" | "vertex" => &["GCP_PROJECT_ID", "GCP_LOCATION"],
        _ => &[],
    }
}

fn is_loopback_base_url(base_url: &str) -> bool {
    let Ok(url) = url::Url::parse(base_url.trim()) else {
        return false;
    };
    matches!(
        url.host_str(),
        Some("127.0.0.1") | Some("localhost") | Some("::1")
    )
}

fn resolve_local_gguf_for_goose(
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
    model: &str,
) -> Result<String> {
    if crate::config::is_gguf_path(model) {
        let path = PathBuf::from(model);
        if !path.is_file() {
            return Err(Error::Config {
                field: "pipeline.llm.model".into(),
                message: format!("GGUF file does not exist: {}", path.display()),
            });
        }
        return Ok(path.to_string_lossy().into_owned());
    }
    let asset = cache
        .manifest()
        .asset(model)
        .ok_or_else(|| Error::ModelCache {
            message: format!("manifest does not contain the {model} asset"),
        })?;
    let path = cache.resolve(asset, fetcher, progress, cancel)?;
    Ok(path.to_string_lossy().into_owned())
}

/// Write `pipeline.llm` into Goose's scoped `config.yaml` and build child env.
pub(crate) fn prepare_goose_from_pipeline(
    path_root: &Path,
    config: &AgentConfig,
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
    llm_api_key: Option<&Zeroizing<String>>,
) -> Result<GoosePipelinePrepared> {
    let (goose_provider, _) = goose_provider_names(&config.llm);
    let goose_model = match &config.llm {
        LlmProvider::Local => {
            resolve_local_gguf_for_goose(cache, fetcher, progress, cancel, &config.llm_model)?
        }
        _ => config.llm_model.clone(),
    };
    let config_path = goose_config_path(path_root);
    std::fs::create_dir_all(config_path.parent().ok_or_else(|| Error::Config {
        field: "developer_harness.agent".into(),
        message: "invalid Goose config directory".into(),
    })?)?;
    let mut merged = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
    if let Some(layer) = read_goose_config(&config_path)? {
        merge_config(&mut merged, layer);
    }
    set_provider_model(&mut merged, goose_provider, &goose_model);
    let serialized = serde_yaml::to_string(&merged).map_err(|err| Error::Config {
        field: "developer_harness.agent".into(),
        message: format!("could not serialize Goose config: {err}"),
    })?;
    std::fs::write(&config_path, serialized).map_err(|err| Error::Config {
        field: "developer_harness.agent".into(),
        message: format!(
            "could not write Goose config {}: {err}",
            config_path.display()
        ),
    })?;
    let mut child_env = Vec::new();
    if let Some(base_url) = config.llm_base_url.as_deref() {
        if goose_provider == "openai" {
            child_env.push(("OPENAI_HOST".into(), openai_host_from_base_url(base_url)));
        } else if goose_provider == "ollama" {
            child_env.push(("OLLAMA_HOST".into(), ollama_host_from_base_url(base_url)));
        }
    }

    let syllabix_key = llm_api_key.cloned().or_else(|| {
        std::env::var(API_KEY_ENV)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(Zeroizing::new)
    });

    if let Some(env_name) = single_key_env_for_goose(goose_provider) {
        let needs_key = !config
            .llm_base_url
            .as_deref()
            .is_some_and(is_loopback_base_url);
        let key = if needs_key {
            syllabix_key.ok_or_else(|| Error::Config {
                field: API_KEY_ENV.into(),
                message: format!(
                    "is required for pipeline.llm.provider {} (mapped to Goose provider {goose_provider})",
                    config.llm.as_str()
                ),
            })?
        } else {
            syllabix_key.unwrap_or_else(|| Zeroizing::new("local".into()))
        };
        child_env.push((env_name.into(), key.to_string()));
    } else if goose_provider != "local" && goose_provider != "ollama" {
        let vars = passthrough_env_vars(goose_provider);
        if vars.is_empty() {
            return Err(Error::Config {
                field: "pipeline.llm.provider".into(),
                message: format!(
                    "Goose provider {goose_provider} needs credentials in your environment; \
                     Syllabix only maps SYLLABIX_LLM_API_KEY for openai, anthropic, xai, openrouter, and google"
                ),
            });
        }
        let mut copied = false;
        for name in vars {
            if let Ok(value) = std::env::var(name) {
                if !value.trim().is_empty() {
                    child_env.push((name.to_string(), value));
                    copied = true;
                }
            }
        }
        if !copied {
            return Err(Error::Config {
                field: "pipeline.llm.provider".into(),
                message: format!(
                    "could not find credentials for Goose provider {goose_provider}; set one of: {}",
                    vars.join(", ")
                ),
            });
        }
    }

    let display_model = resolve_goose_model(path_root, false)?;
    Ok(GoosePipelinePrepared {
        display_model,
        child_env,
    })
}

fn merge_config(base: &mut serde_yaml::Value, over: serde_yaml::Value) {
    match (base, over) {
        (serde_yaml::Value::Mapping(base), serde_yaml::Value::Mapping(over)) => {
            for (key, value) in over {
                match base.get_mut(&key) {
                    Some(existing) => merge_config(existing, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, over) => *base = over,
    }
}

/// Read Goose's active provider/model the way Goose does: environment first
/// (only when the child inherits it), then the merged `active_provider`/
/// `providers`, then the legacy flat keys. Syllabix sets none of these.
pub(crate) fn resolve_goose_model(path_root: &Path, use_env: bool) -> Result<String> {
    resolve_goose_model_in(&goose_config_candidates(path_root, use_env), use_env)
}

fn resolve_goose_model_in(paths: &[PathBuf], use_env: bool) -> Result<String> {
    let mut config = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
    for path in paths {
        if let Some(layer) = read_goose_config(path)? {
            merge_config(&mut config, layer);
        }
    }
    let env = |name: &str| {
        if use_env {
            std::env::var(name).ok()
        } else {
            None
        }
        .and_then(|value| non_empty(Some(&value)))
    };
    let provider = env("GOOSE_PROVIDER")
        .or_else(|| env("ACTIVE_PROVIDER"))
        .or_else(|| config_string(&config, "active_provider"))
        .or_else(|| config_string(&config, "GOOSE_PROVIDER"));
    let model = env("GOOSE_MODEL")
        .or_else(|| {
            provider
                .as_deref()
                .and_then(|provider| provider_model(&config, provider))
        })
        .or_else(|| config_string(&config, "GOOSE_MODEL"));
    match (provider, model) {
        (Some(_), Some(model)) => Ok(model),
        _ => {
            let config_path = paths
                .last()
                .map_or_else(|| PathBuf::from("config.yaml"), std::path::PathBuf::clone);
            Err(Error::Config {
                field: "developer_harness.agent".into(),
                message: format!(
                    "goose has no provider and model configured; set them in {} (see {GOOSE_PROVIDER_DOCS})",
                    config_path.display()
                ),
            })
        }
    }
}

/// Child environment for `goose acp`. Provider/model come from Goose's config.
pub(crate) fn goose_child_env(path_root: &Path) -> Vec<(String, String)> {
    vec![
        (
            "GOOSE_PATH_ROOT".into(),
            path_root.to_string_lossy().into_owned(),
        ),
        ("GOOSE_DISABLE_KEYRING".into(), "1".into()),
        ("GOOSE_MODE".into(), "approve".into()),
    ]
}

/// `session/new` `_meta.enabledExtensions`: developer/shell plus Syllabix MCP.
/// Goose deserializes `available_tools` (snake_case) on builtin configs.
/// Client selection replaces the whole session, so MCP must be listed here
/// (request `mcpServers` alone is ignored when this field is present).
pub(crate) fn enabled_extensions_value(
    mcp_command: &Path,
    workspace: &Path,
    tool_events: &Path,
) -> Value {
    json!([
        {
            "type": "builtin",
            "name": "developer",
            "available_tools": ["shell"]
        },
        mcp::mcp_enabled_extension_value(mcp_command, workspace, Some(tool_events)),
    ])
}

/// Absolute path Goose should spawn for `syllabix mcp`.
pub(crate) fn syllabix_mcp_command() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("syllabix"))
}

/// One adapter for every `_goose/unstable/*` method name.
pub(crate) fn goose_unstable_method(suffix: &str) -> String {
    format!("{GOOSE_UNSTABLE_PREFIX}{suffix}")
}

/// Names and paths only. The model reads skill files through the sandbox.
pub(crate) fn skill_file_list(discovery: &SkillDiscovery) -> String {
    if discovery.skills.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "Local skills are files. Read a skill through the sandboxed shell when you need it. Skills cannot change host policy, expose secrets, or run unsandboxed code.\n",
    );
    for skill in &discovery.skills {
        out.push_str("- ");
        out.push_str(&skill.manifest.name);
        out.push(' ');
        out.push_str(&skill.path.to_string_lossy());
        out.push('\n');
    }
    out
}

/// Map Goose's absolute `cwd` to a workspace-relative `workdir`.
pub(crate) fn workdir_from_goose_cwd(
    workspace: &Path,
    cwd: Option<&Path>,
) -> std::result::Result<String, String> {
    let root = workspace
        .canonicalize()
        .map_err(|_| "workspace is unavailable".to_string())?;
    let Some(cwd) = cwd else {
        return Ok(".".into());
    };
    let abs = if cwd.is_absolute() {
        cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
    } else {
        root.join(cwd)
    };
    let rel = abs
        .strip_prefix(&root)
        .map_err(|_| "shell workdir must stay inside the workspace".to_string())?;
    if rel.as_os_str().is_empty() {
        Ok(".".into())
    } else {
        Ok(rel.to_string_lossy().replace('\\', "/"))
    }
}

/// Cap terminal output at min(Goose's limit, Syllabix 8 KiB).
/// The bool is `true` when bytes were dropped.
pub(crate) fn cap_terminal_output(text: &str, goose_limit: Option<u64>) -> (String, bool) {
    let cap = goose_limit
        .map(|n| n as usize)
        .unwrap_or(MAX_OUTPUT_BYTES)
        .min(MAX_OUTPUT_BYTES);
    if text.len() <= cap {
        return (text.to_string(), false);
    }
    let mut end = cap;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

/// `permission.yaml` under `{GOOSE_PATH_ROOT}/config`.
pub(crate) fn write_permission_yaml(path_root: &Path) -> Result<PathBuf> {
    let config_dir = path_root.join("config");
    std::fs::create_dir_all(&config_dir)?;
    let path = config_dir.join("permission.yaml");
    let extras: Vec<String> = mcp::permission_tool_names().into();
    std::fs::write(&path, permission_yaml_body(&extras))?;
    Ok(path)
}

pub(crate) fn permission_yaml_body(extra_always_allow: &[String]) -> String {
    let mut always = vec!["shell".to_string()];
    always.extend(extra_always_allow.iter().cloned());
    always.sort();
    always.dedup();
    let mut never = vec![
        "tree".to_string(),
        "write".to_string(),
        "edit".to_string(),
        "read_image".to_string(),
    ];
    never.sort();
    let body = json!({
        "user": {
            "always_allow": always,
            "ask_before": Vec::<String>::new(),
            "never_allow": never,
        }
    });
    serde_yaml::to_string(&body).unwrap_or_else(|_| {
        "user:\n  always_allow:\n    - shell\n  ask_before: []\n  never_allow:\n    - edit\n    - read_image\n    - tree\n    - write\n".into()
    })
}

fn provider_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

/// Resolve the pinned binary, or a test override that skips the network.
pub(crate) fn resolve_goose_exe(
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<PathBuf> {
    #[cfg(test)]
    {
        if let Some(path) = test_goose_bin() {
            return Ok(path);
        }
    }
    crate::models::resolve_goose_binary(cache, fetcher, progress, cancel)
}

#[cfg(test)]
static TEST_GOOSE_BIN: Mutex<Option<PathBuf>> = Mutex::new(None);
#[cfg(test)]
static TEST_GOOSE_BIN_SERIAL: Mutex<()> = Mutex::new(());
#[cfg(test)]
static TEST_IDLE_TIMEOUT: Mutex<Option<Duration>> = Mutex::new(None);
#[cfg(test)]
static TEST_IDLE_SERIAL: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn test_goose_bin() -> Option<PathBuf> {
    TEST_GOOSE_BIN.lock().expect("goose test-bin mutex").clone()
}

/// Serializes tests that inject a fake Goose binary into [`build_llm`].
#[cfg(test)]
pub(crate) struct GooseBinGuard {
    _serial: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl GooseBinGuard {
    pub(crate) fn set(path: PathBuf) -> Self {
        let serial = TEST_GOOSE_BIN_SERIAL
            .lock()
            .expect("goose test-bin serial mutex");
        *TEST_GOOSE_BIN.lock().expect("goose test-bin mutex") = Some(path);
        Self { _serial: serial }
    }
}

#[cfg(test)]
impl Drop for GooseBinGuard {
    fn drop(&mut self) {
        *TEST_GOOSE_BIN.lock().expect("goose test-bin mutex") = None;
    }
}

/// Overrides [`idle_limit`] for one test (serial with other idle overrides).
#[cfg(test)]
pub(crate) struct GooseIdleGuard {
    _serial: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl GooseIdleGuard {
    pub(crate) fn set(limit: Duration) -> Self {
        let serial = TEST_IDLE_SERIAL
            .lock()
            .expect("goose idle-timeout serial mutex");
        *TEST_IDLE_TIMEOUT.lock().expect("goose idle-timeout mutex") = Some(limit);
        Self { _serial: serial }
    }
}

#[cfg(test)]
impl Drop for GooseIdleGuard {
    fn drop(&mut self) {
        *TEST_IDLE_TIMEOUT.lock().expect("goose idle-timeout mutex") = None;
    }
}

impl GooseAgent {
    /// Start Goose, open a session, and set the system prompt.
    pub fn start(start: GooseStart) -> Result<Self> {
        let GooseStart {
            binary,
            cache,
            config,
            workspace,
            skill_list,
            resolved_model: model,
            provider_env,
        } = start;
        let path_root = {
            let mut root = cache;
            if !root.is_absolute() {
                root = std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(root);
            }
            std::fs::create_dir_all(&root)?;
            root.canonicalize().unwrap_or(root)
        };
        write_permission_yaml(&path_root)?;
        let workspace = workspace.canonicalize().unwrap_or(workspace);
        let mcp_tool_events = path_root.join("mcp-tool-events.jsonl");
        let _ = std::fs::write(&mcp_tool_events, "");
        let mcp_command = syllabix_mcp_command();
        let env = goose_child_env(&path_root);
        let mut command = Command::new(&binary);
        let stderr_file = std::fs::File::create(path_root.join("goose.stderr"))
            .map_err(|err| provider_err(format!("goose stderr file: {err}")))?;
        command
            .arg("acp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr_file));
        if !config.inherit_current_session {
            command.env_clear();
        }
        for (key, value) in env.iter().chain(provider_env.iter()) {
            command.env(key, value);
        }
        let mut child = command
            .spawn()
            .map_err(|err| provider_err(format!("could not start Goose: {err}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| provider_err("Goose did not accept input"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| provider_err("Goose produced no output stream"))?;

        let (in_tx, in_rx) = mpsc::channel::<AcpIn>();
        let reader_tx = in_tx.clone();
        thread::Builder::new()
            .name("syllabix-acp-read".into())
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    match line {
                        Ok(line) if line.trim().is_empty() => {}
                        Ok(line) => {
                            if reader_tx.send(AcpIn::Line(Ok(line))).is_err() {
                                return;
                            }
                        }
                        Err(err) => {
                            let _ = reader_tx.send(AcpIn::Line(Err(err.to_string())));
                            return;
                        }
                    }
                }
                let _ = reader_tx.send(AcpIn::Line(Err("Goose closed its output".into())));
            })
            .map_err(|err| provider_err(format!("Goose output reader: {err}")))?;

        let (evt_tx, evt_rx) = mpsc::channel();
        let tool_events = Arc::new(Mutex::new(Vec::new()));
        let diagnostics_dir = config
            .diagnostics_enabled()
            .then(|| config.diagnostics_directory.clone());
        let acp_trace = diagnostics_dir
            .is_some()
            .then(|| Arc::new(Mutex::new(Vec::<AcpTraceLine>::new())));
        let shared = Arc::new(Shared {
            workspace: workspace.clone(),
            permissions: config.llm_developer_permissions.clone(),
            cancel: Mutex::new(Cancel::new()),
            terminals: Mutex::new(HashMap::new()),
            next_terminal: AtomicU64::new(1),
            shell_inflight: AtomicU64::new(0),
            acp_tool_calls: Mutex::new(HashSet::new()),
        });
        let worker_shared = Arc::clone(&shared);
        let worker_events = Arc::clone(&tool_events);
        let worker_trace = acp_trace.clone();
        let system_prompt = config.system_prompt.clone();
        let skill_list_worker = skill_list.clone();
        let language = config.language.clone();
        let wake_tx = in_tx.clone();
        let worker_mcp_command = mcp_command.clone();
        let worker_mcp_events = mcp_tool_events.clone();
        let worker = thread::Builder::new()
            .name("syllabix-acp".into())
            .spawn(move || {
                acp_thread(AcpThread {
                    child,
                    stdin,
                    in_rx,
                    wake_tx,
                    evt_tx,
                    tool_events: worker_events,
                    acp_trace: worker_trace,
                    shared: worker_shared,
                    workspace,
                    system_prompt,
                    skill_list: skill_list_worker,
                    language,
                    mcp_command: worker_mcp_command,
                    mcp_tool_events: worker_mcp_events,
                    pending_waits: Vec::new(),
                    prompt_cancelled: false,
                });
            })
            .map_err(|err| provider_err(format!("could not start Goose worker: {err}")))?;

        let handshake = match evt_rx.recv_timeout(HANDSHAKE_TIMEOUT) {
            Ok(AcpEvt::Ready) => Ok(()),
            Ok(AcpEvt::Fail(message) | AcpEvt::Died(message)) => Err(provider_err(message)),
            Ok(_) => Err(provider_err("Goose sent an unexpected startup reply")),
            Err(RecvTimeoutError::Timeout) => Err(provider_err("Goose did not start in time")),
            Err(RecvTimeoutError::Disconnected) => {
                Err(provider_err("Goose stopped during startup"))
            }
        };
        if let Err(err) = handshake {
            let _ = in_tx.send(AcpIn::Cmd(AcpCmd::Shutdown));
            let _ = worker.join();
            return Err(err);
        }

        Ok(Self {
            model,
            endpoint: String::new(),
            cmd_tx: in_tx,
            evt_rx,
            tool_events,
            mcp_tool_events,
            acp_trace,
            diagnostics_dir,
            shared,
            worker: Some(worker),
        })
    }

    /// Drain buffered Goose transcript lines (diagnostics only). Empty when off.
    pub fn take_acp_trace(&mut self) -> Vec<AcpTraceLine> {
        let Some(trace) = &self.acp_trace else {
            return Vec::new();
        };
        std::mem::take(&mut *trace.lock().expect("goose acp-trace mutex"))
    }

    fn flush_acp_trace(&mut self, turn: crate::types::TurnId) {
        let Some(root) = self.diagnostics_dir.clone() else {
            return;
        };
        let lines = self.take_acp_trace();
        if lines.is_empty() {
            return;
        }
        let dir = root.join(format!("turn-{:03}", turn.0));
        if let Err(err) = write_acp_jsonl(&dir, &lines) {
            tracing::warn!(error = %err, "failed to write goose acp.jsonl");
        }
    }

    /// Distinct ACP `toolCallId`s this session. Not [`Llm::take_tool_events`].
    pub fn acp_tool_call_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .shared
            .acp_tool_calls
            .lock()
            .expect("goose acp-tool-calls mutex")
            .iter()
            .cloned()
            .collect();
        ids.sort();
        ids
    }

    /// `terminal/create` requests handled this session.
    pub fn terminal_create_count(&self) -> u64 {
        self.shared
            .next_terminal
            .load(Ordering::SeqCst)
            .saturating_sub(1)
    }
}

struct AcpThread {
    child: Child,
    stdin: ChildStdin,
    in_rx: mpsc::Receiver<AcpIn>,
    /// Clone of the multiplex sender; shell threads wake the worker via `TerminalDone`.
    wake_tx: Sender<AcpIn>,
    evt_tx: Sender<AcpEvt>,
    tool_events: Arc<Mutex<Vec<ToolTurnEvent>>>,
    acp_trace: Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    shared: Arc<Shared>,
    workspace: PathBuf,
    system_prompt: String,
    skill_list: String,
    language: String,
    mcp_command: PathBuf,
    mcp_tool_events: PathBuf,
    /// `wait_for_exit` RPC ids waiting on async shells.
    pending_waits: Vec<(Value, String)>,
    /// Set when barge-in sent `session/cancel` for the in-flight prompt.
    prompt_cancelled: bool,
}

fn acp_thread(mut ctx: AcpThread) {
    let mut next_id = 1u64;
    let mut pending: HashMap<u64, Pending> = HashMap::new();
    let mut session_id = String::new();
    if let Err(err) = handshake(&mut ctx, &mut next_id, &mut pending, &mut session_id) {
        let _ = ctx.evt_tx.send(AcpEvt::Fail(err));
        let _ = ctx.child.kill();
        let _ = ctx.child.wait();
        return;
    }
    let _ = ctx.evt_tx.send(AcpEvt::Ready);
    loop {
        match ctx.in_rx.recv() {
            Ok(AcpIn::Cmd(AcpCmd::Shutdown)) => break,
            Ok(AcpIn::Cmd(AcpCmd::Prompt(text))) => {
                ctx.prompt_cancelled = false;
                if let Err(err) =
                    send_prompt(&mut ctx, &mut next_id, &mut pending, &session_id, &text)
                {
                    let _ = ctx.evt_tx.send(AcpEvt::Fail(err));
                }
            }
            Ok(AcpIn::Cmd(AcpCmd::Cancel)) => {
                ctx.prompt_cancelled = true;
                kill_running_terminals(&ctx.shared);
                if !session_id.is_empty() {
                    let _ = write_rpc(
                        &mut ctx.stdin,
                        &ctx.acp_trace,
                        0,
                        "session/cancel",
                        &json!({ "sessionId": session_id }),
                    );
                }
            }
            Ok(AcpIn::TerminalDone) => {
                complete_pending_waits(&mut ctx);
            }
            Ok(AcpIn::Line(Ok(line))) => {
                if let Err(err) = handle_line(&mut ctx, &mut pending, &mut session_id, &line) {
                    let _ = ctx.evt_tx.send(AcpEvt::Fail(err));
                    break;
                }
            }
            Ok(AcpIn::Line(Err(err))) => {
                let _ = ctx.evt_tx.send(AcpEvt::Died(err));
                break;
            }
            Err(_) => break,
        }
    }
    kill_running_terminals(&ctx.shared);
    let _ = ctx.child.kill();
    let _ = ctx.child.wait();
}

fn handshake(
    ctx: &mut AcpThread,
    next_id: &mut u64,
    pending: &mut HashMap<u64, Pending>,
    session_id: &mut String,
) -> std::result::Result<(), String> {
    write_rpc(
        &mut ctx.stdin,
        &ctx.acp_trace,
        *next_id,
        "initialize",
        &json!({
            "protocolVersion": 1,
            "clientCapabilities": { "terminal": true },
            "clientInfo": { "name": "syllabix" },
        }),
    )?;
    pending.insert(*next_id, Pending::Init);
    *next_id += 1;
    wait_pending(ctx, pending, session_id, Pending::Init)?;

    let _ = write_rpc(&mut ctx.stdin, &ctx.acp_trace, 0, "initialized", &json!({}));

    let mcp_servers = vec![mcp::mcp_server_value(
        &ctx.mcp_command,
        &ctx.workspace,
        Some(&ctx.mcp_tool_events),
    )];
    write_rpc(
        &mut ctx.stdin,
        &ctx.acp_trace,
        *next_id,
        "session/new",
        &json!({
            "cwd": ctx.workspace,
            "mcpServers": mcp_servers,
            "_meta": {
                "enabledExtensions": enabled_extensions_value(
                    &ctx.mcp_command,
                    &ctx.workspace,
                    &ctx.mcp_tool_events,
                ),
            },
        }),
    )?;
    pending.insert(*next_id, Pending::NewSession);
    *next_id += 1;
    wait_pending(ctx, pending, session_id, Pending::NewSession)?;

    let language = ctx.language.clone();
    set_system_prompt(ctx, next_id, pending, session_id, &language)?;
    Ok(())
}

fn set_system_prompt(
    ctx: &mut AcpThread,
    next_id: &mut u64,
    pending: &mut HashMap<u64, Pending>,
    session_id: &str,
    language: &str,
) -> std::result::Result<(), String> {
    let mut text = render_system_prompt(&ctx.system_prompt, language);
    if !ctx.skill_list.is_empty() {
        text.push('\n');
        text.push_str(&ctx.skill_list);
    }
    let params = json!({
        "sessionId": session_id,
        "mode": "set",
        "key": SYSTEM_PROMPT_KEY,
        "text": text,
    });
    write_rpc(
        &mut ctx.stdin,
        &ctx.acp_trace,
        *next_id,
        &goose_unstable_method("session/system-prompt/set"),
        &params,
    )?;
    pending.insert(*next_id, Pending::Unstable);
    *next_id += 1;
    wait_pending(ctx, pending, &mut session_id.to_string(), Pending::Unstable)
}

fn send_prompt(
    ctx: &mut AcpThread,
    next_id: &mut u64,
    pending: &mut HashMap<u64, Pending>,
    session_id: &str,
    text: &str,
) -> std::result::Result<(), String> {
    write_rpc(
        &mut ctx.stdin,
        &ctx.acp_trace,
        *next_id,
        "session/prompt",
        &json!({
            "sessionId": session_id,
            "prompt": [{ "type": "text", "text": text }],
        }),
    )?;
    pending.insert(*next_id, Pending::Prompt);
    *next_id += 1;
    Ok(())
}

fn wait_pending(
    ctx: &mut AcpThread,
    pending: &mut HashMap<u64, Pending>,
    session_id: &mut String,
    want: Pending,
) -> std::result::Result<(), String> {
    let deadline = Instant::now() + HANDSHAKE_TIMEOUT;
    loop {
        if Instant::now() >= deadline {
            return Err("Goose did not start in time".into());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        match ctx.in_rx.recv_timeout(remaining) {
            Ok(AcpIn::Line(Ok(line))) => {
                handle_line(ctx, pending, session_id, &line)?;
                if !pending
                    .values()
                    .any(|pending| matches_pending(pending, &want))
                {
                    return Ok(());
                }
            }
            Ok(AcpIn::Line(Err(err))) => return Err(err),
            Ok(AcpIn::Cmd(AcpCmd::Shutdown)) => return Err("Goose shut down during startup".into()),
            Ok(AcpIn::Cmd(_) | AcpIn::TerminalDone) => {}
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Err("Goose worker channel closed".into()),
        }
    }
}

fn matches_pending(pending: &Pending, want: &Pending) -> bool {
    std::mem::discriminant(pending) == std::mem::discriminant(want)
}

fn handle_line(
    ctx: &mut AcpThread,
    pending: &mut HashMap<u64, Pending>,
    session_id: &mut String,
    line: &str,
) -> std::result::Result<(), String> {
    record_acp_trace(&ctx.acp_trace, "in", line);
    let value: Value =
        serde_json::from_str(line).map_err(|err| format!("bad message from Goose: {err}"))?;
    if let Some(id) = value.get("id") {
        if value.get("method").is_some() {
            handle_request(ctx, &value)?;
            return Ok(());
        }
        let id = id
            .as_u64()
            .ok_or_else(|| "Goose message id must be a number".to_string())?;
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Goose error")
                .to_string();
            match pending.remove(&id) {
                Some(Pending::Prompt) if ctx.prompt_cancelled => {
                    let _ = ctx.evt_tx.send(AcpEvt::TurnEnd { cancelled: true });
                    return Ok(());
                }
                Some(Pending::Prompt) => {
                    let _ = ctx.evt_tx.send(AcpEvt::Fail(message));
                    return Ok(());
                }
                Some(_) | None => return Err(message),
            }
        }
        let result = value.get("result").cloned().unwrap_or(Value::Null);
        match pending.remove(&id) {
            Some(Pending::NewSession) => {
                if let Some(sid) = result
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .or_else(|| result.get("session_id").and_then(Value::as_str))
                {
                    *session_id = sid.to_string();
                } else {
                    return Err("session/new did not return sessionId".into());
                }
            }
            Some(Pending::Prompt) => {
                let _ = ctx.evt_tx.send(AcpEvt::TurnEnd {
                    cancelled: prompt_was_cancelled(&result),
                });
            }
            Some(Pending::Init | Pending::Unstable) | None => {}
        }
        return Ok(());
    }
    if let Some(method) = value.get("method").and_then(Value::as_str) {
        if method == "session/update" {
            handle_session_update(
                &ctx.evt_tx,
                &ctx.shared.acp_tool_calls,
                value.get("params").unwrap_or(&Value::Null),
            );
        }
    }
    Ok(())
}

fn handle_request(ctx: &mut AcpThread, value: &Value) -> std::result::Result<(), String> {
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = value.get("id").cloned().unwrap_or(Value::Null);
    let params = value.get("params").cloned().unwrap_or(Value::Null);
    match method {
        "terminal/create" => match terminal_create(ctx, &params) {
            Ok(result) => write_result(&mut ctx.stdin, &ctx.acp_trace, id, result)?,
            Err(err) => write_rpc_error(&mut ctx.stdin, &ctx.acp_trace, id, -32602, err)?,
        },
        "terminal/wait_for_exit" => {
            let tid = terminal_id(&params);
            if let Some((_, ok, _)) = terminal_finished(&ctx.shared, &tid) {
                write_wait_result(&mut ctx.stdin, &ctx.acp_trace, id, ok)?;
            } else {
                ctx.pending_waits.push((id, tid));
            }
        }
        "terminal/output" => {
            let tid = terminal_id(&params);
            let (output, ok, truncated, done) = terminal_snapshot(&ctx.shared, &tid);
            let mut body = json!({
                "output": output,
                "truncated": truncated,
            });
            if done {
                body["exitStatus"] = json!({ "exitCode": i64::from(!ok) });
            }
            write_result(&mut ctx.stdin, &ctx.acp_trace, id, body)?;
        }
        "terminal/release" => {
            let tid = terminal_id(&params);
            if let Some(stored) = ctx
                .shared
                .terminals
                .lock()
                .expect("goose terminals mutex")
                .remove(&tid)
            {
                stored.cancel.shutdown();
            }
            write_result(&mut ctx.stdin, &ctx.acp_trace, id, json!({}))?;
        }
        "terminal/kill" => {
            let tid = terminal_id(&params);
            kill_terminal(&ctx.shared, &tid);
            write_result(&mut ctx.stdin, &ctx.acp_trace, id, json!({}))?;
        }
        other => {
            write_rpc_error(
                &mut ctx.stdin,
                &ctx.acp_trace,
                id,
                -32601,
                format!("method not supported: {other}"),
            )?;
        }
    }
    Ok(())
}

fn terminal_id(params: &Value) -> String {
    params
        .get("terminalId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn prompt_was_cancelled(result: &Value) -> bool {
    result
        .get("stopReason")
        .and_then(Value::as_str)
        .or_else(|| result.get("stop_reason").and_then(Value::as_str))
        == Some("cancelled")
}

fn kill_running_terminals(shared: &Shared) {
    let terminals = shared.terminals.lock().expect("goose terminals mutex");
    for stored in terminals.values() {
        stored.cancel.shutdown();
    }
}

fn kill_terminal(shared: &Shared, tid: &str) {
    if let Some(stored) = shared
        .terminals
        .lock()
        .expect("goose terminals mutex")
        .get(tid)
    {
        stored.cancel.shutdown();
    }
}

fn terminal_finished(shared: &Shared, tid: &str) -> Option<(String, bool, bool)> {
    let terminals = shared.terminals.lock().expect("goose terminals mutex");
    let stored = terminals.get(tid)?;
    let finished = stored
        .finished
        .lock()
        .expect("goose terminal result mutex")
        .clone();
    finished
}

fn terminal_snapshot(shared: &Shared, tid: &str) -> (String, bool, bool, bool) {
    match terminal_finished(shared, tid) {
        Some((output, ok, truncated)) => (output, ok, truncated, true),
        None => (String::new(), false, false, false),
    }
}

fn write_wait_result(
    stdin: &mut ChildStdin,
    trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    id: Value,
    ok: bool,
) -> std::result::Result<(), String> {
    write_result(
        stdin,
        trace,
        id,
        json!({ "exitCode": i64::from(!ok), "signal": Value::Null }),
    )
}

fn complete_pending_waits(ctx: &mut AcpThread) {
    let pending = std::mem::take(&mut ctx.pending_waits);
    let mut still = Vec::new();
    for (id, tid) in pending {
        if let Some((_, ok, _)) = terminal_finished(&ctx.shared, &tid) {
            let _ = write_wait_result(&mut ctx.stdin, &ctx.acp_trace, id, ok);
        } else {
            still.push((id, tid));
        }
    }
    ctx.pending_waits = still;
}

fn terminal_create(ctx: &mut AcpThread, params: &Value) -> std::result::Result<Value, String> {
    let command = params
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params
        .get("args")
        .and_then(Value::as_array)
        .map(|args| {
            args.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let command = if args.is_empty() {
        command.to_string()
    } else {
        format!("{command} {args}")
    };
    let cwd = params.get("cwd").and_then(Value::as_str).map(PathBuf::from);
    let limit = params.get("outputByteLimit").and_then(Value::as_u64);
    let workdir = workdir_from_goose_cwd(&ctx.shared.workspace, cwd.as_deref())?;
    let id = format!(
        "t{}",
        ctx.shared.next_terminal.fetch_add(1, Ordering::SeqCst)
    );
    let term_cancel = Cancel::new();
    {
        let pipeline = ctx.shared.cancel.lock().expect("goose cancel mutex");
        // Barge-in is cancel_generation, not shutdown; also refuse shells after
        // session/cancel so they cannot outlive generate().
        if pipeline.is_shutdown() || ctx.prompt_cancelled {
            term_cancel.shutdown();
        }
    }
    let finished = Arc::new(Mutex::new(None));
    ctx.shared
        .terminals
        .lock()
        .expect("goose terminals mutex")
        .insert(
            id.clone(),
            StoredTerminal {
                cancel: term_cancel.clone(),
                finished: Arc::clone(&finished),
            },
        );
    spawn_shell(
        ctx,
        id.clone(),
        command,
        workdir,
        limit,
        term_cancel,
        finished,
    );
    Ok(json!({ "terminalId": id }))
}

fn spawn_shell(
    ctx: &AcpThread,
    id: String,
    command: String,
    workdir: String,
    limit: Option<u64>,
    exec_cancel: Cancel,
    finished: Arc<Mutex<Option<(String, bool, bool)>>>,
) {
    let workspace = ctx.shared.workspace.clone();
    let permissions = ctx.shared.permissions.clone();
    let tool_events = Arc::clone(&ctx.tool_events);
    let wake_tx = ctx.wake_tx.clone();
    let shell_inflight = Arc::clone(&ctx.shared);
    let spawn_finished = Arc::clone(&finished);
    ctx.shared.shell_inflight.fetch_add(1, Ordering::SeqCst);
    let spawn = thread::Builder::new()
        .name(format!("syllabix-acp-{id}"))
        .spawn(move || {
            let call = ToolCall {
                id: id.clone(),
                name: "shell".into(),
                arguments: json!({ "command": command, "workdir": workdir }),
            };
            note_tool(
                &tool_events,
                "call",
                "shell",
                &id,
                &call.arguments.to_string(),
                "",
            );
            let result =
                executor::execute_with_permissions(&call, &workspace, &permissions, &exec_cancel);
            note_tool(
                &tool_events,
                "result",
                "shell",
                &id,
                &call.arguments.to_string(),
                &result.content,
            );
            let (output, truncated) = cap_terminal_output(&result.content, limit);
            *spawn_finished.lock().expect("goose terminal result mutex") =
                Some((output, result.ok, truncated));
            shell_inflight.shell_inflight.fetch_sub(1, Ordering::SeqCst);
            let _ = wake_tx.send(AcpIn::TerminalDone);
        });
    if let Err(err) = spawn {
        *finished.lock().expect("goose terminal result mutex") = Some((
            format!("shell command could not start: {err}"),
            false,
            false,
        ));
        ctx.shared.shell_inflight.fetch_sub(1, Ordering::SeqCst);
        let _ = ctx.wake_tx.send(AcpIn::TerminalDone);
    }
}

fn handle_session_update(
    evt_tx: &Sender<AcpEvt>,
    acp_tool_calls: &Mutex<HashSet<String>>,
    params: &Value,
) {
    let update = params.get("update").unwrap_or(params);
    let kind = update
        .get("sessionUpdate")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "agent_message_chunk" => {
            if let Some(text) = text_block(update.get("content")) {
                let _ = evt_tx.send(AcpEvt::Token(text));
            }
        }
        // ACP `title` is a UI label (`shell · find . …`), not a tool name.
        // `terminal/create` already records `shell` call/result via
        // `execute_with_permissions`. Record only the `toolCallId` so a
        // validity check can require one `terminal/create` per Goose call.
        "tool_call" | "tool_call_update" => {
            if let Some(id) = update.get("toolCallId").and_then(Value::as_str) {
                if !id.is_empty() {
                    acp_tool_calls
                        .lock()
                        .expect("goose acp-tool-calls mutex")
                        .insert(id.to_string());
                }
            }
        }
        _ => {}
    }
}

fn text_block(content: Option<&Value>) -> Option<String> {
    let content = content?;
    if let Some(text) = content.get("text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    content.as_str().map(str::to_string)
}

fn note_tool(
    events: &Mutex<Vec<ToolTurnEvent>>,
    kind: &str,
    name: &str,
    call_id: &str,
    arguments: &str,
    content: &str,
) {
    events
        .lock()
        .expect("goose tool-events mutex")
        .push(ToolTurnEvent {
            kind: kind.into(),
            name: name.into(),
            call_id: call_id.into(),
            arguments: arguments.into(),
            content: content.into(),
        });
}

fn record_acp_trace(trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>, dir: &'static str, line: &str) {
    let Some(trace) = trace else {
        return;
    };
    let raw = line.trim_end_matches(['\n', '\r']).to_string();
    if raw.is_empty() {
        return;
    }
    trace
        .lock()
        .expect("goose acp-trace mutex")
        .push(AcpTraceLine {
            dir: dir.into(),
            raw,
        });
}

// Hand-rolled NDJSON over stdio: `agent-client-protocol` 3.3.0 cannot link
// here because `agent-client-protocol-schema` 1.11.0 pulls `serde_with` →
// `serde_json ^1.0.145` while this workspace pins `serde_json =1.0.134`.
fn write_line(
    stdin: &mut ChildStdin,
    trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    line: &str,
) -> std::result::Result<(), String> {
    record_acp_trace(trace, "out", line);
    let mut bytes = line.as_bytes().to_vec();
    if !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    stdin
        .write_all(&bytes)
        .and_then(|_| stdin.flush())
        .map_err(|err| err.to_string())
}

fn write_rpc(
    stdin: &mut ChildStdin,
    trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    id: u64,
    method: &str,
    params: &impl serde::Serialize,
) -> std::result::Result<(), String> {
    let mut msg = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    });
    if id != 0 {
        msg["id"] = json!(id);
    }
    let line = serde_json::to_string(&msg).map_err(|err| err.to_string())?;
    write_line(stdin, trace, &line)
}

fn write_result(
    stdin: &mut ChildStdin,
    trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    id: Value,
    result: Value,
) -> std::result::Result<(), String> {
    let msg = json!({ "jsonrpc": "2.0", "id": id, "result": result });
    let line = serde_json::to_string(&msg).map_err(|err| err.to_string())?;
    write_line(stdin, trace, &line)
}

fn write_rpc_error(
    stdin: &mut ChildStdin,
    trace: &Option<Arc<Mutex<Vec<AcpTraceLine>>>>,
    id: Value,
    code: i64,
    message: String,
) -> std::result::Result<(), String> {
    let msg = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    });
    let line = serde_json::to_string(&msg).map_err(|err| err.to_string())?;
    write_line(stdin, trace, &line)
}

impl Llm for GooseAgent {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn debug_meta(&self) -> Option<LlmDebugMeta> {
        Some(LlmDebugMeta {
            provider: PROVIDER_NAME.into(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
            request_id: String::new(),
        })
    }

    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        let mut events =
            std::mem::take(&mut *self.tool_events.lock().expect("goose tool-events mutex"));
        events.extend(mcp::take_tool_events_file(&self.mcp_tool_events));
        events
    }

    fn generate(
        &mut self,
        history: &[HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        let _ = history;
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if let Some(err) = self.drain_stale_events() {
            return Err(err);
        }
        *self.shared.cancel.lock().expect("goose cancel mutex") = cancel.clone();
        let generation = cancel.generation();
        self.cmd_tx
            .send(AcpIn::Cmd(AcpCmd::Prompt(user.text.clone())))
            .map_err(|_| provider_err("Goose stopped"))?;
        let mut index = 0u32;
        let mut pending: Option<TokenChunk> = None;
        let mut idle_since = Instant::now();
        let mut cancelling = false;
        let mut cancel_deadline: Option<Instant> = None;
        let result = loop {
            if !cancelling && (cancel.is_stale(generation) || cancel.is_shutdown()) {
                self.begin_cancel(&mut cancelling, &mut cancel_deadline, &mut pending);
            }
            if cancelling && cancel_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                self.drop_stale_events();
                break Err(Error::Cancelled);
            }
            let wait = cancel_deadline
                .map(|deadline| {
                    deadline
                        .saturating_duration_since(Instant::now())
                        .min(CANCEL_POLL)
                })
                .unwrap_or(CANCEL_POLL);
            match self.evt_rx.recv_timeout(wait) {
                Ok(AcpEvt::Token(text)) => {
                    if cancelling {
                        continue;
                    }
                    idle_since = Instant::now();
                    if let Some(earlier) = pending.replace(TokenChunk {
                        turn: user.turn,
                        generation,
                        index,
                        text,
                        is_last: false,
                    }) {
                        index += 1;
                        if on_token(earlier).is_err() {
                            self.begin_cancel(&mut cancelling, &mut cancel_deadline, &mut pending);
                        }
                    }
                }
                Ok(AcpEvt::TurnEnd { cancelled }) => {
                    if cancelling || cancelled {
                        self.drop_stale_events();
                        break Err(Error::Cancelled);
                    }
                    let mut chunk = pending.take().unwrap_or(TokenChunk {
                        turn: user.turn,
                        generation,
                        index,
                        text: String::new(),
                        is_last: false,
                    });
                    chunk.index = index;
                    chunk.is_last = true;
                    if on_token(chunk).is_err() {
                        break Err(Error::Cancelled);
                    }
                    break Ok(());
                }
                Ok(AcpEvt::Died(message) | AcpEvt::Fail(message)) => {
                    break Err(provider_err(message));
                }
                Ok(AcpEvt::Ready) => {}
                Err(RecvTimeoutError::Timeout) => {
                    // Shells run off the worker; do not count that wait as idle
                    // silence from Goose.
                    if !cancelling && self.shared.shell_inflight.load(Ordering::SeqCst) > 0 {
                        idle_since = Instant::now();
                    } else if !cancelling && idle_since.elapsed() >= idle_limit() {
                        break Err(provider_err("Goose sent no reply"));
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    break Err(provider_err("Goose stopped"));
                }
            }
        };
        self.flush_acp_trace(user.turn);
        result
    }
}

impl GooseAgent {
    fn begin_cancel(
        &self,
        cancelling: &mut bool,
        cancel_deadline: &mut Option<Instant>,
        pending: &mut Option<TokenChunk>,
    ) {
        if *cancelling {
            return;
        }
        *cancelling = true;
        *cancel_deadline = Some(Instant::now() + CANCEL_DEADLINE);
        *pending = None;
        let _ = self.cmd_tx.send(AcpIn::Cmd(AcpCmd::Cancel));
    }

    fn drain_stale_events(&mut self) -> Option<Error> {
        loop {
            match self.evt_rx.try_recv() {
                Ok(AcpEvt::Died(message) | AcpEvt::Fail(message)) => {
                    return Some(provider_err(message));
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Some(provider_err("Goose stopped"));
                }
            }
        }
    }

    fn drop_stale_events(&mut self) {
        while let Ok(event) = self.evt_rx.try_recv() {
            if let AcpEvt::Died(_) | AcpEvt::Fail(_) = event {
                break;
            }
        }
    }
}

impl Drop for GooseAgent {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(AcpIn::Cmd(AcpCmd::Shutdown));
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{DiscoveredSkill, SkillManifest, SkillSource};
    use std::path::PathBuf;

    fn scratch_path_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "syllabix-goose-config-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(goose_config_path(&root).parent().unwrap()).unwrap();
        root
    }

    #[test]
    fn prepare_goose_merge_keeps_user_keys_and_updates_provider_model() {
        let root = scratch_path_root("prepare-merge");
        let path = goose_config_path(&root);
        std::fs::write(
            &path,
            "GOOSE_TEMPERATURE: 0.2\nproviders:\n  anthropic:\n    enabled: true\n    model: old\n",
        )
        .unwrap();
        let gguf = std::env::temp_dir().join(format!(
            "syllabix-goose-prepare-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&gguf, b"x").unwrap();
        let mut config = crate::AgentConfig::v0();
        config.llm_model = gguf.to_string_lossy().into();
        let cache = crate::models::ModelCache::v0();
        let prepared = prepare_goose_from_pipeline(
            &root,
            &config,
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            None,
        )
        .expect("local goose prepare");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("GOOSE_TEMPERATURE: 0.2"), "{body}");
        assert!(body.contains("anthropic"), "{body}");
        assert!(body.contains("active_provider: local"), "{body}");
        assert!(body.contains(&gguf.to_string_lossy().to_string()), "{body}");
        assert!(!body.contains("OPENAI_API_KEY"), "{body}");
        assert!(prepared.child_env.is_empty());
    }

    #[test]
    fn prepare_goose_openai_compatible_maps_key_env_not_config() {
        let root = scratch_path_root("prepare-openai");
        let mut config = crate::AgentConfig::v0();
        config.llm = LlmProvider::OpenAiCompatible;
        config.llm_model = "gpt-test".into();
        config.llm_base_url = Some("http://127.0.0.1:11434/v1".into());
        let cache = crate::models::ModelCache::v0();
        let prepared = prepare_goose_from_pipeline(
            &root,
            &config,
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            Some(&Zeroizing::new("placeholder-key".into())),
        )
        .expect("openai-compatible goose prepare");
        let body = std::fs::read_to_string(goose_config_path(&root)).unwrap();
        assert!(body.contains("active_provider: openai"), "{body}");
        assert!(!body.contains("placeholder-key"), "{body}");
        assert!(
            prepared
                .child_env
                .iter()
                .any(|(k, v)| k == "OPENAI_API_KEY" && v == "placeholder-key"),
            "{:?}",
            prepared.child_env
        );
        assert!(
            prepared.child_env.iter().any(|(k, _)| k == "OPENAI_HOST"),
            "{:?}",
            prepared.child_env
        );
    }

    fn prepare_fixture_config() -> (crate::models::ModelCache, crate::AgentConfig) {
        (crate::models::ModelCache::v0(), crate::AgentConfig::v0())
    }

    fn run_prepare(
        root: &Path,
        config: &AgentConfig,
        llm_api_key: Option<&Zeroizing<String>>,
    ) -> Result<GoosePipelinePrepared> {
        let (cache, _) = prepare_fixture_config();
        prepare_goose_from_pipeline(
            root,
            config,
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut crate::models::NoProgress,
            &Cancel::new(),
            llm_api_key,
        )
    }

    fn api_key_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().expect("api key env lock")
    }

    #[test]
    fn prepare_goose_anthropic_maps_syllabix_key_to_anthropic_api_key() {
        let root = scratch_path_root("prepare-anthropic");
        let mut config = crate::AgentConfig::v0();
        config.llm = LlmProvider::External("anthropic".into());
        config.llm_model = "claude-sonnet-4-5".into();
        let prepared = run_prepare(
            &root,
            &config,
            Some(&Zeroizing::new("anthropic-test-key".into())),
        )
        .expect("anthropic prepare");
        let body = std::fs::read_to_string(goose_config_path(&root)).unwrap();
        assert!(body.contains("active_provider: anthropic"), "{body}");
        assert!(!body.contains("anthropic-test-key"), "{body}");
        assert!(
            prepared
                .child_env
                .iter()
                .any(|(k, v)| k == "ANTHROPIC_API_KEY" && v == "anthropic-test-key"),
            "{:?}",
            prepared.child_env
        );
    }

    #[test]
    fn prepare_goose_bedrock_passthrough_copies_listed_env_vars() {
        let _lock = api_key_env_lock();
        let prior = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            API_KEY_ENV,
        ]
        .iter()
        .map(|name| (*name, std::env::var(name).ok()))
        .collect::<Vec<_>>();
        std::env::set_var("AWS_ACCESS_KEY_ID", "ak-test");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "sk-test");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::remove_var(API_KEY_ENV);

        let root = scratch_path_root("prepare-bedrock");
        let mut config = crate::AgentConfig::v0();
        config.llm = LlmProvider::External("bedrock".into());
        config.llm_model = "anthropic.claude-3-5-sonnet-20241022-v2:0".into();
        let prepared = run_prepare(&root, &config, None).expect("bedrock prepare");

        for (name, value) in prior {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }

        let body = std::fs::read_to_string(goose_config_path(&root)).unwrap();
        assert!(!body.contains("ak-test"), "{body}");
        assert!(!body.contains("sk-test"), "{body}");
        let env: HashMap<_, _> = prepared.child_env.into_iter().collect();
        assert_eq!(
            env.get("AWS_ACCESS_KEY_ID").map(String::as_str),
            Some("ak-test")
        );
        assert_eq!(
            env.get("AWS_SECRET_ACCESS_KEY").map(String::as_str),
            Some("sk-test")
        );
        assert_eq!(env.get("AWS_REGION").map(String::as_str), Some("us-east-1"));
    }

    #[test]
    fn prepare_goose_unknown_provider_reports_credential_mapping_error() {
        let root = scratch_path_root("prepare-unknown");
        let mut config = crate::AgentConfig::v0();
        config.llm = LlmProvider::External("some-unknown-vendor".into());
        config.llm_model = "model-x".into();
        let err = match run_prepare(&root, &config, None) {
            Err(err) => err,
            Ok(_) => panic!("expected unknown-provider prepare to fail"),
        };
        assert!(err.to_string().contains("pipeline.llm.provider"), "{err}");
        assert!(
            err.to_string()
                .contains("Syllabix only maps SYLLABIX_LLM_API_KEY"),
            "{err}"
        );
    }

    #[test]
    fn prepare_goose_openai_compatible_requires_key_for_non_loopback_base_url() {
        let _lock = api_key_env_lock();
        let prior = std::env::var(API_KEY_ENV).ok();
        std::env::remove_var(API_KEY_ENV);

        let root = scratch_path_root("prepare-openai-missing-key");
        let mut config = crate::AgentConfig::v0();
        config.llm = LlmProvider::OpenAiCompatible;
        config.llm_model = "gpt-4o-mini".into();
        config.llm_base_url = Some("https://api.openai.com/v1".into());
        let err = match run_prepare(&root, &config, None) {
            Err(err) => err,
            Ok(_) => panic!("expected missing-key prepare to fail"),
        };
        match prior {
            Some(value) => std::env::set_var(API_KEY_ENV, value),
            None => std::env::remove_var(API_KEY_ENV),
        }
        assert!(err.to_string().contains(API_KEY_ENV), "{err}");
        assert!(err.to_string().contains("openai-compatible"), "{err}");
    }

    #[test]
    fn spawn_env_sets_approve_mode_and_disables_keyring() {
        let env = goose_child_env(Path::new("/tmp/syllabix-goose"));
        let map: HashMap<_, _> = env.into_iter().collect();
        assert_eq!(map["GOOSE_PATH_ROOT"], "/tmp/syllabix-goose");
        assert_eq!(map["GOOSE_DISABLE_KEYRING"], "1");
        assert_eq!(map["GOOSE_MODE"], "approve");
        assert_eq!(map.len(), 3);
        assert!(!map.contains_key("GOOSE_ENABLE_SCHEDULER"));
        assert!(!map.contains_key("GOOSE_PROVIDER"));
        assert!(!map.contains_key("GOOSE_MODEL"));
        assert!(!map.contains_key("OPENAI_API_KEY"));
    }

    #[test]
    fn resolve_goose_model_reads_structured_provider_config() {
        let root = scratch_path_root("structured");
        std::fs::write(
            goose_config_path(&root),
            "active_provider: local\nproviders:\n  local:\n    enabled: true\n    model: lfm2.5-2.6b\n",
        )
        .unwrap();
        assert_eq!(resolve_goose_model(&root, false).unwrap(), "lfm2.5-2.6b");
    }

    #[test]
    fn resolve_goose_model_reads_legacy_flat_keys() {
        let root = scratch_path_root("legacy");
        std::fs::write(
            goose_config_path(&root),
            "GOOSE_PROVIDER: openai\nGOOSE_MODEL: gpt-4o-mini\n",
        )
        .unwrap();
        assert_eq!(resolve_goose_model(&root, false).unwrap(), "gpt-4o-mini");
    }

    #[test]
    fn resolve_goose_model_names_goose_config_when_missing() {
        let root = scratch_path_root("missing");
        let err = resolve_goose_model_in(&[goose_config_path(&root)], false).unwrap_err();
        assert!(err.to_string().contains("developer_harness.agent"), "{err}");
        assert!(err.to_string().contains("config.yaml"), "{err}");
        assert!(err.to_string().contains(GOOSE_PROVIDER_DOCS), "{err}");
    }

    #[test]
    fn resolve_goose_model_merges_config_layers() {
        let root = scratch_path_root("merge");
        let system = root.join("system.yaml");
        std::fs::write(&system, "providers:\n  local:\n    model: lfm2.5-2.6b\n").unwrap();
        let extra = root.join("extra.yaml");
        std::fs::write(&extra, "active_provider: local\n").unwrap();
        let user = goose_config_path(&root);
        std::fs::write(&user, "providers:\n  local:\n    enabled: true\n").unwrap();

        assert_eq!(
            resolve_goose_model_in(&[system, extra, user], false).unwrap(),
            "lfm2.5-2.6b"
        );
    }

    fn goose_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &LOCK
    }

    #[test]
    fn resolve_goose_model_prefers_inherited_goose_env() {
        let _lock = goose_env_lock().lock().expect("goose env lock");
        let names = ["GOOSE_PROVIDER", "GOOSE_MODEL", "ACTIVE_PROVIDER"];
        let prior: Vec<_> = names.iter().map(std::env::var_os).collect();
        std::env::set_var("GOOSE_PROVIDER", "openai");
        std::env::set_var("GOOSE_MODEL", "env-model");
        std::env::set_var("ACTIVE_PROVIDER", "local");

        let root = scratch_path_root("env");
        std::fs::write(
            goose_config_path(&root),
            "active_provider: local\nproviders:\n  local:\n    model: yaml-model\n",
        )
        .unwrap();
        let resolved = resolve_goose_model_in(&[goose_config_path(&root)], true);
        for (name, value) in names.iter().zip(prior) {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        assert_eq!(resolved.unwrap(), "env-model");
    }

    #[test]
    fn permission_yaml_allows_shell_and_refuses_file_tools() {
        let yaml = permission_yaml_body(&[]);
        assert!(yaml.contains("shell"), "{yaml}");
        assert!(yaml.contains("never_allow"), "{yaml}");
        for tool in ["tree", "write", "edit", "read_image"] {
            assert!(yaml.contains(tool), "{tool} in {yaml}");
        }
        let extra = permission_yaml_body(&["syllabix__announce".into()]);
        assert!(extra.contains("syllabix__announce"), "{extra}");
    }

    #[test]
    fn enabled_extensions_include_developer_shell_and_syllabix_mcp() {
        let value = enabled_extensions_value(
            Path::new("/bin/syllabix"),
            Path::new("/ws"),
            Path::new("/tmp/events.jsonl"),
        );
        let first = &value[0];
        assert_eq!(first["type"], "builtin");
        assert_eq!(first["name"], "developer");
        assert_eq!(first["available_tools"], json!(["shell"]));
        let mcp = &value[1];
        assert_eq!(mcp["type"], "mcp");
        assert_eq!(mcp["server"]["name"], "syllabix");
        assert_eq!(mcp["available_tools"], json!(["web_search", "web_fetch"]));
        assert_eq!(value.as_array().unwrap().len(), 2);
        let encoded = serde_json::to_string(&value).expect("encode");
        assert!(
            encoded.contains("\"available_tools\""),
            "Goose reads snake_case available_tools: {encoded}"
        );
        assert!(
            !encoded.contains("availableTools"),
            "camelCase is dropped by Goose: {encoded}"
        );
    }

    #[test]
    fn unstable_methods_share_one_prefix() {
        assert_eq!(
            goose_unstable_method("session/system-prompt/set"),
            "_goose/unstable/session/system-prompt/set"
        );
    }

    #[test]
    fn cancelled_stop_reason_is_detected() {
        assert!(prompt_was_cancelled(&json!({ "stopReason": "cancelled" })));
        assert!(!prompt_was_cancelled(&json!({ "stopReason": "end_turn" })));
    }

    #[test]
    fn skill_list_is_names_and_paths() {
        let discovery = SkillDiscovery {
            skills: vec![DiscoveredSkill {
                manifest: SkillManifest {
                    name: "repository-guide".into(),
                    description: "repo".into(),
                    inputs: Default::default(),
                },
                body: "full text that must not appear".into(),
                path: PathBuf::from("/tmp/skills/repository-guide/SKILL.md"),
                source: SkillSource::Repository,
            }],
            diagnostics: Vec::new(),
        };
        let list = skill_file_list(&discovery);
        assert!(list.contains("repository-guide"));
        assert!(list.contains("/tmp/skills/repository-guide/SKILL.md"));
        assert!(!list.contains("full text that must not appear"));
        assert!(skill_file_list(&SkillDiscovery::default()).is_empty());
    }

    #[test]
    fn goose_cwd_maps_to_workspace_relative_workdir() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-goose-wd-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("sub")).unwrap();
        let root = root.canonicalize().unwrap();
        assert_eq!(workdir_from_goose_cwd(&root, Some(&root)).unwrap(), ".");
        assert_eq!(
            workdir_from_goose_cwd(&root, Some(&root.join("sub"))).unwrap(),
            "sub"
        );
        assert!(workdir_from_goose_cwd(&root, Some(Path::new("/tmp"))).is_err());
    }

    #[test]
    fn output_cap_is_min_of_goose_and_8k() {
        let big = "x".repeat(20_000);
        let (out, truncated) = cap_terminal_output(&big, None);
        assert_eq!(out.len(), MAX_OUTPUT_BYTES);
        assert!(truncated);
        let (out, truncated) = cap_terminal_output(&big, Some(16));
        assert_eq!(out.len(), 16);
        assert!(truncated);
        let (out, truncated) = cap_terminal_output("hi", Some(1_000_000));
        assert_eq!(out, "hi");
        assert!(!truncated);
    }

    #[test]
    fn acp_tool_call_updates_do_not_record_take_tool_events() {
        let (tx, rx) = mpsc::channel();
        let acp_tool_calls = Mutex::new(HashSet::new());
        for status in ["pending", "in_progress", "completed"] {
            handle_session_update(
                &tx,
                &acp_tool_calls,
                &json!({
                    "update": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "call-1",
                        "title": "shell · find . -type f …",
                        "status": status,
                        "rawInput": { "command": "find . -type f" },
                    }
                }),
            );
        }
        handle_session_update(
            &tx,
            &acp_tool_calls,
            &json!({
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-2",
                    "title": "shell · find . -type f …",
                    "status": "pending",
                }
            }),
        );
        handle_session_update(
            &tx,
            &acp_tool_calls,
            &json!({
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": "done" },
                }
            }),
        );
        let mut ids: Vec<String> = acp_tool_calls
            .lock()
            .expect("ids")
            .iter()
            .cloned()
            .collect();
        ids.sort();
        assert_eq!(ids, ["call-1", "call-2"]);
        assert_eq!(rx.try_recv().ok(), Some(AcpEvt::Token("done".into())));
        assert!(rx.try_recv().is_err());
    }
}

#[cfg(test)]
#[path = "goose_tests.rs"]
mod goose_tests;
