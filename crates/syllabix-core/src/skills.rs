//! Read-only discovery of user-provided local SKILL.md instruction packages.
//!
//! Skills are parsed as small declarative manifests. Entrypoints run only
//! through host policy checks; optional content pins can freeze a skill file.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::policy::{FilesystemMode, NetworkMode, SecretPolicy};
use serde_yaml::{Mapping, Value};
use sha2::{Digest, Sha256};

/// Maximum size of one skill file before it is rejected from model context.
pub const MAX_SKILL_FILE_BYTES: usize = 64 * 1024;
/// Maximum size of a skill description.
pub const MAX_SKILL_DESCRIPTION_BYTES: usize = 512;
/// Maximum number of argv values in one declarative entrypoint.
pub const MAX_SKILL_ARGV_VALUES: usize = 32;
/// Maximum UTF-8 size of one argv value.
pub const MAX_SKILL_ARGV_VALUE_BYTES: usize = 4096;
/// Maximum skill execution timeout.
pub const MAX_SKILL_TIMEOUT_SECONDS: u64 = 300;
/// Default timeout for an entrypoint that does not specify one.
pub const DEFAULT_SKILL_TIMEOUT: Duration = Duration::from_secs(120);

/// Where a discovered skill came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkillSource {
    /// A user-level skill root explicitly configured by the user.
    Global,
    /// A repository-level skill root explicitly configured by the user.
    Repository,
}

impl SkillSource {
    /// Configuration label for this source.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Repository => "repository",
        }
    }

    /// Stable, model-facing category.
    pub fn model_label(self) -> &'static str {
        match self {
            Self::Global | Self::Repository => "custom",
        }
    }
}

/// One explicitly configured local skill root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRoot {
    /// Root path containing one directory per skill.
    pub path: PathBuf,
    /// Whether this is a repository or user-global root.
    pub source: SkillSource,
}

/// Explicit skill roots from syllabix.yaml.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillsConfig {
    /// Repository/global roots explicitly configured by the user.
    pub roots: Vec<SkillRoot>,
    /// Optional content pins. A matching pin freezes a file against edits;
    /// skills without a pin remain executable when their root is configured.
    pub pins: Vec<SkillPin>,
}

/// Optional content pin for one skill document. The path is resolved relative
/// to the workspace for repository skills and must be absolute for global ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillPin {
    pub path: PathBuf,
    pub sha256: String,
}

/// A deliberately small input declaration for a declarative entrypoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillInput {
    /// One of string, integer, number, or boolean.
    pub kind: String,
    /// Human-readable input guidance.
    pub description: String,
    /// Whether the entrypoint requires this input.
    pub required: bool,
    /// Optional declarative default.
    pub default: Option<Value>,
    /// Optional allowed values.
    pub enum_values: Option<Vec<Value>>,
}

/// A list-only, shell-free entrypoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillEntrypoint {
    /// Program and arguments. Placeholders must occupy a complete value.
    pub argv: Vec<String>,
}

/// Optional authority requested by a skill. Every field is a request, never a
/// grant; the host resolves it against the immutable session ceiling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillPermissions {
    pub filesystem: Option<FilesystemMode>,
    pub network: Option<NetworkMode>,
    pub secrets: Option<SecretPolicy>,
}

/// Parsed YAML front matter plus Markdown instructions.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillManifest {
    /// Lowercase kebab-case skill id.
    pub name: String,
    /// Short model-facing summary.
    pub description: String,
    /// Optional future input declarations.
    pub inputs: BTreeMap<String, SkillInput>,
    /// Optional declarative executable entrypoint.
    pub entrypoint: Option<SkillEntrypoint>,
    /// Maximum authority requested by this skill.
    pub permissions: SkillPermissions,
    /// Optional bounded wall-clock timeout.
    pub timeout: Option<Duration>,
}

/// One valid skill visible to the developer-harness model.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredSkill {
    /// Parsed metadata.
    pub manifest: SkillManifest,
    /// Markdown body, kept separate from machine metadata.
    pub body: String,
    /// Canonical SKILL.md path.
    pub path: PathBuf,
    /// Configured user-root provenance.
    pub source: SkillSource,
    /// Whether the configured root and optional content pin allow execution.
    pub entrypoint_available: bool,
}

/// A load problem kept out of model context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDiagnostic {
    /// File or root associated with the problem, when known.
    pub path: Option<PathBuf>,
    /// Diagnostic code for tests and UI adapters.
    pub code: String,
    /// Human-readable detail.
    pub message: String,
}

/// Result of deterministic skill discovery.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkillDiscovery {
    /// Valid, unique skills safe to place in model context.
    pub skills: Vec<DiscoveredSkill>,
    /// Invalid/duplicate entries for diagnostics only.
    pub diagnostics: Vec<SkillDiagnostic>,
}

impl SkillDiscovery {
    /// Discover explicitly configured user roots only.
    ///
    /// Missing configured roots are diagnostics, never fatal to the rest of
    /// the developer harness.
    pub fn discover(workspace: &Path, config: &SkillsConfig) -> Self {
        let roots = config.roots.iter().cloned();
        let mut candidates = Vec::new();
        let mut diagnostics = Vec::new();

        for root in roots {
            let root_path = resolve_root(&root, workspace);
            match fs::metadata(&root_path) {
                Ok(metadata) if metadata.is_dir() => {}
                Ok(_) => {
                    diagnostics.push(diagnostic(
                        root_path,
                        "root-not-directory",
                        "skill root is not a directory",
                    ));
                    continue;
                }
                Err(error) => {
                    diagnostics.push(diagnostic(root_path, "root-unavailable", error.to_string()));
                    continue;
                }
            }

            let mut entries = match fs::read_dir(&root_path) {
                Ok(entries) => entries
                    .filter_map(std::result::Result::ok)
                    .collect::<Vec<_>>(),
                Err(error) => {
                    diagnostics.push(diagnostic(root_path, "root-unreadable", error.to_string()));
                    continue;
                }
            };
            entries.sort_by_key(|entry| entry.file_name());

            for entry in entries {
                let skill_dir = entry.path();
                if !entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                    continue;
                }
                let skill_path = skill_dir.join("SKILL.md");
                let bytes = match fs::read(&skill_path) {
                    Ok(bytes) => bytes,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => {
                        diagnostics.push(diagnostic(
                            skill_path,
                            "skill-unreadable",
                            error.to_string(),
                        ));
                        continue;
                    }
                };
                if bytes.len() > MAX_SKILL_FILE_BYTES {
                    diagnostics.push(diagnostic(
                        skill_path,
                        "skill-too-large",
                        format!(
                            "SKILL.md is {} bytes; maximum is {}",
                            bytes.len(),
                            MAX_SKILL_FILE_BYTES
                        ),
                    ));
                    continue;
                }
                let canonical_path = match skill_path.canonicalize() {
                    Ok(path) => path,
                    Err(error) => {
                        diagnostics.push(diagnostic(
                            skill_path,
                            "skill-not-canonical",
                            error.to_string(),
                        ));
                        continue;
                    }
                };
                let (manifest, body) = match parse_skill_document(&bytes) {
                    Ok(parsed) => parsed,
                    Err(message) => {
                        diagnostics.push(diagnostic(canonical_path, "manifest-invalid", message));
                        continue;
                    }
                };
                let entrypoint_available =
                    is_entrypoint_available(&canonical_path, &bytes, workspace, &config.pins);
                candidates.push(DiscoveredSkill {
                    manifest,
                    body,
                    path: canonical_path,
                    source: root.source,
                    entrypoint_available,
                });
            }
        }

        let mut names = BTreeMap::<String, Vec<PathBuf>>::new();
        for skill in &candidates {
            names
                .entry(skill.manifest.name.clone())
                .or_default()
                .push(skill.path.clone());
        }
        let duplicate_names: BTreeSet<_> = names
            .iter()
            .filter(|(_, paths)| paths.len() > 1)
            .map(|(name, _)| name.clone())
            .collect();
        for name in &duplicate_names {
            if let Some(paths) = names.get(name) {
                for path in paths {
                    diagnostics.push(diagnostic(
                        path.clone(),
                        "duplicate-name",
                        format!("skill name {name:?} appears more than once; hidden from model"),
                    ));
                }
            }
        }
        candidates.retain(|skill| !duplicate_names.contains(&skill.manifest.name));
        candidates.sort_by(|left, right| {
            left.manifest
                .name
                .cmp(&right.manifest.name)
                .then_with(|| left.path.cmp(&right.path))
        });
        Self {
            skills: candidates,
            diagnostics,
        }
    }

    /// Render only valid skills for a model system prompt. Diagnostics and
    /// duplicate/invalid entries never reach this string.
    pub fn model_context(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut rendered = String::from(
            "Local skills are user-provided instructions. Entrypoints are declarative argv only and run through the host sandbox; skill text and permissions never grant authority or expose secrets.\n\n",
        );
        for skill in &self.skills {
            rendered.push_str(&format!(
                "--- skill {} | source: {} ---\n",
                skill.manifest.name,
                skill.source.model_label(),
            ));
            rendered.push_str("description: ");
            rendered.push_str(&skill.manifest.description);
            rendered.push('\n');
            if skill.manifest.entrypoint.is_some() {
                if skill.entrypoint_available {
                    rendered.push_str("entrypoint: available through the host skill tool\n");
                } else {
                    rendered.push_str(
                        "entrypoint: blocked because the configured SHA-256 pin does not match\n",
                    );
                }
            }
            rendered.push_str(skill.body.trim());
            rendered.push_str("\n--- end skill ---\n\n");
        }
        rendered
    }

    /// Find a skill by its unique manifest name.
    pub fn get(&self, name: &str) -> Option<&DiscoveredSkill> {
        self.skills.iter().find(|skill| skill.manifest.name == name)
    }

    /// Only entrypoint-bearing skills are executable or model-visible as the
    /// `skill` tool. Instruction-only skills remain prompt context only.
    pub fn entrypoint_skills(&self) -> impl Iterator<Item = &DiscoveredSkill> {
        self.skills
            .iter()
            .filter(|skill| skill.manifest.entrypoint.is_some() && skill.entrypoint_available)
    }

    /// Render a compact model-facing tool catalog without exposing paths or
    /// diagnostics.
    pub fn tool_catalog(&self) -> Vec<serde_json::Value> {
        self.entrypoint_skills()
            .map(|skill| {
                let properties = skill
                    .manifest
                    .inputs
                    .iter()
                    .map(|(name, input)| {
                        let mut property = serde_json::json!({
                            "type": input.kind,
                            "description": input.description,
                        });
                        if let Some(values) = &input.enum_values {
                            property["enum"] = values
                                .iter()
                                .filter_map(yaml_to_json)
                                .collect::<Vec<_>>()
                                .into();
                        }
                        (name.clone(), property)
                    })
                    .collect::<serde_json::Map<_, _>>();
                let required = skill
                    .manifest
                    .inputs
                    .iter()
                    .filter(|(_, input)| input.required && input.default.is_none())
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "name": skill.manifest.name,
                    "description": skill.manifest.description,
                    "inputs": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": properties,
                        "required": required,
                    }
                })
            })
            .collect()
    }
}

fn is_entrypoint_available(path: &Path, bytes: &[u8], workspace: &Path, pins: &[SkillPin]) -> bool {
    let digest = format!("{:x}", Sha256::digest(bytes));
    let mut matching_pins = pins.iter().filter(|binding| {
        let binding_path = if binding.path.is_absolute() {
            binding.path.clone()
        } else {
            workspace.join(&binding.path)
        };
        binding_path
            .canonicalize()
            .ok()
            .is_some_and(|binding_path| binding_path == path)
    });
    matching_pins
        .next()
        .is_none_or(|binding| binding.sha256.eq_ignore_ascii_case(&digest))
}

fn diagnostic(path: PathBuf, code: &str, message: impl Into<String>) -> SkillDiagnostic {
    SkillDiagnostic {
        path: Some(path),
        code: code.into(),
        message: message.into(),
    }
}

fn resolve_root(root: &SkillRoot, workspace: &Path) -> PathBuf {
    if root.path.is_absolute() {
        root.path.clone()
    } else {
        workspace.join(&root.path)
    }
}

fn parse_skill_document(bytes: &[u8]) -> std::result::Result<(SkillManifest, String), String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "SKILL.md must be UTF-8".to_string())?;
    let (front_matter, body) = split_front_matter(text)?;
    let value: Value = serde_yaml::from_str(front_matter)
        .map_err(|error| format!("front matter is not valid YAML: {error}"))?;
    let map = value
        .as_mapping()
        .ok_or_else(|| "front matter must be a YAML mapping".to_string())?;
    deny_unknown_manifest_keys(map)?;
    let name = required_string(map, "name")?;
    validate_skill_name(name)?;
    let description = required_string(map, "description")?.trim().to_string();
    if description.len() > MAX_SKILL_DESCRIPTION_BYTES {
        return Err(format!(
            "description exceeds {MAX_SKILL_DESCRIPTION_BYTES} bytes"
        ));
    }
    if body.trim().is_empty() {
        return Err("Markdown body must be non-empty".into());
    }
    Ok((
        SkillManifest {
            name: name.to_string(),
            description,
            inputs: parse_inputs(map.get("inputs"))?,
            entrypoint: parse_entrypoint(map.get("entrypoint"))?,
            permissions: parse_permissions(map.get("permissions"))?,
            timeout: parse_timeout(map.get("timeout_seconds"))?,
        },
        body.to_string(),
    ))
}

fn split_front_matter(text: &str) -> std::result::Result<(&str, &str), String> {
    let Some(text) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return Err("SKILL.md must start with YAML front matter (---)".into());
    };
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            return Ok((&text[..offset], &text[offset + line.len()..]));
        }
        offset += line.len();
    }
    Err("YAML front matter is missing its closing ---".into())
}

fn deny_unknown_manifest_keys(map: &Mapping) -> std::result::Result<(), String> {
    for key in map.keys() {
        let name = key
            .as_str()
            .ok_or_else(|| "front matter keys must be strings".to_string())?;
        if !matches!(
            name,
            "name" | "description" | "inputs" | "entrypoint" | "permissions" | "timeout_seconds"
        ) {
            return Err(format!("unknown front matter key {name:?}"));
        }
    }
    Ok(())
}

fn parse_entrypoint(value: Option<&Value>) -> std::result::Result<Option<SkillEntrypoint>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let map = value
        .as_mapping()
        .ok_or_else(|| "entrypoint must be a YAML mapping".to_string())?;
    for key in map.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| "entrypoint keys must be strings".to_string())?;
        if key != "argv" {
            return Err(format!("unknown field entrypoint.{key}"));
        }
    }
    let argv = map
        .get("argv")
        .and_then(Value::as_sequence)
        .ok_or_else(|| "entrypoint.argv must be a non-empty sequence".to_string())?;
    if argv.is_empty() || argv.len() > MAX_SKILL_ARGV_VALUES {
        return Err(format!(
            "entrypoint.argv must contain 1-{MAX_SKILL_ARGV_VALUES} values"
        ));
    }
    let mut values = Vec::with_capacity(argv.len());
    for (index, value) in argv.iter().enumerate() {
        let value = value
            .as_str()
            .ok_or_else(|| format!("entrypoint.argv[{index}] must be a string"))?;
        if value.is_empty() || value.len() > MAX_SKILL_ARGV_VALUE_BYTES || value.contains('\0') {
            return Err(format!("entrypoint.argv[{index}] has an invalid length"));
        }
        values.push(value.to_string());
    }
    Ok(Some(SkillEntrypoint { argv: values }))
}

fn parse_permissions(value: Option<&Value>) -> std::result::Result<SkillPermissions, String> {
    let Some(value) = value else {
        return Ok(SkillPermissions::default());
    };
    let map = value
        .as_mapping()
        .ok_or_else(|| "permissions must be a YAML mapping".to_string())?;
    for key in map.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| "permissions keys must be strings".to_string())?;
        if !matches!(key, "filesystem" | "network" | "secrets") {
            return Err(format!("unknown field permissions.{key}"));
        }
    }
    let filesystem = match permission_string(map, "filesystem")? {
        None => None,
        Some("read-only") => Some(FilesystemMode::ReadOnly),
        Some("workspace-write") => Some(FilesystemMode::WorkspaceWrite),
        Some("danger-full-access") => Some(FilesystemMode::DangerFullAccess),
        Some(value) => {
            return Err(format!(
                "unsupported permissions.filesystem value {value:?}"
            ))
        }
    };
    let network = match permission_string(map, "network")? {
        None => None,
        Some("none") => Some(NetworkMode::None),
        Some("allow") => Some(NetworkMode::Allow),
        Some(value) => return Err(format!("unsupported permissions.network value {value:?}")),
    };
    let secrets = match permission_string(map, "secrets")? {
        None => None,
        Some("none") => Some(SecretPolicy::None),
        Some(value) => return Err(format!("unsupported permissions.secrets value {value:?}")),
    };
    Ok(SkillPermissions {
        filesystem,
        network,
        secrets,
    })
}

fn permission_string<'a>(
    map: &'a Mapping,
    key: &str,
) -> std::result::Result<Option<&'a str>, String> {
    let Some(value) = map.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(Some)
        .ok_or_else(|| format!("permissions.{key} must be a string"))
}

fn parse_timeout(value: Option<&Value>) -> std::result::Result<Option<Duration>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let seconds = value
        .as_u64()
        .ok_or_else(|| "timeout_seconds must be a positive integer".to_string())?;
    if seconds == 0 || seconds > MAX_SKILL_TIMEOUT_SECONDS {
        return Err(format!(
            "timeout_seconds must be between 1 and {MAX_SKILL_TIMEOUT_SECONDS}"
        ));
    }
    Ok(Some(Duration::from_secs(seconds)))
}

fn required_string<'a>(map: &'a Mapping, key: &str) -> std::result::Result<&'a str, String> {
    let value = map
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a non-empty string"))?;
    if value.trim().is_empty() {
        return Err(format!("{key} must be a non-empty string"));
    }
    Ok(value)
}

fn validate_skill_name(name: &str) -> std::result::Result<(), String> {
    let bytes = name.as_bytes();
    if name.len() > 64
        || name.is_empty()
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes[name.len() - 1].is_ascii_alphanumeric()
    {
        return Err("name must be lowercase kebab-case (1-64 characters)".into());
    }
    Ok(())
}

fn parse_inputs(
    value: Option<&Value>,
) -> std::result::Result<BTreeMap<String, SkillInput>, String> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let map = value
        .as_mapping()
        .ok_or_else(|| "inputs must be a YAML mapping".to_string())?;
    let mut inputs = BTreeMap::new();
    for (key, value) in map {
        let name = key
            .as_str()
            .ok_or_else(|| "input names must be strings".to_string())?;
        if name.is_empty()
            || !name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
            })
        {
            return Err(format!("input name {name:?} is invalid"));
        }
        let spec = value
            .as_mapping()
            .ok_or_else(|| format!("inputs.{name} must be a YAML mapping"))?;
        for field in spec.keys() {
            let field = field
                .as_str()
                .ok_or_else(|| format!("inputs.{name} keys must be strings"))?;
            if !matches!(
                field,
                "type" | "description" | "required" | "default" | "enum"
            ) {
                return Err(format!("unknown field inputs.{name}.{field}"));
            }
        }
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("inputs.{name}.type must be a string"))?;
        if !matches!(kind, "string" | "integer" | "number" | "boolean") {
            return Err(format!("inputs.{name}.type is unsupported: {kind:?}"));
        }
        let description = spec
            .get("description")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("inputs.{name}.description must be non-empty"))?
            .to_string();
        let required = spec
            .get("required")
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| format!("inputs.{name}.required must be a boolean"))
            })
            .transpose()?
            .unwrap_or(false);
        let enum_values = spec
            .get("enum")
            .map(|value| {
                value
                    .as_sequence()
                    .cloned()
                    .ok_or_else(|| format!("inputs.{name}.enum must be a sequence"))
            })
            .transpose()?;
        if let Some(default) = spec.get("default") {
            if !yaml_value_matches_kind(default, kind) {
                return Err(format!("inputs.{name}.default must be a {kind}"));
            }
        }
        if let Some(enum_values) = &enum_values {
            for value in enum_values {
                if !yaml_value_matches_kind(value, kind) {
                    return Err(format!("inputs.{name}.enum contains a non-{kind} value"));
                }
            }
            if let Some(default) = spec.get("default") {
                if !enum_values.iter().any(|value| value == default) {
                    return Err(format!("inputs.{name}.default is not in enum"));
                }
            }
        }
        inputs.insert(
            name.to_string(),
            SkillInput {
                kind: kind.to_string(),
                description,
                required,
                default: spec.get("default").cloned(),
                enum_values,
            },
        );
    }
    Ok(inputs)
}

/// Validate JSON input values and apply declarative defaults. Unknown inputs,
/// type mismatches, and enum violations are rejected before any child starts.
pub fn validate_inputs(
    manifest: &SkillManifest,
    supplied: Option<&serde_json::Value>,
) -> std::result::Result<BTreeMap<String, serde_json::Value>, String> {
    let supplied = supplied
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let object = supplied
        .as_object()
        .ok_or_else(|| "skill inputs must be an object".to_string())?;
    for name in object.keys() {
        if !manifest.inputs.contains_key(name) {
            return Err(format!("skill input {name:?} is not declared"));
        }
    }
    let mut values = BTreeMap::new();
    for (name, spec) in &manifest.inputs {
        let Some(value) = object
            .get(name)
            .cloned()
            .or_else(|| spec.default.as_ref().and_then(yaml_to_json))
            .or_else(|| (!spec.required).then_some(serde_json::Value::Null))
        else {
            return Err(format!("required skill input {name:?} is missing"));
        };
        if value.is_null() && !spec.required {
            continue;
        }
        if !json_value_matches_kind(&value, &spec.kind) {
            return Err(format!("skill input {name:?} must be a {}", spec.kind));
        }
        if let Some(enum_values) = &spec.enum_values {
            if !enum_values
                .iter()
                .any(|allowed| yaml_to_json(allowed).is_some_and(|allowed| allowed == value))
            {
                return Err(format!("skill input {name:?} is not an allowed value"));
            }
        }
        values.insert(name.clone(), value);
    }
    Ok(values)
}

/// Expand an entrypoint into direct argv values. Placeholder syntax is
/// deliberately whole-value-only so user data can never become shell syntax.
pub fn resolve_entrypoint_argv(
    manifest: &SkillManifest,
    inputs: Option<&serde_json::Value>,
    skill_dir: &Path,
    workspace: &Path,
    temp_dir: &Path,
) -> std::result::Result<Vec<String>, String> {
    let entrypoint = manifest
        .entrypoint
        .as_ref()
        .ok_or_else(|| "skill has no entrypoint".to_string())?;
    let inputs = validate_inputs(manifest, inputs)?;
    entrypoint
        .argv
        .iter()
        .map(|value| {
            resolve_argv_value(
                value,
                &manifest.inputs,
                &inputs,
                skill_dir,
                workspace,
                temp_dir,
            )
        })
        .collect()
}

fn resolve_argv_value(
    value: &str,
    declared_inputs: &BTreeMap<String, SkillInput>,
    inputs: &BTreeMap<String, serde_json::Value>,
    skill_dir: &Path,
    workspace: &Path,
    temp_dir: &Path,
) -> std::result::Result<String, String> {
    let is_placeholder = value.starts_with("{{") && value.ends_with("}}");
    if value.contains("{{") || value.contains("}}") {
        if !is_placeholder {
            return Err("skill placeholders must occupy a complete argv value".into());
        }
        let expression = value[2..value.len() - 2].trim();
        let (expression, default) = match expression.split_once("| default:") {
            Some((name, default)) => (name.trim(), Some(parse_default_literal(default.trim())?)),
            None => (expression, None),
        };
        let resolved = match expression {
            "skill_dir" => Some(skill_dir.to_string_lossy().into_owned()),
            "workspace" => Some(workspace.to_string_lossy().into_owned()),
            "temp_dir" => Some(temp_dir.to_string_lossy().into_owned()),
            name if name.starts_with("inputs.") => {
                let input = &name["inputs.".len()..];
                if input.is_empty() || !declared_inputs.contains_key(input) {
                    return Err(format!("skill input {input:?} is not declared"));
                }
                if !inputs.contains_key(input) {
                    return default
                        .map(|value| value.to_string())
                        .ok_or_else(|| format!("skill input {input:?} is unavailable"));
                }
                Some(json_scalar_to_string(inputs.get(input).expect("checked"))?)
            }
            other => return Err(format!("unsupported skill placeholder {other:?}")),
        };
        return Ok(resolved
            .or(default.map(|value| value.to_string()))
            .unwrap_or_default());
    }
    Ok(value.to_string())
}

fn parse_default_literal(value: &str) -> std::result::Result<String, String> {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('\'') && value.ends_with('\''))
            || (value.starts_with('"') && value.ends_with('"')))
    {
        Ok(value[1..value.len() - 1].to_string())
    } else if !value.is_empty() && !value.contains(['{', '}', '\n', '\r']) {
        Ok(value.to_string())
    } else {
        Err("skill placeholder default must be a scalar literal".into())
    }
}

fn yaml_to_json(value: &Value) -> Option<serde_json::Value> {
    serde_json::to_value(value).ok()
}

fn yaml_value_matches_kind(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.as_str().is_some(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.as_f64().is_some(),
        "boolean" => value.as_bool().is_some(),
        _ => false,
    }
}

fn json_value_matches_kind(value: &serde_json::Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        _ => false,
    }
}

fn json_scalar_to_string(value: &serde_json::Value) -> std::result::Result<String, String> {
    match value {
        serde_json::Value::String(value) => Ok(value.clone()),
        serde_json::Value::Bool(value) => Ok(value.to_string()),
        serde_json::Value::Number(value) => Ok(value.to_string()),
        _ => Err("skill placeholders require scalar input values".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "syllabix-skills-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_skill(root: &Path, dir: &str, text: &str) -> PathBuf {
        let path = root.join(dir);
        fs::create_dir_all(&path).unwrap();
        let file = path.join("SKILL.md");
        fs::write(&file, text).unwrap();
        file
    }

    fn config(root: &Path) -> SkillsConfig {
        SkillsConfig {
            roots: vec![SkillRoot {
                path: root.to_path_buf(),
                source: SkillSource::Repository,
            }],
            pins: Vec::new(),
        }
    }

    #[test]
    fn parses_front_matter_and_body_without_entrypoints() {
        let text = b"---\nname: release-check\ndescription: Inspect a release.\ninputs:\n  ref:\n    type: string\n    description: Git ref\n    required: false\n---\n\n# Release check\n\nRead only.\n";
        let (manifest, body) = parse_skill_document(text).unwrap();
        assert_eq!(manifest.name, "release-check");
        assert_eq!(manifest.inputs["ref"].kind, "string");
        assert!(body.contains("# Release check"));
    }

    #[test]
    fn unknown_metadata_and_entrypoints_fail_closed() {
        for key in ["entrypoint", "permissions", "timeout_seconds", "surprise"] {
            let text = format!("---\nname: demo\ndescription: Demo\n{key}: true\n---\nbody\n");
            let error = parse_skill_document(text.as_bytes()).unwrap_err();
            assert!(error.contains(key), "{key}: {error}");
        }
    }

    #[test]
    fn invalid_and_duplicate_skills_are_diagnostics_only() {
        let root = temp_root("diagnostics");
        write_skill(
            &root,
            "broken",
            "---\nname: Bad_Name\ndescription: broken\n---\nbody\n",
        );
        let document = "---\nname: same\ndescription: One\n---\nbody\n";
        write_skill(&root, "one", document);
        write_skill(&root, "two", document);
        let found = SkillDiscovery::discover(&root, &config(&root));
        assert!(found.skills.is_empty());
        assert!(found
            .diagnostics
            .iter()
            .any(|diag| diag.code == "manifest-invalid"));
        assert_eq!(
            found
                .diagnostics
                .iter()
                .filter(|diag| diag.code == "duplicate-name")
                .count(),
            2
        );
        assert!(found.model_context().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn configured_skills_are_custom_and_have_model_labels() {
        let root = temp_root("sources");
        write_skill(
            &root,
            "global",
            "---\nname: global\ndescription: User\n---\nbody\n",
        );
        let found = SkillDiscovery::discover(&root, &config(&root));
        let context = found.model_context();
        assert!(context.contains("source: custom"));
        assert!(context.contains("user-provided instructions"));
        assert!(!context.contains("untrusted"));
        assert!(!context.contains("source: default"));
        assert!(!context.contains("trust:"));
        assert!(!context.contains("sha256:"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn labels_and_roots_are_total() {
        assert_eq!(SkillSource::Global.as_str(), "global");
        assert_eq!(SkillSource::Repository.as_str(), "repository");
        assert_eq!(SkillSource::Global.model_label(), "custom");
        assert_eq!(SkillSource::Repository.model_label(), "custom");

        let workspace = PathBuf::from("/workspace");
        assert_eq!(
            resolve_root(
                &SkillRoot {
                    path: PathBuf::from("skills"),
                    source: SkillSource::Repository,
                },
                &workspace,
            ),
            workspace.join("skills")
        );
        let absolute = PathBuf::from("/tmp/skills");
        assert_eq!(
            resolve_root(
                &SkillRoot {
                    path: absolute.clone(),
                    source: SkillSource::Global,
                },
                &workspace,
            ),
            absolute
        );
    }

    #[test]
    fn document_validation_rejects_malformed_shapes_and_limits() {
        for text in [
            "body only",
            "---\nname: demo\ndescription: Demo\nbody\n",
            "---\n[broken\n---\nbody\n",
            "---\n- item\n---\nbody\n",
            "---\ndescription: Demo\n---\nbody\n",
            "---\nname: demo\ndescription: Demo\n---\n",
            "---\nname: Demo_Name\ndescription: Demo\n---\nbody\n",
            "---\nname: demo_\ndescription: Demo\n---\nbody\n",
        ] {
            assert!(parse_skill_document(text.as_bytes()).is_err(), "{text:?}");
        }
        let long_name = "a".repeat(65);
        assert!(parse_skill_document(
            format!("---\nname: {long_name}\ndescription: Demo\n---\nbody\n").as_bytes()
        )
        .is_err());
        let long_description = "x".repeat(MAX_SKILL_DESCRIPTION_BYTES + 1);
        assert!(parse_skill_document(
            format!("---\nname: demo\ndescription: {long_description}\n---\nbody\n").as_bytes()
        )
        .is_err());
        assert!(parse_skill_document(b"---\nname: demo\ndescription: Demo\n---\n   \n").is_err());
        assert!(
            parse_skill_document(b"---\r\nname: demo\r\ndescription: Demo\r\n---\r\nbody\r\n")
                .is_ok()
        );
        assert!(parse_skill_document(&[0xff, 0xfe]).is_err());
    }

    #[test]
    fn input_validation_covers_optional_metadata_and_errors() {
        let valid = b"---\nname: demo\ndescription: Demo\ninputs:\n  query:\n    type: string\n    description: Search text\n    required: true\n    default: hello\n    enum: [hello, world]\n---\nbody\n";
        let (manifest, _) = parse_skill_document(valid).unwrap();
        let input = &manifest.inputs["query"];
        assert!(input.required);
        assert_eq!(input.default.as_ref().unwrap().as_str(), Some("hello"));
        assert_eq!(input.enum_values.as_ref().unwrap().len(), 2);

        for (label, inputs) in [
            ("not-a-map", "inputs: []"),
            (
                "bad-name",
                "inputs:\n  Bad_Name:\n    type: string\n    description: x",
            ),
            ("bad-spec", "inputs:\n  query: string"),
            (
                "bad-field",
                "inputs:\n  query:\n    type: string\n    description: x\n    extra: true",
            ),
            ("missing-type", "inputs:\n  query:\n    description: x"),
            (
                "bad-type",
                "inputs:\n  query:\n    type: path\n    description: x",
            ),
            (
                "bad-description",
                "inputs:\n  query:\n    type: string\n    description: '  '",
            ),
            (
                "bad-required",
                "inputs:\n  query:\n    type: string\n    description: x\n    required: yes",
            ),
            (
                "bad-enum",
                "inputs:\n  query:\n    type: string\n    description: x\n    enum: value",
            ),
        ] {
            let text = format!("---\nname: demo\ndescription: Demo\n{inputs}\n---\nbody\n");
            assert!(parse_skill_document(text.as_bytes()).is_err(), "{label}");
        }
    }

    #[test]
    fn discovery_reports_unavailable_roots_and_oversized_files() {
        let root = temp_root("root-diagnostics");
        let file_root = root.join("not-a-directory");
        fs::write(&file_root, b"file").unwrap();
        let oversized = "x".repeat(MAX_SKILL_FILE_BYTES + 1);
        write_skill(&root, "oversized", &oversized);
        fs::write(root.join("plain-file"), b"ignored").unwrap();
        let missing = root.join("missing");
        let found = SkillDiscovery::discover(
            &root,
            &SkillsConfig {
                roots: vec![
                    SkillRoot {
                        path: root.clone(),
                        source: SkillSource::Repository,
                    },
                    SkillRoot {
                        path: file_root.clone(),
                        source: SkillSource::Repository,
                    },
                    SkillRoot {
                        path: missing,
                        source: SkillSource::Global,
                    },
                ],
                pins: Vec::new(),
            },
        );
        assert!(found.skills.is_empty());
        assert!(found
            .diagnostics
            .iter()
            .any(|diag| diag.code == "root-not-directory"));
        assert!(found
            .diagnostics
            .iter()
            .any(|diag| diag.code == "root-unavailable"));
        assert!(found
            .diagnostics
            .iter()
            .any(|diag| diag.code == "skill-too-large"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn entrypoint_inputs_and_placeholders_are_strict_and_data_only() {
        let text = b"---
name: release-check
description: Check a release.
inputs:
  ref:
    type: string
    description: Git ref.
    required: false
    default: HEAD
  mode:
    type: string
    description: Check mode.
    enum: [fast, full]
    default: fast
entrypoint:
  argv: [bash, scripts/check.sh, '{{inputs.ref}}', '{{inputs.mode}}', '{{workspace}}', '{{temp_dir}}']
permissions:
  filesystem: read-only
  network: none
  secrets: none
timeout_seconds: 30
---
body
";
        let (manifest, _) = parse_skill_document(text).unwrap();
        assert_eq!(manifest.entrypoint.as_ref().unwrap().argv[0], "bash");
        assert_eq!(
            manifest.permissions.filesystem,
            Some(FilesystemMode::ReadOnly)
        );
        assert_eq!(manifest.timeout, Some(Duration::from_secs(30)));
        let argv = resolve_entrypoint_argv(
            &manifest,
            Some(&serde_json::json!({"ref": "feature with spaces", "mode": "full"})),
            Path::new("/skills/release-check"),
            Path::new("/workspace"),
            Path::new("/tmp/private"),
        )
        .unwrap();
        assert_eq!(argv[2], "feature with spaces");
        assert_eq!(argv[4], "/workspace");
        assert!(resolve_entrypoint_argv(
            &manifest,
            Some(&serde_json::json!({"ref": "$(touch pwned)", "mode": "fast"})),
            Path::new("/skills/release-check"),
            Path::new("/workspace"),
            Path::new("/tmp/private"),
        )
        .unwrap()[2]
            .contains("$(touch pwned)"));
        let embedded = SkillManifest {
            entrypoint: Some(SkillEntrypoint {
                argv: vec!["bash".into(), "prefix-{{inputs.ref}}".into()],
            }),
            ..manifest.clone()
        };
        assert!(resolve_entrypoint_argv(
            &embedded,
            Some(&serde_json::json!({"ref": "x"})),
            Path::new("/skills/release-check"),
            Path::new("/workspace"),
            Path::new("/tmp/private"),
        )
        .is_err());
    }

    #[test]
    fn entrypoint_permissions_and_inputs_fail_closed() {
        let text = b"---
name: demo
description: Demo.
inputs:
  count:
    type: integer
    description: Count.
    required: true
    enum: [1, 2]
entrypoint:
  argv: [printf, '{{inputs.count}}']
permissions:
  filesystem: danger-full-access
---
body
";
        let (manifest, _) = parse_skill_document(text).unwrap();
        assert!(validate_inputs(&manifest, Some(&serde_json::json!({}))).is_err());
        assert!(validate_inputs(&manifest, Some(&serde_json::json!({"count": 3}))).is_err());
        assert!(parse_skill_document(
            b"---\nname: demo\ndescription: Demo\npermissions:\n  network: true\n---\nbody\n"
        )
        .is_err());
    }

    #[test]
    fn custom_entrypoints_run_without_a_pin_and_optional_pins_block_changes() {
        let root = temp_root("trust");
        let file = write_skill(
            &root,
            "release-check",
            "---\nname: release-check\ndescription: Check.\nentrypoint:\n  argv: [printf, ok]\n---\nbody\n",
        );
        let bytes = fs::read(&file).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let unpinned = SkillDiscovery::discover(&root, &config(&root));
        assert_eq!(unpinned.entrypoint_skills().count(), 1);
        assert!(unpinned.model_context().contains("entrypoint: available"));
        let found = SkillDiscovery::discover(
            &root,
            &SkillsConfig {
                roots: vec![SkillRoot {
                    path: root.clone(),
                    source: SkillSource::Repository,
                }],
                pins: vec![SkillPin {
                    path: file.clone(),
                    sha256: hash,
                }],
            },
        );
        assert_eq!(found.entrypoint_skills().count(), 1);
        assert!(found.model_context().contains("entrypoint: available"));
        fs::write(&file, "---\nname: release-check\ndescription: Changed.\nentrypoint:\n  argv: [printf, changed]\n---\nbody\n").unwrap();
        let changed = SkillDiscovery::discover(
            &root,
            &SkillsConfig {
                roots: vec![SkillRoot {
                    path: root.clone(),
                    source: SkillSource::Repository,
                }],
                pins: vec![SkillPin {
                    path: file,
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                }],
            },
        );
        assert_eq!(changed.entrypoint_skills().count(), 0);
        assert!(changed
            .model_context()
            .contains("blocked because the configured SHA-256 pin does not match"));
        let live_changed = SkillDiscovery::discover(&root, &config(&root));
        assert_eq!(live_changed.entrypoint_skills().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
