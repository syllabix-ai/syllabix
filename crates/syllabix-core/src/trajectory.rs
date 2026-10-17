//! Opt-in per-turn agent trajectory (`diagnostics.trajectory`).
//!
//! Writes `turn-{id:03}/trajectory.jsonl`: the user utterance, every tool call
//! with its validated arguments, every bounded tool result, the assistant
//! reply, and the turn outcome — the same normalized shape for
//! `pipeline.agent: builtin` and `pipeline.agent: goose`, both of which feed
//! [`crate::turn_debug::TurnDebug`].
//!
//! Trajectories are the exported copy: the canonical `turn.json`, WAVs, and
//! `acp.jsonl` are never rewritten. Every field passes through a [`Redactor`]
//! before serialization so secret environment values, credential-shaped
//! tokens, and JSON values under sensitive keys never reach disk.

use serde_json::{json, Map, Value};

/// Marker written in place of a redacted secret.
pub const REDACTED: &str = "[REDACTED]";

/// Byte cap applied to each free-text field (arguments/content/text). Longer
/// values keep their leading bytes.
pub const MAX_FIELD_BYTES: usize = 4096;

/// Values shorter than this are never treated as exact secrets: matching them
/// would corrupt ordinary words in the transcript.
const MIN_SECRET_LEN: usize = 8;

/// One step in a turn's trajectory, rendered as a JSON object per line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryEntry {
    /// The user's recognized utterance for this turn.
    User { text: String },
    /// The assistant's reply for this turn.
    Assistant { text: String },
    /// A tool call the model requested, with its validated arguments.
    ToolCall {
        name: String,
        call_id: String,
        arguments: String,
    },
    /// A bounded tool result (or a rejection/limit) returned to the model.
    ToolResult {
        name: String,
        call_id: String,
        content: String,
        ok: bool,
    },
    /// How the turn ended (`completed`, `skipped`, or `cancelled`).
    Outcome { outcome: String },
}

impl TrajectoryEntry {
    /// Render this step as one redacted JSON object. Secrets are removed and
    /// long fields are truncated before serialization.
    pub fn to_json(&self, redactor: &Redactor) -> Value {
        match self {
            Self::User { text } => {
                json!({ "kind": "user", "text": redactor.redact_text(text) })
            }
            Self::Assistant { text } => {
                json!({ "kind": "assistant", "text": redactor.redact_text(text) })
            }
            Self::ToolCall {
                name,
                call_id,
                arguments,
            } => {
                let parsed = parse_arguments(arguments);
                json!({
                    "kind": "tool_call",
                    "name": name,
                    "call_id": call_id,
                    "arguments": redactor.redact_value(&parsed),
                })
            }
            Self::ToolResult {
                name,
                call_id,
                content,
                ok,
            } => {
                json!({
                    "kind": "tool_result",
                    "name": name,
                    "call_id": call_id,
                    "ok": ok,
                    "content": redactor.redact_text(content),
                })
            }
            Self::Outcome { outcome } => {
                json!({ "kind": "outcome", "outcome": outcome })
            }
        }
    }
}

/// Parse tool arguments as JSON when possible; the raw string otherwise.
fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

/// Redacts secrets from trajectory fields before they are written.
///
/// Two layers, following the pattern used by agent-telemetry redactors:
/// exact values gathered from secret-named environment variables, and
/// credential shapes (bearer/basic headers, `sk-`/GitHub/Slack/AWS keys, and
/// JWT triples) for tokens that arrive in tool output rather than the
/// launcher environment. Object values under sensitive key names (for example
/// `api_key`) are replaced outright.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Redactor {
    exact: Vec<String>,
}

impl Redactor {
    /// Build from the current process environment. Names matching the secret
    /// prefixes (`SYLLABIX_*_API_KEY`, `KITE_*`, `*_TOKEN`, `*_SECRET`,
    /// `*_PASSWORD`) contribute their literal values as exact secrets.
    pub fn from_env() -> Self {
        Self::from_pairs(std::env::vars())
    }

    /// Whether `name` looks like a secret-bearing environment variable.
    pub fn is_secret_name(name: &str) -> bool {
        let upper = name.to_ascii_uppercase();
        upper.starts_with("KITE_")
            || upper.ends_with("_API_KEY")
            || upper.ends_with("_API_SECRET")
            || upper.ends_with("_ACCESS_KEY")
            || upper.ends_with("_ACCESS_KEY_ID")
            || upper.ends_with("_TOKEN")
            || upper.ends_with("_SECRET")
            || upper.ends_with("_PASSWORD")
    }

    /// Build from explicit `(name, value)` pairs (the environment seam).
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut exact = Vec::new();
        for (name, value) in pairs {
            if value.len() >= MIN_SECRET_LEN && Self::is_secret_name(&name) {
                exact.push(value);
            }
        }
        exact.sort();
        exact.dedup();
        Self { exact }
    }

    /// Add one exact secret value. Values below [`MIN_SECRET_LEN`] are ignored.
    pub fn with_secret(mut self, value: impl Into<String>) -> Self {
        let value = value.into();
        if value.len() >= MIN_SECRET_LEN {
            self.exact.push(value);
            self.exact.sort();
            self.exact.dedup();
        }
        self
    }

    /// Redact and truncate one free-text field.
    pub fn redact_field(&self, text: &str) -> String {
        truncate(&self.redact(text))
    }

    /// Redact one trajectory field. A JSON object or array is redacted
    /// key-wise with [`Self::redact_value`]; anything else is treated as plain
    /// text. Either way long values truncate to [`MAX_FIELD_BYTES`].
    pub fn redact_text(&self, text: &str) -> String {
        match serde_json::from_str::<Value>(text) {
            Ok(value @ (Value::Object(_) | Value::Array(_))) => {
                self.redact_value(&value).to_string()
            }
            _ => self.redact_field(text),
        }
    }

    /// Redact every string in `text`.
    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for secret in &self.exact {
            if out.contains(secret.as_str()) {
                out = out.replace(secret.as_str(), REDACTED);
            }
        }
        redact_shapes(&out)
    }

    /// Recursively redact a JSON value: strings by shape, and object values
    /// under sensitive key names outright.
    pub fn redact_value(&self, value: &Value) -> Value {
        match value {
            Value::String(text) => Value::String(self.redact_field(text)),
            Value::Array(items) => {
                Value::Array(items.iter().map(|v| self.redact_value(v)).collect())
            }
            Value::Object(map) => {
                let mut out = Map::new();
                for (key, value) in map {
                    if is_sensitive_key(key) {
                        out.insert(key.clone(), Value::String(REDACTED.into()));
                    } else {
                        out.insert(key.clone(), self.redact_value(value));
                    }
                }
                Value::Object(out)
            }
            other => other.clone(),
        }
    }
}

/// Keys whose values are replaced wholesale, keyed by normalized token.
fn is_sensitive_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "authorization"
            | "cookie"
            | "credential"
            | "credentials"
            | "password"
            | "passwd"
            | "secret"
            | "token"
            | "apikey"
            | "api_key"
            | "api-key"
            | "access_token"
            | "refresh_token"
            | "id_token"
            | "client_secret"
            | "private_key"
    ) || lower.ends_with("_secret")
        || lower.ends_with("_password")
        || lower.ends_with("_api_key")
        || lower.ends_with("_token")
        || lower.ends_with("_apikey")
}

/// Token prefixes that identify well-known credential formats.
const TOKEN_PREFIXES: [&str; 11] = [
    "sk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "xoxb-", "xoxp-", "xoxa-", "AKIA", "AIza",
];

/// Replace credential-shaped tokens anywhere in `text`.
fn redact_shapes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut idx = 0;
    while idx < text.len() {
        if let Some(len) = credential_span_at(&text[idx..]) {
            out.push_str(&text[cursor..idx]);
            out.push_str(REDACTED);
            idx += len;
            cursor = idx;
            continue;
        }
        idx += text[idx..].chars().next().map(char::len_utf8).unwrap_or(1);
    }
    out.push_str(&text[cursor..]);
    out
}

/// Byte length of a credential token at the start of `s`, when one begins
/// there. Handles `Bearer`/`Basic` schemes and known key prefixes.
fn credential_span_at(s: &str) -> Option<usize> {
    for scheme in ["Bearer ", "Basic "] {
        if let Some(rest) = s.strip_prefix(scheme) {
            let len = token_len(rest);
            if len >= MIN_SECRET_LEN {
                return Some(scheme.len() + len);
            }
        }
    }
    for prefix in TOKEN_PREFIXES {
        if s.starts_with(prefix) {
            let len = token_len(s);
            if len >= prefix.len() + 8 {
                return Some(len);
            }
        }
    }
    if let Some(len) = jwt_span(s) {
        return Some(len);
    }
    None
}

/// Byte length of a leading `eyJ…` JWT triple, when `s` starts with one.
fn jwt_span(s: &str) -> Option<usize> {
    if !s.starts_with("eyJ") {
        return None;
    }
    let len = token_len(s);
    let token = &s[..len];
    if token.matches('.').count() == 2 && len > 20 {
        Some(len)
    } else {
        None
    }
}

/// Byte length of the leading run of credential-safe token characters.
fn token_len(s: &str) -> usize {
    let mut len = 0;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '+' | '/' | '=') {
            len += ch.len_utf8();
        } else {
            break;
        }
    }
    len
}

/// Truncate `text` to [`MAX_FIELD_BYTES`] on a char boundary.
fn truncate(text: &str) -> String {
    if text.len() <= MAX_FIELD_BYTES {
        return text.to_string();
    }
    let mut end = MAX_FIELD_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redactor() -> Redactor {
        Redactor::default()
            .with_secret("KITE_SECRET_VALUE_1234")
            .with_secret("short")
    }

    #[test]
    fn exact_env_values_are_replaced_and_short_values_ignored() {
        let r = redactor();
        assert_eq!(
            r.redact("key=KITE_SECRET_VALUE_1234 done"),
            "key=[REDACTED] done"
        );
        // Below MIN_SECRET_LEN: never treated as a secret.
        assert_eq!(r.redact("a short word"), "a short word");
    }

    #[test]
    fn from_pairs_only_keeps_secret_named_long_values() {
        let r = Redactor::from_pairs(vec![
            (
                "SYLLABIX_LLM_API_KEY".to_string(),
                "sk-abcdef0123456789".to_string(),
            ),
            (
                "KITE_API_SECRET".to_string(),
                "kite-secret-value".to_string(),
            ),
            ("HOME".to_string(), "/Users/example".to_string()),
            ("KITE_EMPTY".to_string(), "tiny".to_string()),
        ]);
        let dumped = format!("{r:?}");
        assert!(dumped.contains("sk-abcdef0123456789"), "{dumped}");
        assert!(dumped.contains("kite-secret-value"), "{dumped}");
        assert!(!dumped.contains("/Users/example"), "{dumped}");
        assert!(!dumped.contains("tiny"), "{dumped}");
    }

    #[test]
    fn secret_names_cover_the_documented_prefixes() {
        for name in [
            "KITE_API_KEY",
            "KITE_API_SECRET",
            "SYLLABIX_LLM_API_KEY",
            "SYLLABIX_STT_API_KEY",
            "DEPLOY_TOKEN",
            "SERVICE_SECRET",
            "DB_PASSWORD",
            "AWS_ACCESS_KEY_ID",
        ] {
            assert!(Redactor::is_secret_name(name), "{name}");
        }
        for name in ["HOME", "PATH", "SYLLABIX_LLM_MODEL", "KITE"] {
            assert!(!Redactor::is_secret_name(name), "{name}");
        }
    }

    #[test]
    fn credential_shapes_are_redacted() {
        let r = Redactor::default();
        // Split the distinctive prefixes from their bodies so the test
        // source never contains a scannable credential literal.
        let samples = [
            format!(
                "Authorization: Bearer {}",
                concat!("sk-", "abcdef0123456789")
            ),
            format!("Basic {}", "dXNlcjpwYXNzd29yZA=="),
            format!("token {}", concat!("sk-proj-", "abcdefghijklmnop")),
            format!("{}abcdefghijklmnopqrstuvwxyz0123456789", "ghp_"),
            format!("{}{}", "xoxb-", "1234567890-abcdefghijklmno"),
            format!("{}{}", "AKIA", "IOSFODNN7EXAMPLE"),
            format!(
                "{}.{}.signaturevalue",
                "eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NSJ9"
            ),
        ];
        for sample in samples {
            let out = r.redact(&sample);
            assert!(out.contains(REDACTED), "not redacted: {sample} -> {out}");
        }
    }

    #[test]
    fn ordinary_text_and_token_counts_survive() {
        let r = Redactor::default();
        let text = "the tokenizer counted inputTokens=42 and tokenUsage=7";
        assert_eq!(r.redact(text), text);
    }

    #[test]
    fn object_values_under_sensitive_keys_are_replaced() {
        let r = Redactor::default();
        let value = json!({
            "url": "https://example.test",
            "api_key": "anything-at-all-secret",
            "nested": { "access_token": "abc", "keep": "visible" },
            "token_count": 3,
            "headers": ["Authorization: Bearer tokvalue1234", "accept: json"],
        });
        let out = r.redact_value(&value);
        assert_eq!(out["api_key"], REDACTED);
        assert_eq!(out["nested"]["access_token"], REDACTED);
        assert_eq!(out["nested"]["keep"], "visible");
        assert_eq!(out["token_count"], 3);
        assert_eq!(out["url"], "https://example.test");
        assert!(!out["headers"][0].as_str().unwrap().contains("tokvalue1234"));
        assert_eq!(out["headers"][1], "accept: json");
    }

    #[test]
    fn entries_render_normalized_json() {
        let r = redactor();
        let user = TrajectoryEntry::User {
            text: "what are my holdings?".into(),
        };
        assert_eq!(
            user.to_json(&r),
            json!({ "kind": "user", "text": "what are my holdings?" })
        );
        let call = TrajectoryEntry::ToolCall {
            name: "shell".into(),
            call_id: "c1".into(),
            arguments: r#"{"command":"ls","api_key":"KITE_SECRET_VALUE_1234"}"#.into(),
        };
        let value = call.to_json(&r);
        assert_eq!(value["kind"], "tool_call");
        assert_eq!(value["arguments"]["command"], "ls");
        assert_eq!(value["arguments"]["api_key"], REDACTED);
        let result = TrajectoryEntry::ToolResult {
            name: "shell".into(),
            call_id: "c1".into(),
            content: "KITE_SECRET_VALUE_1234".into(),
            ok: true,
        };
        let value = result.to_json(&r);
        assert_eq!(value["kind"], "tool_result");
        assert_eq!(value["content"], REDACTED);
        assert_eq!(value["ok"], true);
        // JSON tool output is redacted by key, not just by token shape.
        let json_result = TrajectoryEntry::ToolResult {
            name: "shell".into(),
            call_id: "c1".into(),
            content: r#"{"api_key":"plain-value","keep":"visible"}"#.into(),
            ok: true,
        };
        let content = json_result.to_json(&r)["content"]
            .as_str()
            .expect("content")
            .to_string();
        assert!(content.contains(REDACTED), "{content}");
        assert!(!content.contains("plain-value"), "{content}");
        assert!(content.contains("visible"), "{content}");
        let outcome = TrajectoryEntry::Outcome {
            outcome: "completed".into(),
        };
        assert_eq!(
            outcome.to_json(&r),
            json!({ "kind": "outcome", "outcome": "completed" })
        );
    }

    #[test]
    fn long_fields_are_truncated_on_a_char_boundary() {
        let r = Redactor::default();
        let long = "é".repeat(MAX_FIELD_BYTES);
        let entry = TrajectoryEntry::Assistant { text: long };
        let value = entry.to_json(&r);
        let text = value["text"].as_str().expect("text");
        assert!(text.len() <= MAX_FIELD_BYTES);
        assert!(text.chars().all(|c| c == 'é'));
    }

    #[test]
    fn non_json_arguments_are_kept_as_a_string() {
        let r = Redactor::default();
        let entry = TrajectoryEntry::ToolCall {
            name: "shell".into(),
            call_id: "c1".into(),
            arguments: "not json".into(),
        };
        assert_eq!(entry.to_json(&r)["arguments"], "not json");
    }
}
