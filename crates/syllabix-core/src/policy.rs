//! Session capability policy for the developer harness.
//!
//! This module is the typed seam for capability-sandboxed execution
//! ([#111](https://github.com/syllabix-ai/syllabix/issues/111)). The configured
//! session permissions are a ceiling: a call or skill may narrow them, never
//! widen them. The current allowlist executor does not yet consume
//! [`ExecutionPlan`]; that wiring lands with the generic shell.

use std::ffi::OsString;
use std::path::PathBuf;
use std::time::Duration;

/// Filesystem effect mode for a sandboxed child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FilesystemMode {
    /// Workspace and host reads; writes denied.
    ReadOnly,
    /// Writes under the workspace root plus a private temp directory.
    WorkspaceWrite,
    /// Unconfined filesystem access.
    DangerFullAccess,
}

impl FilesystemMode {
    /// Yaml / denial token (`read-only`, `workspace-write`, `danger-full-access`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::DangerFullAccess => "danger-full-access",
        }
    }
}

/// Network policy for a sandboxed child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NetworkMode {
    /// No network access promised / requested.
    None,
    /// Network access allowed.
    Allow,
}

impl NetworkMode {
    /// Yaml / denial token (`none`, `allow`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Allow => "allow",
        }
    }
}

/// Secret injection policy. Only `none` is supported in this seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecretPolicy {
    /// Scrubbed environment; no host credentials injected.
    None,
}

impl SecretPolicy {
    /// Yaml / denial token (`none`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
        }
    }
}

/// How completely a platform provider can uphold the requested confinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // consumed when sandbox providers report enforcement
pub enum Enforcement {
    /// Full enforcement of the requested mode on this platform.
    Full,
    /// Partial enforcement (for example Windows write restriction only).
    Partial,
}

/// Why this plan was built: generic shell tool or a trusted skill entrypoint.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // consumed when generic-shell and skill adapters land
pub enum Provenance {
    /// Model-facing generic shell tool.
    Shell,
    /// Trusted skill entrypoint (`id` is the skill name).
    #[allow(dead_code)] // skill adapter lands after the generic shell
    Skill { id: String },
}

/// Session (or per-call) developer permission ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeveloperPermissions {
    /// Filesystem effect ceiling.
    pub filesystem: FilesystemMode,
    /// Network ceiling.
    pub network: NetworkMode,
    /// Secret-injection ceiling.
    pub secrets: SecretPolicy,
}

impl DeveloperPermissions {
    /// Default session ceiling: read-only filesystem, no network, no secrets.
    pub fn default_session() -> Self {
        Self {
            filesystem: FilesystemMode::ReadOnly,
            network: NetworkMode::None,
            secrets: SecretPolicy::None,
        }
    }
}

impl Default for DeveloperPermissions {
    fn default() -> Self {
        Self::default_session()
    }
}

/// Host-owned plan for one sandboxed child. Built by future shell/skill
/// adapters; unused by the current allowlist path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // constructed when generic-shell and skill adapters land
pub struct ExecutionPlan {
    /// Absolute or PATH-resolved program.
    pub program: PathBuf,
    /// Full argv including `program` as argv0 when appropriate.
    pub argv: Vec<OsString>,
    /// Canonical working directory for the child.
    pub cwd: PathBuf,
    /// Effective filesystem mode after policy resolution.
    pub filesystem: FilesystemMode,
    /// Effective network mode after policy resolution.
    pub network: NetworkMode,
    /// Effective secret policy after policy resolution.
    pub secrets: SecretPolicy,
    /// Wall-clock timeout for the child.
    pub timeout: Duration,
    /// Combined stdout/stderr capture budget.
    pub max_output_bytes: usize,
    /// Shell tool vs skill entrypoint.
    pub provenance: Provenance,
}

/// Structured denial when a request would widen the session ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // returned by resolve_effective; call sites land with generic shell
pub struct PolicyDeny {
    /// Human-readable crossed-boundary message.
    pub message: String,
}

impl std::fmt::Display for PolicyDeny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PolicyDeny {}

/// Resolve the effective permissions for one call.
///
/// `requested` may only narrow `session`. Widening any axis returns
/// [`PolicyDeny`] naming the crossed boundary.
#[allow(dead_code)] // wired through executor plan builder in a follow-up
pub fn resolve_effective(
    session: &DeveloperPermissions,
    requested: Option<&DeveloperPermissions>,
) -> Result<DeveloperPermissions, PolicyDeny> {
    let Some(requested) = requested else {
        return Ok(session.clone());
    };

    if requested.filesystem > session.filesystem {
        return Err(PolicyDeny {
            message: format!(
                "{} required; configured filesystem mode is {}",
                requested.filesystem.as_str(),
                session.filesystem.as_str(),
            ),
        });
    }
    if requested.network > session.network {
        return Err(PolicyDeny {
            message: format!(
                "{} required; configured network mode is {}",
                requested.network.as_str(),
                session.network.as_str(),
            ),
        });
    }
    if requested.secrets > session.secrets {
        return Err(PolicyDeny {
            message: format!(
                "{} required; configured secrets policy is {}",
                requested.secrets.as_str(),
                session.secrets.as_str(),
            ),
        });
    }

    Ok(requested.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filesystem_mode_orders_read_only_below_write_below_danger() {
        assert!(FilesystemMode::ReadOnly < FilesystemMode::WorkspaceWrite);
        assert!(FilesystemMode::WorkspaceWrite < FilesystemMode::DangerFullAccess);
        assert!(FilesystemMode::ReadOnly < FilesystemMode::DangerFullAccess);
    }

    #[test]
    fn network_mode_orders_none_below_allow() {
        assert!(NetworkMode::None < NetworkMode::Allow);
    }

    #[test]
    fn resolve_effective_returns_session_when_no_request() {
        let session = DeveloperPermissions {
            filesystem: FilesystemMode::WorkspaceWrite,
            network: NetworkMode::Allow,
            secrets: SecretPolicy::None,
        };
        assert_eq!(resolve_effective(&session, None).unwrap(), session);
    }

    #[test]
    fn resolve_effective_allows_narrowing() {
        let session = DeveloperPermissions {
            filesystem: FilesystemMode::WorkspaceWrite,
            network: NetworkMode::Allow,
            secrets: SecretPolicy::None,
        };
        let requested = DeveloperPermissions::default_session();
        assert_eq!(
            resolve_effective(&session, Some(&requested)).unwrap(),
            requested
        );
    }

    #[test]
    fn resolve_effective_allows_equal_request() {
        let session = DeveloperPermissions::default_session();
        assert_eq!(
            resolve_effective(&session, Some(&session)).unwrap(),
            session
        );
    }

    #[test]
    fn resolve_effective_denies_filesystem_widening() {
        let session = DeveloperPermissions::default_session();
        let requested = DeveloperPermissions {
            filesystem: FilesystemMode::WorkspaceWrite,
            network: NetworkMode::None,
            secrets: SecretPolicy::None,
        };
        let err = resolve_effective(&session, Some(&requested)).unwrap_err();
        assert_eq!(
            err.message,
            "workspace-write required; configured filesystem mode is read-only"
        );
    }

    #[test]
    fn resolve_effective_denies_network_widening() {
        let session = DeveloperPermissions {
            filesystem: FilesystemMode::WorkspaceWrite,
            network: NetworkMode::None,
            secrets: SecretPolicy::None,
        };
        let requested = DeveloperPermissions {
            filesystem: FilesystemMode::ReadOnly,
            network: NetworkMode::Allow,
            secrets: SecretPolicy::None,
        };
        let err = resolve_effective(&session, Some(&requested)).unwrap_err();
        assert_eq!(
            err.message,
            "allow required; configured network mode is none"
        );
    }

    #[test]
    fn resolve_effective_denies_danger_widening_from_workspace_write() {
        let session = DeveloperPermissions {
            filesystem: FilesystemMode::WorkspaceWrite,
            network: NetworkMode::None,
            secrets: SecretPolicy::None,
        };
        let requested = DeveloperPermissions {
            filesystem: FilesystemMode::DangerFullAccess,
            ..session.clone()
        };
        let err = resolve_effective(&session, Some(&requested)).unwrap_err();
        assert_eq!(
            err.message,
            "danger-full-access required; configured filesystem mode is workspace-write"
        );
    }
}
