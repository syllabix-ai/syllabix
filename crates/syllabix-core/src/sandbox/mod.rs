//! Capability sandbox providers for host-owned developer-harness commands.
//!
//! The provider gives the generic shell and future skill entrypoints one seam
//! for asking the host to enforce a policy. Developer-harness shell calls fail
//! closed when the selected provider cannot uphold the requested mode.
//!
//! Platform backends:
//! - macOS: Seatbelt via `sandbox-exec`
//! - Linux: Bubblewrap primary, Landlock launcher fallback when the kernel ABI
//!   probe succeeds
//! - other: fail closed

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::policy::{Enforcement, FilesystemMode, NetworkMode};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// Inputs needed to construct an OS sandbox for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxRequest {
    /// Canonical project root. Workspace-write is confined to this directory.
    pub workspace: PathBuf,
    /// Fresh, private temporary directory owned by this invocation.
    pub temp_dir: PathBuf,
    /// Filesystem capability requested by the invocation.
    pub filesystem: FilesystemMode,
    /// Network policy. Denial is enforced when the selected backend can uphold it.
    pub network: NetworkMode,
}

impl SandboxRequest {
    pub fn new(
        workspace: impl Into<PathBuf>,
        temp_dir: impl Into<PathBuf>,
        filesystem: FilesystemMode,
        network: NetworkMode,
    ) -> Self {
        Self {
            workspace: workspace.into(),
            temp_dir: temp_dir.into(),
            filesystem,
            network,
        }
    }
}

/// A host error from preparing or probing a sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    pub code: &'static str,
    pub message: String,
}

impl SandboxError {
    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: "SANDBOX_UNAVAILABLE",
            message: message.into(),
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "SANDBOX_INVALID_REQUEST",
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for SandboxError {}

/// One platform implementation of filesystem sandboxing.
pub trait SandboxProvider: Send + Sync {
    /// Run a harmless, functional probe. A successful probe is the only basis
    /// for reporting [`Enforcement::Full`] or [`Enforcement::Partial`].
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError>;

    /// Wrap a program and argv in the provider's confinement.
    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError>;
}

/// Provider selected for the current host. Unsupported hosts fail closed.
pub fn current_provider() -> Box<dyn SandboxProvider> {
    #[cfg(target_os = "macos")]
    {
        Box::new(macos::MacOsSandboxProvider)
    }
    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LinuxSandboxProvider)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Box::new(UnavailableSandboxProvider)
    }
}

/// Explicit fail-closed provider for hosts without a sandbox backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableSandboxProvider;

impl SandboxProvider for UnavailableSandboxProvider {
    fn probe(&self, _: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        Err(SandboxError::unavailable(
            "no sandbox provider is available on this platform",
        ))
    }

    fn command(&self, _: &SandboxRequest, _: &Path, _: &[String]) -> Result<Command, SandboxError> {
        Err(SandboxError::unavailable(
            "no sandbox provider is available on this platform",
        ))
    }
}

#[cfg(target_os = "macos")]
pub use macos::MacOsSandboxProvider;

#[cfg(target_os = "linux")]
pub use linux::{
    BubblewrapSandboxProvider, LandlockSandboxProvider, LinuxSandboxProvider, LinuxSandboxRunner,
};

pub(crate) fn validate_request(request: &SandboxRequest) -> Result<(), SandboxError> {
    if !request.workspace.is_absolute() || !request.temp_dir.is_absolute() {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be absolute paths",
        ));
    }
    if request.workspace == Path::new("/") {
        return Err(SandboxError::invalid(
            "workspace cannot be the filesystem root",
        ));
    }
    if request.filesystem == FilesystemMode::DangerFullAccess {
        return Err(SandboxError::unavailable(
            "danger-full-access has no sandbox provider",
        ));
    }
    if !request.workspace.is_dir() || !request.temp_dir.is_dir() {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be existing directories",
        ));
    }
    if request.workspace.canonicalize().ok().as_deref() != Some(request.workspace.as_path())
        || request.temp_dir.canonicalize().ok().as_deref() != Some(request.temp_dir.as_path())
    {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be canonical directories",
        ));
    }
    Ok(())
}

/// Host path used to prove outside-workspace writes are denied.
pub(crate) fn outside_probe_path(request: &SandboxRequest) -> Result<PathBuf, SandboxError> {
    // Prefer a sibling of the workspace when that parent is not shadowed by a
    // sandbox remount (for example bwrap `--tmpfs /tmp`). `/var/tmp` stays on
    // the real host filesystem for Linux probes.
    #[cfg(target_os = "linux")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static OUTSIDE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = OUTSIDE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let _ = request;
        Ok(PathBuf::from("/var/tmp").join(format!(
            ".syllabix-sandbox-outside-probe-{}-{}",
            std::process::id(),
            n
        )))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(request
            .workspace
            .parent()
            .ok_or_else(|| SandboxError::invalid("workspace has no parent"))?
            .join(".syllabix-sandbox-outside-probe"))
    }
}

pub(crate) fn shell_quote_single(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "'\\''")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: FilesystemMode) -> SandboxRequest {
        #[cfg(target_os = "macos")]
        {
            SandboxRequest::new(
                "/private/tmp/syllabix-workspace",
                "/private/tmp/syllabix-temp",
                mode,
                NetworkMode::None,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            SandboxRequest::new(
                "/tmp/syllabix-workspace",
                "/tmp/syllabix-temp",
                mode,
                NetworkMode::None,
            )
        }
    }

    #[test]
    fn unsupported_platform_provider_fails_closed() {
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        assert_eq!(
            current_provider()
                .probe(&request(FilesystemMode::ReadOnly))
                .unwrap_err()
                .code,
            "SANDBOX_UNAVAILABLE"
        );
    }

    #[test]
    fn danger_full_access_is_not_a_sandbox_profile() {
        let err = validate_request(&request(FilesystemMode::DangerFullAccess)).unwrap_err();
        assert_eq!(err.code, "SANDBOX_UNAVAILABLE");
    }

    #[test]
    fn relative_paths_are_rejected() {
        let mut request = request(FilesystemMode::ReadOnly);
        request.workspace = PathBuf::from("workspace");
        assert_eq!(
            validate_request(&request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_provider_prefers_bubblewrap_when_available() {
        // Skip when a test harness forces a backend via the environment.
        if std::env::var_os("SYLLABIX_SANDBOX_BACKEND").is_some() {
            return;
        }
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let root = std::env::temp_dir().join(format!(
            "syllabix-sandbox-select-{}-{}",
            std::process::id(),
            n
        ));
        let workspace = root.join("workspace");
        let temp = root.join("temp");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&temp).unwrap();
        let workspace = workspace.canonicalize().unwrap();
        let temp = temp.canonicalize().unwrap();
        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::ReadOnly,
            NetworkMode::None,
        );
        let (runner, _) = linux::select_runner(&request).expect("linux runner");
        if Path::new("/usr/bin/bwrap").exists() || Path::new("/bin/bwrap").exists() {
            assert_eq!(runner, LinuxSandboxRunner::Bubblewrap);
        } else {
            assert_eq!(runner, LinuxSandboxRunner::Landlock);
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
