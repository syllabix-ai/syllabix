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

#[cfg(all(target_os = "linux", not(coverage)))]
mod linux;
#[cfg(all(target_os = "macos", not(coverage)))]
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
    #[cfg(all(target_os = "macos", not(coverage)))]
    {
        Box::new(macos::MacOsSandboxProvider)
    }
    #[cfg(all(target_os = "macos", coverage))]
    {
        Box::new(UnavailableSandboxProvider)
    }
    #[cfg(all(target_os = "linux", not(coverage)))]
    {
        Box::new(linux::LinuxSandboxProvider)
    }
    #[cfg(all(target_os = "linux", coverage))]
    {
        Box::new(UnavailableSandboxProvider)
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

#[cfg(all(target_os = "macos", not(coverage)))]
pub use macos::MacOsSandboxProvider;

#[cfg(all(target_os = "linux", not(coverage)))]
pub use linux::{
    BubblewrapSandboxProvider, LandlockSandboxProvider, LinuxSandboxProvider, LinuxSandboxRunner,
};

#[cfg(test)]
mod tests {
    #[cfg(coverage)]
    use super::*;

    #[cfg(all(target_os = "linux", not(coverage)))]
    use super::{linux, FilesystemMode, LinuxSandboxRunner, NetworkMode, SandboxRequest};
    #[cfg(all(target_os = "linux", not(coverage)))]
    use std::path::Path;

    #[cfg(coverage)]
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

    #[cfg(coverage)]
    #[test]
    fn coverage_provider_fails_closed() {
        let provider = current_provider();
        let request = request(FilesystemMode::ReadOnly);
        assert_eq!(
            provider.probe(&request).unwrap_err().code,
            "SANDBOX_UNAVAILABLE"
        );
        assert_eq!(
            provider
                .command(&request, Path::new("/bin/true"), &[])
                .unwrap_err()
                .code,
            "SANDBOX_UNAVAILABLE"
        );
    }

    #[cfg(all(target_os = "linux", not(coverage)))]
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
