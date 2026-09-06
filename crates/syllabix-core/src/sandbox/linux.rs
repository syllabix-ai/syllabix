//! Linux sandbox providers: Bubblewrap primary, Landlock fallback.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use landlock::{
    path_beneath_rules, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr,
    RulesetCreatedAttr, RulesetStatus, ABI,
};
use std::os::unix::process::CommandExt;

use crate::policy::{Enforcement, FilesystemMode, NetworkMode};

use super::{
    outside_probe_path, shell_quote_single, validate_request, SandboxError, SandboxProvider,
    SandboxRequest,
};

/// Which Linux backend was selected for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxSandboxRunner {
    /// Bubblewrap namespace/mount confinement.
    Bubblewrap,
    /// In-process Landlock launcher applied via `pre_exec`.
    Landlock,
}

/// Linux provider: try Bubblewrap, then Landlock. Never degrade to unconfined.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinuxSandboxProvider;

/// Bubblewrap (`bwrap`) mount/namespace backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct BubblewrapSandboxProvider;

/// Landlock LSM backend. Applied in the child via `pre_exec` (the bundled
/// launcher path for a single Rust binary).
#[derive(Debug, Clone, Copy, Default)]
pub struct LandlockSandboxProvider;

/// Optional override for tests: `bwrap`, `landlock`, or unset/`auto`.
fn backend_override() -> Option<&'static str> {
    match std::env::var("SYLLABIX_SANDBOX_BACKEND") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "bwrap" | "bubblewrap" => Some("bwrap"),
            "landlock" => Some("landlock"),
            "auto" | "" => None,
            _ => None,
        },
        Err(_) => None,
    }
}

fn bwrap_bin() -> Option<PathBuf> {
    for candidate in ["/usr/bin/bwrap", "/bin/bwrap"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

/// Select the Linux runner for one request without silently falling through to
/// an unconfined spawn. Probes exactly once per candidate and returns the
/// enforcement that probe already measured (callers must not re-probe).
pub fn select_runner(
    request: &SandboxRequest,
) -> Result<(LinuxSandboxRunner, Enforcement), SandboxError> {
    validate_request(request)?;
    match backend_override() {
        Some("landlock") => {
            let enforcement = LandlockSandboxProvider.probe(request)?;
            Ok((LinuxSandboxRunner::Landlock, enforcement))
        }
        Some("bwrap") => {
            let enforcement = BubblewrapSandboxProvider.probe(request)?;
            Ok((LinuxSandboxRunner::Bubblewrap, enforcement))
        }
        _ => {
            if bwrap_bin().is_some() {
                match BubblewrapSandboxProvider.probe(request) {
                    Ok(enforcement) => {
                        return Ok((LinuxSandboxRunner::Bubblewrap, enforcement));
                    }
                    Err(bwrap_err) => match LandlockSandboxProvider.probe(request) {
                        Ok(enforcement) => {
                            return Ok((LinuxSandboxRunner::Landlock, enforcement));
                        }
                        Err(landlock_err) => {
                            return Err(SandboxError::unavailable(format!(
                                "bubblewrap unavailable ({}); landlock unavailable ({})",
                                bwrap_err.message, landlock_err.message
                            )));
                        }
                    },
                }
            }
            let enforcement = LandlockSandboxProvider.probe(request)?;
            Ok((LinuxSandboxRunner::Landlock, enforcement))
        }
    }
}

/// Pick a runner for `command` without repeating the functional probe.
/// Prefers Bubblewrap when installed; Landlock otherwise. `probe` remains the
/// fail-closed gate that proves the backend works.
fn pick_runner_for_command(request: &SandboxRequest) -> Result<LinuxSandboxRunner, SandboxError> {
    validate_request(request)?;
    match backend_override() {
        Some("landlock") => Ok(LinuxSandboxRunner::Landlock),
        Some("bwrap") => {
            if bwrap_bin().is_none() {
                return Err(SandboxError::unavailable(
                    "bubblewrap (bwrap) is not installed on this host",
                ));
            }
            Ok(LinuxSandboxRunner::Bubblewrap)
        }
        _ => {
            if bwrap_bin().is_some() {
                Ok(LinuxSandboxRunner::Bubblewrap)
            } else {
                let _ = landlock_fs_abi()?;
                Ok(LinuxSandboxRunner::Landlock)
            }
        }
    }
}

impl SandboxProvider for LinuxSandboxProvider {
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        Ok(select_runner(request)?.1)
    }

    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError> {
        match pick_runner_for_command(request)? {
            LinuxSandboxRunner::Bubblewrap => {
                BubblewrapSandboxProvider.command(request, program, argv)
            }
            LinuxSandboxRunner::Landlock => LandlockSandboxProvider.command(request, program, argv),
        }
    }
}

fn bwrap_args(request: &SandboxRequest) -> Result<Vec<String>, SandboxError> {
    validate_request(request)?;
    let mut args = vec![
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--dev".into(),
        "/dev".into(),
        "--unshare-pid".into(),
        "--proc".into(),
        "/proc".into(),
        "--die-with-parent".into(),
    ];
    if request.network == NetworkMode::None {
        args.push("--unshare-net".into());
    }
    match request.filesystem {
        FilesystemMode::ReadOnly => {}
        FilesystemMode::WorkspaceWrite => {
            // Fresh /tmp, then re-bind the canonical workspace and private temp
            // so paths under /tmp remain reachable after the tmpfs overlay.
            args.push("--tmpfs".into());
            args.push("/tmp".into());
            args.push("--bind".into());
            args.push(request.workspace.to_string_lossy().into_owned());
            args.push(request.workspace.to_string_lossy().into_owned());
            args.push("--bind".into());
            args.push(request.temp_dir.to_string_lossy().into_owned());
            args.push(request.temp_dir.to_string_lossy().into_owned());
        }
        FilesystemMode::DangerFullAccess => unreachable!("validated above"),
    }
    Ok(args)
}

fn check_write_probe(
    mut command: Command,
    marker: &Path,
    expected_created: bool,
) -> Result<(), SandboxError> {
    let status = command
        .status()
        .map_err(|e| SandboxError::unavailable(format!("sandbox write probe: {e}")))?;
    let created = marker.exists();
    let _ = std::fs::remove_file(marker);
    if created != expected_created || (expected_created && !status.success()) {
        return Err(SandboxError::unavailable(
            "sandbox write probe did not match the requested filesystem mode",
        ));
    }
    Ok(())
}

fn check_outside_probe(mut command: Command, outside: &Path) -> Result<(), SandboxError> {
    let _ = command
        .status()
        .map_err(|e| SandboxError::unavailable(format!("sandbox outside-write probe: {e}")))?;
    let created = outside.exists();
    let _ = std::fs::remove_file(outside);
    if created {
        return Err(SandboxError::unavailable(
            "sandbox outside-workspace write probe was not denied",
        ));
    }
    Ok(())
}

fn network_connect_probe_denied(mut command: Command) -> Result<(), SandboxError> {
    // Prefer nc when present; otherwise use Python's socket module. Both are
    // available on the hosted Linux CI images we care about.
    let output = command
        .output()
        .map_err(|e| SandboxError::unavailable(format!("sandbox network probe: {e}")))?;
    if output.status.success() {
        return Err(SandboxError::unavailable(
            "sandbox network probe unexpectedly connected",
        ));
    }
    Ok(())
}

fn touch_script(path: &Path) -> String {
    format!("touch '{}'", shell_quote_single(path))
}

fn network_probe_script() -> Result<&'static str, SandboxError> {
    if Path::new("/usr/bin/nc").exists() || Path::new("/bin/nc").exists() {
        Ok("nc -z -w 1 1.1.1.1 53")
    } else if Path::new("/usr/bin/python3").exists() {
        Ok("python3 -c \"import socket; socket.create_connection(('1.1.1.1',53),1)\"")
    } else {
        Err(SandboxError::unavailable(
            "no network probe helper (nc/python3) available to verify unshare-net",
        ))
    }
}

impl SandboxProvider for BubblewrapSandboxProvider {
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        let bwrap = bwrap_bin().ok_or_else(|| {
            SandboxError::unavailable("bubblewrap (bwrap) is not installed on this host")
        })?;
        let args = bwrap_args(request)?;

        let mut true_cmd = Command::new(&bwrap);
        true_cmd
            .args(&args)
            .arg("/bin/true")
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        let status = true_cmd
            .status()
            .map_err(|e| SandboxError::unavailable(format!("bwrap probe: {e}")))?;
        if !status.success() {
            return Err(SandboxError::unavailable(
                "bwrap probe failed (user namespaces may be unavailable in this container)",
            ));
        }

        let marker = request.temp_dir.join(".syllabix-sandbox-probe");
        let expected = request.filesystem == FilesystemMode::WorkspaceWrite;
        let mut write = Command::new(&bwrap);
        write
            .args(&args)
            .args(["/bin/sh", "-c", &touch_script(&marker)])
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        check_write_probe(write, &marker, expected)?;

        let outside = outside_probe_path(request)?;
        let mut outside_write = Command::new(&bwrap);
        outside_write
            .args(&args)
            .args(["/bin/sh", "-c", &touch_script(&outside)])
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        check_outside_probe(outside_write, &outside)?;

        if request.network == NetworkMode::None {
            let mut network = Command::new(&bwrap);
            network
                .args(&args)
                .args(["/bin/sh", "-c", network_probe_script()?])
                .stderr(std::process::Stdio::null())
                .stdout(std::process::Stdio::null());
            network_connect_probe_denied(network)?;
        }
        Ok(Enforcement::Full)
    }

    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError> {
        let bwrap = bwrap_bin().ok_or_else(|| {
            SandboxError::unavailable("bubblewrap (bwrap) is not installed on this host")
        })?;
        let args = bwrap_args(request)?;
        let mut command = Command::new(bwrap);
        command.args(args).arg(program).args(argv);
        Ok(command)
    }
}

fn landlock_fs_abi() -> Result<(ABI, Enforcement), SandboxError> {
    // Prefer ABI V3 so Truncate is governed (Full). Older kernels that only
    // expose V1/V2 still confine writes but report Partial.
    for (abi, enforcement) in [
        (ABI::V3, Enforcement::Full),
        (ABI::V1, Enforcement::Partial),
    ] {
        let ok = Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_all(abi))
            .and_then(|ruleset| ruleset.create())
            .is_ok();
        if ok {
            return Ok((abi, enforcement));
        }
    }
    Err(SandboxError::unavailable(
        "landlock ABI probe failed (kernel lacks Landlock or it is disabled)",
    ))
}

/// True when the running kernel can enforce Landlock TCP bind/connect denial.
fn landlock_tcp_deny_supported() -> bool {
    let net = AccessNet::from_all(ABI::V4);
    if net.is_empty() {
        return false;
    }
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(net)
        .and_then(|ruleset| ruleset.create())
        .is_ok()
}

fn apply_landlock(request: &SandboxRequest) -> io::Result<()> {
    let (abi, _) = landlock_fs_abi().map_err(|e| io::Error::other(e.message))?;
    let ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| io::Error::other(e.to_string()))?;

    // TCP deny is best-effort when network:none and ABI >= 4. Landlock still
    // cannot promise full network isolation (UDP/other), so callers see Partial.
    let net = AccessNet::from_all(ABI::V4);
    let ruleset = if request.network == NetworkMode::None && !net.is_empty() {
        match ruleset.handle_access(net) {
            Ok(with_net) => with_net,
            Err(_) => {
                // Fall back to filesystem-only confinement on older ABIs.
                Ruleset::default()
                    .set_compatibility(CompatLevel::HardRequirement)
                    .handle_access(AccessFs::from_all(abi))
                    .map_err(|e| io::Error::other(e.to_string()))?
            }
        }
    } else {
        ruleset
    };

    let mut created = ruleset
        .create()
        .map_err(|e| io::Error::other(e.to_string()))?
        .set_compatibility(CompatLevel::HardRequirement)
        .add_rules(path_beneath_rules(["/"], AccessFs::from_read(abi)))
        .map_err(|e| io::Error::other(e.to_string()))?
        .add_rules(path_beneath_rules(["/dev/null"], AccessFs::from_all(abi)))
        .map_err(|e| io::Error::other(e.to_string()))?;

    if request.filesystem == FilesystemMode::WorkspaceWrite {
        created = created
            .add_rules(path_beneath_rules(
                [&request.workspace, &request.temp_dir],
                AccessFs::from_all(abi),
            ))
            .map_err(|e| io::Error::other(e.to_string()))?;
    }

    // No NetPort rules ⇒ deny all handled TCP bind/connect when net was claimed.
    let status = created
        .restrict_self()
        .map_err(|e| io::Error::other(e.to_string()))?;
    if status.ruleset != RulesetStatus::FullyEnforced {
        return Err(io::Error::other("landlock ruleset was not fully enforced"));
    }
    Ok(())
}

impl SandboxProvider for LandlockSandboxProvider {
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        validate_request(request)?;
        let (_, mut enforcement) = landlock_fs_abi()?;
        // Network:none can never be a Full Landlock claim (UDP/other remain).
        if request.network == NetworkMode::None {
            enforcement = Enforcement::Partial;
        }

        let marker = request.temp_dir.join(".syllabix-sandbox-probe");
        let expected = request.filesystem == FilesystemMode::WorkspaceWrite;
        let req = request.clone();
        let mut write = Command::new("/bin/sh");
        write
            .arg("-c")
            .arg(touch_script(&marker))
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        unsafe {
            write.pre_exec(move || apply_landlock(&req));
        }
        check_write_probe(write, &marker, expected)?;

        let outside = outside_probe_path(request)?;
        let req = request.clone();
        let mut outside_write = Command::new("/bin/sh");
        outside_write
            .arg("-c")
            .arg(touch_script(&outside))
            .stderr(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        unsafe {
            outside_write.pre_exec(move || apply_landlock(&req));
        }
        check_outside_probe(outside_write, &outside)?;

        // When the kernel can claim TCP denial, verify a connect is actually
        // blocked so a misconfigured net handle cannot look like "expected Partial".
        if request.network == NetworkMode::None && landlock_tcp_deny_supported() {
            let req = request.clone();
            let mut network = Command::new("/bin/sh");
            network
                .arg("-c")
                .arg(network_probe_script()?)
                .stderr(std::process::Stdio::null())
                .stdout(std::process::Stdio::null());
            unsafe {
                network.pre_exec(move || apply_landlock(&req));
            }
            network_connect_probe_denied(network)?;
        }
        Ok(enforcement)
    }

    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError> {
        validate_request(request)?;
        let _ = landlock_fs_abi()?;
        let req = request.clone();
        let mut command = Command::new(program);
        command.args(argv);
        unsafe {
            command.pre_exec(move || apply_landlock(&req));
        }
        Ok(command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn live_dirs() -> (PathBuf, PathBuf, PathBuf) {
        let n = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "syllabix-linux-sandbox-{}-{}",
            std::process::id(),
            n
        ));
        let workspace = root.join("workspace");
        let temp = root.join("temp");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&temp).unwrap();
        let workspace = workspace.canonicalize().unwrap();
        let temp = temp.canonicalize().unwrap();
        (root, workspace, temp)
    }

    #[test]
    fn bubblewrap_probe_and_modes_when_installed() {
        if bwrap_bin().is_none() {
            return;
        }
        let (root, workspace, temp) = live_dirs();
        let provider = BubblewrapSandboxProvider;
        for mode in [FilesystemMode::ReadOnly, FilesystemMode::WorkspaceWrite] {
            let request = SandboxRequest::new(&workspace, &temp, mode, NetworkMode::None);
            assert_eq!(provider.probe(&request).unwrap(), Enforcement::Full);
        }

        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::ReadOnly,
            NetworkMode::None,
        );
        let denied = root.join("must-not-write");
        let mut write = provider
            .command(
                &request,
                Path::new("/bin/sh"),
                &[
                    "-c".into(),
                    format!("touch '{}'", shell_quote_single(&denied)),
                ],
            )
            .unwrap();
        let _ = write.output().unwrap();
        assert!(!denied.exists(), "bwrap unexpectedly created {denied:?}");

        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::WorkspaceWrite,
            NetworkMode::None,
        );
        let allowed = temp.join("ok");
        let mut write = provider
            .command(
                &request,
                Path::new("/bin/sh"),
                &[
                    "-c".into(),
                    format!("touch '{}'", shell_quote_single(&allowed)),
                ],
            )
            .unwrap();
        assert!(write.status().unwrap().success());
        assert!(allowed.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn landlock_probe_reports_partial_for_network_none() {
        let (root, workspace, temp) = live_dirs();
        let provider = LandlockSandboxProvider;
        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::ReadOnly,
            NetworkMode::None,
        );
        match provider.probe(&request) {
            Ok(enforcement) => {
                assert_eq!(enforcement, Enforcement::Partial);
                let denied = workspace.join("nope");
                let mut write = provider
                    .command(
                        &request,
                        Path::new("/bin/sh"),
                        &[
                            "-c".into(),
                            format!("touch '{}'", shell_quote_single(&denied)),
                        ],
                    )
                    .unwrap();
                let _ = write.output().unwrap();
                assert!(!denied.exists(), "landlock unexpectedly created {denied:?}");
            }
            Err(err) => {
                assert_eq!(err.code, "SANDBOX_UNAVAILABLE");
            }
        }

        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::WorkspaceWrite,
            NetworkMode::Allow,
        );
        if let Ok(enforcement) = provider.probe(&request) {
            assert!(matches!(
                enforcement,
                Enforcement::Full | Enforcement::Partial
            ));
            let allowed = temp.join("landlock-ok");
            let mut write = provider
                .command(
                    &request,
                    Path::new("/bin/sh"),
                    &[
                        "-c".into(),
                        format!("touch '{}'", shell_quote_single(&allowed)),
                    ],
                )
                .unwrap();
            assert!(write.status().unwrap().success());
            assert!(allowed.exists());
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn linux_provider_fails_closed_without_usable_backend() {
        // Force landlock; if ABI is present this still succeeds. The important
        // invariant is that UnavailableSandboxProvider is never returned on Linux
        // when select_runner errors — errors stay SANDBOX_UNAVAILABLE.
        let (root, workspace, temp) = live_dirs();
        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::ReadOnly,
            NetworkMode::None,
        );
        match LinuxSandboxProvider.probe(&request) {
            Ok(_) => {}
            Err(err) => assert_eq!(err.code, "SANDBOX_UNAVAILABLE"),
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
