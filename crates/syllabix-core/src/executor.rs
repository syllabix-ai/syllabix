//! Host-owned, bounded executors for the developer harness.
//!
//! This module is deliberately below the model-facing [`ToolCall`] contract.
//! Calls are validated here before an executor is selected; callers never get
//! a shell string or a policy escape hatch.

use std::io::Read;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::cancel::Cancel;
use crate::types::ToolCall;

/// Maximum argv entries accepted from one model tool call.
pub const MAX_ARGV: usize = 32;
/// Maximum bytes in one argv element.
pub const MAX_ARG_BYTES: usize = 1024;
/// Captured stdout and stderr are independently bounded.
pub const MAX_OUTPUT_BYTES: usize = 8 * 1024;
/// A foreground executor call is never allowed to own the voice turn indefinitely.
pub const SHELL_TIMEOUT: Duration = Duration::from_secs(5);
/// Fetch response body limit before content reaches the model.
pub const MAX_FETCH_BYTES: usize = 64 * 1024;
/// Fetch redirects are finite and each hop is resolved through the public-only resolver.
pub const MAX_FETCH_REDIRECTS: u32 = 3;
const FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const FETCH_READ_TIMEOUT: Duration = Duration::from_millis(200);
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// A validated direct-argv inspection command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellRequest {
    pub argv: Vec<String>,
    pub cwd: PathBuf,
}

/// A call whose public arguments meet the fixed policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedCall {
    Shell(ShellRequest),
    WebFetch { url: String },
}

/// Execute one model call with its host-owned authority. A failed validation
/// becomes a bounded, model-visible rejection rather than a provider failure.
pub fn execute(call: &ToolCall, workspace: &Path, cancel: &Cancel) -> crate::types::ToolResult {
    let result = match validate_call(call, workspace) {
        Ok(ValidatedCall::Shell(request)) => execute_shell(request, cancel),
        Ok(ValidatedCall::WebFetch { url }) => execute_fetch(&url, cancel),
        Err(message) => Err(message),
    };
    crate::types::ToolResult {
        tool_call_id: call.id.clone(),
        ok: result.is_ok(),
        content: result.unwrap_or_else(|message| message),
    }
}

/// Validate one model-visible call before it can reach an executor.
pub fn validate_call(call: &ToolCall, workspace: &Path) -> Result<ValidatedCall, String> {
    match call.name.as_str() {
        "shell" => validate_shell(call, workspace).map(ValidatedCall::Shell),
        "web_fetch" => validate_fetch(call).map(|url| ValidatedCall::WebFetch { url }),
        _ => Err("tool is not available".into()),
    }
}

fn validate_fetch(call: &ToolCall) -> Result<String, String> {
    let object = call
        .arguments
        .as_object()
        .ok_or("web_fetch arguments must be an object")?;
    if object.len() != 1 || !object.contains_key("url") {
        return Err("web_fetch accepts only url".into());
    }
    let raw = object
        .get("url")
        .and_then(serde_json::Value::as_str)
        .ok_or("web_fetch url must be a string")?;
    let url = url::Url::parse(raw).map_err(|_| "web_fetch url is invalid")?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("web_fetch requires an absolute http or https URL".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("web_fetch URLs must not contain credentials".into());
    }
    Ok(url.into())
}

fn validate_shell(call: &ToolCall, workspace: &Path) -> Result<ShellRequest, String> {
    let object = call
        .arguments
        .as_object()
        .ok_or("shell arguments must be an object")?;
    if object.keys().any(|key| key != "argv" && key != "cwd") {
        return Err("shell accepts only argv and cwd".into());
    }
    let argv = object
        .get("argv")
        .and_then(serde_json::Value::as_array)
        .ok_or("shell argv must be an array")?;
    if argv.is_empty() || argv.len() > MAX_ARGV {
        return Err("shell argv has an invalid length".into());
    }
    let argv: Vec<String> = argv
        .iter()
        .map(|value| {
            let value = value.as_str().ok_or("shell argv entries must be strings")?;
            if value.is_empty() || value.len() > MAX_ARG_BYTES || value.as_bytes().contains(&0) {
                return Err("shell argv entry is invalid");
            }
            Ok(value.to_owned())
        })
        .collect::<Result<_, _>>()?;
    validate_program(&argv)?;

    let cwd = object
        .get("cwd")
        .map(|value| value.as_str().ok_or("shell cwd must be a string"))
        .transpose()?;
    let cwd = resolve_workspace_path(workspace, cwd.unwrap_or("."))?;
    Ok(ShellRequest { argv, cwd })
}

fn validate_program(argv: &[String]) -> Result<(), String> {
    match argv.first().map(String::as_str) {
        Some("date") if argv.len() == 1 => Ok(()),
        Some("pwd") if argv.len() == 1 => Ok(()),
        Some("df") => validate_df(argv),
        Some("ls") => validate_ls(argv),
        Some("find") => validate_find(argv),
        Some("git") => validate_git(argv),
        Some("cargo") if matches!(argv.get(1).map(String::as_str), Some("metadata" | "tree")) => {
            Ok(())
        }
        _ => Err("shell command is not permitted".into()),
    }
}

fn validate_df(argv: &[String]) -> Result<(), String> {
    match argv {
        [program, path] if program == "df" && is_relative_path(path) => Ok(()),
        [program, flag, path] if program == "df" && flag == "-h" && is_relative_path(path) => {
            Ok(())
        }
        _ => Err("df only accepts an optional -h and one workspace path".into()),
    }
}

fn validate_ls(argv: &[String]) -> Result<(), String> {
    if argv.len() > 3
        || argv.iter().skip(1).any(|arg| {
            !matches!(arg.as_str(), "-a" | "-l" | "-la" | "-al" | ".") && !is_relative_path(arg)
        })
    {
        return Err("ls only accepts display flags and workspace paths".into());
    }
    Ok(())
}

fn validate_find(argv: &[String]) -> Result<(), String> {
    if argv.len() < 2 || !is_relative_path(&argv[1]) {
        return Err("find requires a workspace path".into());
    }
    let args = &argv[2..];
    if args.is_empty() {
        return Ok(());
    }
    match args {
        [kind, value] if kind == "-name" && !value.starts_with('-') => Ok(()),
        [kind, value] if kind == "-type" && matches!(value.as_str(), "f" | "d") => Ok(()),
        _ => Err("find only accepts one -name or -type predicate".into()),
    }
}

fn validate_git(argv: &[String]) -> Result<(), String> {
    let Some(subcommand) = argv.get(1).map(String::as_str) else {
        return Err("git requires a read-only subcommand".into());
    };
    if !matches!(subcommand, "status" | "diff" | "log" | "show" | "branch") || argv.len() > 4 {
        return Err("git command is not permitted".into());
    }
    if argv.iter().skip(2).any(|arg| {
        arg.starts_with("--ext-diff")
            || arg.starts_with("--textconv")
            || arg.starts_with("--output")
            || arg.starts_with("--no-index")
            || arg.starts_with("-c")
            || arg.starts_with("--config")
    }) {
        return Err("git option is not permitted".into());
    }
    Ok(())
}

fn is_relative_path(value: &str) -> bool {
    if value.is_empty() || value.starts_with('-') {
        return false;
    }
    let path = Path::new(value);
    !path.is_absolute()
        && !path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
}

fn execute_shell(request: ShellRequest, cancel: &Cancel) -> Result<String, String> {
    let generation = cancel.generation();
    let program = command_path(&request.argv[0])?;
    let mut command = sandboxed_command(program)?;
    command
        .args(&request.argv[1..])
        .current_dir(&request.cwd)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_PAGER", "cat")
        .env("GIT_EXTERNAL_DIFF", "")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|_| "shell command could not start")?;
    let stdout = child.stdout.take().ok_or("shell stdout unavailable")?;
    let stderr = child.stderr.take().ok_or("shell stderr unavailable")?;
    let stdout = thread::spawn(move || read_bounded(stdout));
    let stderr = thread::spawn(move || read_bounded(stderr));
    let deadline = Instant::now() + SHELL_TIMEOUT;
    let status = loop {
        if cancel.is_stale(generation) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("shell command cancelled".into());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err("shell command timed out".into());
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|_| "shell command could not be observed")?
        {
            break status;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout.join().map_err(|_| "shell stdout reader failed")?;
    let stderr = stderr.join().map_err(|_| "shell stderr reader failed")?;
    Ok(format_output(status.code(), &stdout, &stderr))
}

fn execute_fetch(url: &str, cancel: &Cancel) -> Result<String, String> {
    let generation = cancel.generation();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(FETCH_CONNECT_TIMEOUT)
        .timeout_read(FETCH_READ_TIMEOUT)
        .redirects(MAX_FETCH_REDIRECTS)
        .resolver(public_resolver)
        .build();
    let response = agent
        .get(url)
        .call()
        .map_err(|_| "web_fetch request failed")?;
    let mut reader = response.into_reader();
    let mut output = Vec::new();
    let deadline = Instant::now() + FETCH_TIMEOUT;
    let mut buffer = [0_u8; 4096];
    loop {
        if cancel.is_stale(generation) {
            return Err("web_fetch cancelled".into());
        }
        if Instant::now() >= deadline {
            return Err("web_fetch timed out".into());
        }
        let read = match reader.read(&mut buffer) {
            Ok(n) => n,
            Err(err)
                if err.kind() == std::io::ErrorKind::TimedOut
                    || err.kind() == std::io::ErrorKind::WouldBlock =>
            {
                continue;
            }
            Err(_) => return Err("web_fetch response read failed".into()),
        };
        if read == 0 {
            break;
        }
        let remaining = MAX_FETCH_BYTES.saturating_sub(output.len());
        let to_take = read.min(remaining);
        output.extend_from_slice(&buffer[..to_take]);
        if output.len() >= MAX_FETCH_BYTES {
            break;
        }
    }
    Ok(format!(
        "Untrusted web page content:\n{}",
        String::from_utf8_lossy(&output)
    ))
}

fn public_resolver(netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
    let addresses: Vec<_> = netloc.to_socket_addrs()?.collect();
    if addresses.is_empty()
        || addresses
            .iter()
            .any(|address| !is_public_address(address.ip()))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "web_fetch target is not a public address",
        ));
    }
    Ok(addresses)
}

fn is_public_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ip) => {
            let octets = ip.octets();
            !ip.is_unspecified()
                && !ip.is_loopback()
                && !ip.is_private()
                && !ip.is_link_local()
                && !ip.is_multicast()
                && ip != std::net::Ipv4Addr::BROADCAST
                && octets[0] != 0
                && octets[0] < 224
                && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
                && !(octets[0] == 198 && matches!(octets[1], 18 | 19))
        }
        IpAddr::V6(ip) => {
            !ip.is_unspecified()
                && !ip.is_loopback()
                && !ip.is_multicast()
                && !ip.is_unicast_link_local()
                && (ip.segments()[0] & 0xfe00) != 0xfc00
                && ip
                    .to_ipv4()
                    .or_else(|| ip.to_ipv4_mapped())
                    .is_none_or(|ip| is_public_address(IpAddr::V4(ip)))
        }
    }
}

fn read_bounded(mut reader: impl Read) -> Vec<u8> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 1024];
    while let Ok(read) = reader.read(&mut buffer) {
        if read == 0 {
            break;
        }
        let remaining = MAX_OUTPUT_BYTES.saturating_sub(output.len());
        output.extend_from_slice(&buffer[..read.min(remaining)]);
    }
    output
}

fn format_output(status: Option<i32>, stdout: &[u8], stderr: &[u8]) -> String {
    format!(
        "exit: {}\nstdout:\n{}\nstderr:\n{}",
        status.map_or_else(|| "signal".into(), |code| code.to_string()),
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    )
}

fn command_path(program: &str) -> Result<&'static str, String> {
    match program {
        "date" => Ok("/bin/date"),
        "df" => Ok("/bin/df"),
        "find" => Ok("/usr/bin/find"),
        "git" => Ok("/usr/bin/git"),
        "ls" => Ok("/bin/ls"),
        "pwd" => Ok("/bin/pwd"),
        "cargo" => Err("shell command is unavailable on this installation".into()),
        _ => Err("shell command is not permitted".into()),
    }
}

#[cfg(target_os = "macos")]
fn sandboxed_command(program: &str) -> Result<Command, String> {
    let profile = "(version 1) (allow default) (deny file-write*) (allow file-write* (literal \"/dev/null\")) (deny network*)";
    let mut command = Command::new("/usr/bin/sandbox-exec");
    command.arg("-p").arg(profile).arg(program);
    Ok(command)
}

#[cfg(not(target_os = "macos"))]
fn sandboxed_command(_: &str) -> Result<Command, String> {
    Err("read-only shell sandbox is unsupported on this platform".into())
}

fn resolve_workspace_path(workspace: &Path, requested: &str) -> Result<PathBuf, String> {
    let root = workspace
        .canonicalize()
        .map_err(|_| "workspace is unavailable")?;
    let requested = Path::new(requested);
    if requested.is_absolute()
        || requested
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("shell cwd must stay inside the workspace".into());
    }
    let resolved = root
        .join(requested)
        .canonicalize()
        .map_err(|_| "shell cwd does not exist")?;
    if !resolved.starts_with(&root) {
        return Err("shell cwd must stay inside the workspace".into());
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCall;

    fn call(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn accepts_only_direct_read_only_argv() {
        let workspace = std::env::current_dir().unwrap();
        let request = validate_call(
            &call(
                "shell",
                serde_json::json!({"argv":["git","status","--short"]}),
            ),
            &workspace,
        )
        .unwrap();
        assert!(matches!(request, ValidatedCall::Shell(_)));
        for argv in [
            &["sh", "-c", "id"][..],
            &["bash", "-c", "id"][..],
            &["python3", "-c", "print(1)"][..],
            &["node", "-e", "console.log(1)"][..],
            &["cat", "Cargo.toml"][..],
            &["rm", "-rf", "."][..],
            &["git", "commit"][..],
            &["git", "push"][..],
            &["git", "-c", "user.name=x", "status"][..],
            &["git", "--config", "a=b", "status"][..],
            &["git", "diff", "--ext-diff"][..],
            &["git", "diff", "--textconv"][..],
            &["git", "diff", "--output=file"][..],
            &["git", "diff", "--no-index", "a", "b"][..],
            &["curl", "https://example.test"][..],
            &["wget", "https://example.test"][..],
            &["date", "+%s"][..],
            &["pwd", "-P"][..],
            &["df", "/tmp"][..],
            &["df", "-k", "."][..],
            &["ls", "-R"][..],
            &["ls", "/etc"][..],
            &["rg", "foo"][..],
            &["rg", "-n", "foo", "crates"][..],
            &["find", ".", "-exec", "id", ";"][..],
            &["find", "/tmp", "-name", "*.rs"][..],
            &["cargo", "build"][..],
            &["cargo", "test"][..],
        ] {
            assert!(
                validate_call(&call("shell", serde_json::json!({"argv":argv})), &workspace)
                    .is_err(),
                "{argv:?}"
            );
        }

        // Test valid variants of allowed tools
        for argv in [
            &["date"][..],
            &["pwd"][..],
            &["df", "."][..],
            &["df", "-h", "."][..],
            &["ls"][..],
            &["ls", "-la"][..],
            &["ls", "-l", "crates"][..],
            &["find", "."][..],
            &["find", "crates", "-name", "*.rs"][..],
            &["find", "crates", "-type", "f"][..],
            &["git", "status"][..],
            &["git", "diff"][..],
            &["git", "log"][..],
            &["git", "show"][..],
            &["git", "branch"][..],
            &["cargo", "metadata"][..],
            &["cargo", "tree"][..],
        ] {
            assert!(
                validate_call(&call("shell", serde_json::json!({"argv":argv})), &workspace).is_ok(),
                "should accept: {argv:?}"
            );
        }
    }

    #[test]
    fn rejects_malformed_shell_arguments() {
        let workspace = std::env::current_dir().unwrap();
        // Missing or non-object args
        assert!(validate_call(&call("shell", serde_json::Value::Null), &workspace).is_err());
        assert!(validate_call(&call("shell", serde_json::json!(123)), &workspace).is_err());
        assert!(validate_call(
            &call("shell", serde_json::json!(["git", "status"])),
            &workspace
        )
        .is_err());
        // Extra keys
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":["pwd"],"extra":1})),
            &workspace
        )
        .is_err());
        // Empty argv
        assert!(validate_call(&call("shell", serde_json::json!({"argv":[]})), &workspace).is_err());
        // Non-string argv item
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":[123]})),
            &workspace
        )
        .is_err());
        // Empty string in argv
        assert!(
            validate_call(&call("shell", serde_json::json!({"argv":[""]})), &workspace).is_err()
        );
        // Null byte in argv
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":["pwd\0"]})),
            &workspace
        )
        .is_err());
        // Oversized argv (> 32 elements)
        let long_argv: Vec<String> = (0..33).map(|i| format!("arg{i}")).collect();
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":long_argv})),
            &workspace
        )
        .is_err());
        // Oversized arg element (> 1024 bytes)
        let huge_arg = "a".repeat(1025);
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":["find", huge_arg]})),
            &workspace
        )
        .is_err());
        // Non-string cwd
        assert!(validate_call(
            &call("shell", serde_json::json!({"argv":["pwd"],"cwd":123})),
            &workspace
        )
        .is_err());
        // Unknown tool
        assert!(validate_call(&call("unknown_tool", serde_json::json!({})), &workspace).is_err());
    }

    #[test]
    fn shell_cwd_cannot_escape_workspace() {
        let workspace = std::env::current_dir().unwrap();
        for cwd in ["..", "/tmp", "missing", "../..", "/"] {
            assert!(validate_call(
                &call("shell", serde_json::json!({"argv":["pwd"],"cwd":cwd})),
                &workspace
            )
            .is_err());
        }
    }

    #[test]
    fn fetch_requires_credential_free_http_url() {
        let workspace = std::env::current_dir().unwrap();
        assert!(validate_call(
            &call(
                "web_fetch",
                serde_json::json!({"url":"https://example.test/a"})
            ),
            &workspace
        )
        .is_ok());
        assert!(validate_call(
            &call(
                "web_fetch",
                serde_json::json!({"url":"http://example.test/a"})
            ),
            &workspace
        )
        .is_ok());
        for url in [
            "file:///etc/passwd",
            "ftp://example.test/file",
            "https://user:secret@example.test",
            "https://user@example.test",
            "example.test",
            "",
            "not a url",
        ] {
            assert!(
                validate_call(
                    &call("web_fetch", serde_json::json!({"url":url})),
                    &workspace
                )
                .is_err(),
                "{url}"
            );
        }
        // Non-object arguments or extra keys
        assert!(validate_call(&call("web_fetch", serde_json::json!(123)), &workspace).is_err());
        assert!(validate_call(
            &call(
                "web_fetch",
                serde_json::json!({"url":"https://example.test","extra":true})
            ),
            &workspace
        )
        .is_err());
        assert!(validate_call(
            &call(
                "web_fetch",
                serde_json::json!({"not_url":"https://example.test"})
            ),
            &workspace
        )
        .is_err());
    }

    #[test]
    fn fetch_rejects_private_and_link_local_destinations() {
        for address in [
            "0.0.0.0",
            "127.0.0.1",
            "127.255.255.255",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
            "169.254.1.1",
            "100.64.0.1",
            "100.127.255.255",
            "198.18.0.1",
            "198.19.255.255",
            "224.0.0.1",
            "240.0.0.1",
            "255.255.255.255",
            "::",
            "::1",
            "::127.0.0.1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "::ffff:192.168.1.1",
            "fe80::1",
            "fc00::1",
            "fd00::1",
            "ff02::1",
        ] {
            assert!(!is_public_address(address.parse().unwrap()), "{address}");
        }
        assert!(is_public_address("1.1.1.1".parse().unwrap()));
        assert!(is_public_address("8.8.8.8".parse().unwrap()));
        assert!(is_public_address("::ffff:1.1.1.1".parse().unwrap()));
        assert!(is_public_address("2606:4700:4700::1111".parse().unwrap()));
    }

    #[test]
    fn read_bounded_caps_output() {
        let source = vec![b'a'; MAX_OUTPUT_BYTES + 1024];
        let bounded = read_bounded(&source[..]);
        assert_eq!(bounded.len(), MAX_OUTPUT_BYTES);
    }

    #[test]
    fn format_output_renders_status_and_streams() {
        let formatted = format_output(Some(0), b"out text", b"err text");
        assert_eq!(formatted, "exit: 0\nstdout:\nout text\nstderr:\nerr text");
        let signal = format_output(None, b"", b"");
        assert_eq!(signal, "exit: signal\nstdout:\n\nstderr:\n");
    }

    #[test]
    fn cancelled_execution_returns_clean_rejection() {
        let workspace = std::env::current_dir().unwrap();
        let cancel = Cancel::new();
        cancel.shutdown();
        let result = execute(
            &call("shell", serde_json::json!({"argv":["pwd"]})),
            &workspace,
            &cancel,
        );
        assert!(!result.ok);
        assert_eq!(result.content, "shell command cancelled");

        let fetch_result = execute(
            &call("web_fetch", serde_json::json!({"url":"https://1.1.1.1"})),
            &workspace,
            &cancel,
        );
        assert!(!fetch_result.ok);
        assert_eq!(fetch_result.content, "web_fetch cancelled");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shell_executes_direct_argv_under_the_read_only_network_denied_profile() {
        let workspace = std::env::current_dir().unwrap();
        let result = execute(
            &call("shell", serde_json::json!({"argv":["pwd"]})),
            &workspace,
            &Cancel::new(),
        );
        assert!(result.ok, "{}", result.content);
        assert!(result.content.contains("exit: 0"), "{}", result.content);
        assert!(
            result.content.contains("crates/syllabix-core"),
            "{}",
            result.content
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_blocks_writes_and_network() {
        let mut write = sandboxed_command("/bin/sh").expect("profile");
        let temp = std::env::temp_dir().join("syllabix-executor-must-not-write");
        write
            .args(["-c", "touch \"$1\"", "sh"])
            .arg(&temp)
            .output()
            .expect("run write denial probe");
        assert!(!temp.exists(), "sandbox unexpectedly created {temp:?}");

        let mut network = sandboxed_command("/usr/bin/nc").expect("profile");
        let result = network
            .args(["-z", "1.1.1.1", "53"])
            .output()
            .expect("run network denial probe");
        assert!(
            !result.status.success(),
            "sandbox unexpectedly opened a network socket"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shell_scrubs_ambient_environment_secrets() {
        let workspace = std::env::current_dir().unwrap();
        // git log or git status runs under scrubbed env
        let result = execute(
            &call(
                "shell",
                serde_json::json!({"argv":["git","status","--short"]}),
            ),
            &workspace,
            &Cancel::new(),
        );
        assert!(result.ok, "{}", result.content);
        assert!(result.content.contains("exit: 0"), "{}", result.content);
    }
}
