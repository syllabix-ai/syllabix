//! Machine fingerprint for ledger rows (issue #45).
//!
//! Every JSONL record carries who measured it: OS, arch, CPU model, RAM,
//! build profile, crate version, and the git SHA baked in at compile time by
//! `build.rs` (`"unknown"` outside a checkout). Collection is best-effort and
//! std-only — a field that cannot be read degrades to `"unknown"` / `None`
//! instead of failing the run; the CSV groups by whatever is present.

use serde::Serialize;

/// Hardware + build identity of one bench invocation.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Fingerprint {
    /// `std::env::consts::OS` (`macos`, `linux`, `windows`).
    pub os: String,
    /// `std::env::consts::ARCH` (`aarch64`, `x86_64`).
    pub arch: String,
    /// Best-effort CPU model string (`Apple M2`, `model name` from cpuinfo…).
    pub cpu_model: String,
    /// Total physical RAM in GiB (one decimal), when readable.
    pub ram_gb: Option<f64>,
    /// `release` or `debug` — debug rows are excluded from published profiles.
    pub build_profile: &'static str,
    /// Workspace crate version (`CARGO_PKG_VERSION`).
    pub syllabix_version: String,
    /// Short git SHA baked at compile time; `"unknown"` outside a checkout.
    pub git_sha: String,
}

impl Fingerprint {
    /// Read this machine's identity. Never fails: unreadable fields degrade.
    pub fn collect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_model: cpu_model(),
            ram_gb: total_ram_bytes().map(ram_gb),
            build_profile: BUILD_PROFILE,
            syllabix_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: git_sha(),
        }
    }
}

/// Compile-time build profile shared with the report writer.
pub const BUILD_PROFILE: &str = if cfg!(debug_assertions) {
    "debug"
} else {
    "release"
};

fn git_sha() -> String {
    option_env!("SYLLABIX_GIT_SHA")
        .unwrap_or("unknown")
        .to_string()
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        sysctl(&["machdep.cpu.brand_string"])
            .or_else(|| sysctl(&["hw.machine"]))
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| parse_cpuinfo_model(&text))
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unknown".to_string()
    }
}

fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        sysctl(&["hw.memsize"]).and_then(|text| text.trim().parse::<u64>().ok())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| parse_meminfo_total(&text))
    }
    #[cfg(target_os = "windows")]
    {
        // `wmic` is deprecated but still the only key-free probe available to
        // a std-only binary; unreadable RAM stays None.
        wmic_total_physical_memory()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn sysctl(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("sysctl")
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    // `sysctl -n key` still prints `key: value` on newer macOS releases.
    let text = String::from_utf8(output.stdout).ok()?;
    let value = text
        .rsplit_once(':')
        .map(|(_, value)| value.trim())
        .unwrap_or_else(|| text.trim());
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(target_os = "windows")]
fn wmic_total_physical_memory() -> Option<u64> {
    let output = std::process::Command::new("wmic")
        .args(["Computersystem", "get", "TotalPhysicalMemory"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    text.split_whitespace()
        .find_map(|token| token.parse::<u64>().ok())
}

/// First `model name` line of `/proc/cpuinfo` (`Intel(R) Core(TM) …`).
pub fn parse_cpuinfo_model(cpuinfo: &str) -> Option<String> {
    for line in cpuinfo.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            let value = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// `MemTotal:  16384000 kB` → bytes.
pub fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let mut parts = rest.split_whitespace();
            let value: u64 = parts.next()?.parse().ok()?;
            let unit = parts.next().unwrap_or("kB");
            let factor = match unit {
                "kB" => 1024,
                "mB" => 1024 * 1024,
                "gB" => 1024 * 1024 * 1024,
                _ => 1,
            };
            return Some(value.saturating_mul(factor));
        }
    }
    None
}

/// Bytes → GiB with one decimal.
pub fn ram_gb(bytes: u64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0) * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpuinfo_model_name() {
        let cpuinfo = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz\nflags\t: fpu\n";
        assert_eq!(
            parse_cpuinfo_model(cpuinfo).as_deref(),
            Some("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz")
        );
        assert_eq!(parse_cpuinfo_model("processor\t: 0\n"), None);
        assert_eq!(parse_cpuinfo_model("model name\t:   \n"), None);
    }

    #[test]
    fn parses_meminfo_memtotal() {
        let meminfo = "MemTotal:       16384000 kB\nMemFree:         1024000 kB\n";
        assert_eq!(parse_meminfo_total(meminfo), Some(16_384_000 * 1024));
        assert_eq!(parse_meminfo_total("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn ram_gb_rounds_to_one_decimal() {
        assert!((ram_gb(16 * 1024 * 1024 * 1024) - 16.0).abs() < 1e-9);
        assert!((ram_gb(8_589_934_592 + 107_374_182) - 8.1).abs() < 1e-9);
    }

    #[test]
    fn collect_produces_usable_identity_even_without_hardware_reads() {
        let fp = Fingerprint::collect();
        assert!(!fp.os.is_empty());
        assert!(!fp.arch.is_empty());
        assert!(!fp.cpu_model.is_empty());
        assert!(!fp.syllabix_version.is_empty());
        assert!(!fp.git_sha.is_empty());
        assert!(fp.build_profile == "release" || fp.build_profile == "debug");
    }

    #[test]
    fn build_profile_matches_debug_assertions() {
        let expected = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };
        assert_eq!(BUILD_PROFILE, expected);
    }
}
