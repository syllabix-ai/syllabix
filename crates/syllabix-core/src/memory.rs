//! Process RSS used by the real-loop post-warm-up ceiling.

/// After native providers load, RSS may grow by this much across the scripted
/// six-turn fixture (KV/context buffers, not unbounded PCM).
pub const LOOP_RSS_GROWTH_CEILING_BYTES: usize = 512 * 1024 * 1024;

/// Current process resident set, when the OS exposes it.
///
/// This deliberately reports host-process RSS only. Native GPU/Metal driver
/// allocations are platform-owned and must not be mixed into benchmark memory
/// records.
pub fn process_rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            let Some(rest) = line.strip_prefix("VmRSS:") else {
                continue;
            };
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
            .map(|kib| kib.saturating_mul(1024))
    }
    #[cfg(target_os = "windows")]
    {
        let command = format!("(Get-Process -Id {}).WorkingSet64", std::process::id());
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &command])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = assert!(LOOP_RSS_GROWTH_CEILING_BYTES > 0);

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_rss_is_nonzero() {
        let rss = process_rss_bytes().expect("VmRSS");
        assert!(rss > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_rss_is_nonzero() {
        let rss = process_rss_bytes().expect("ps rss");
        assert!(rss > 0);
    }
}
