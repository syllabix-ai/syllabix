//! Process RSS used by the real-loop post-warm-up ceiling.

/// After native providers load, RSS may grow by this much across the scripted
/// six-turn fixture (KV/context buffers, not unbounded PCM).
pub const LOOP_RSS_GROWTH_CEILING_BYTES: usize = 512 * 1024 * 1024;

/// Current process resident set, when the OS exposes it.
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
    #[cfg(not(target_os = "linux"))]
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
}
