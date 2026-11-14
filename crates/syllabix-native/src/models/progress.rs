//! Download progress line. First `run` prints this to stderr.

use std::io::{self, Write};

use super::manifest::ModelAsset;

/// Callbacks while resolving an asset.
pub trait Progress {
    /// About to fetch or reuse `asset`.
    fn start(&mut self, asset: &ModelAsset);
    /// Bytes on disk so far versus the manifest size.
    fn bytes(&mut self, downloaded: u64, total: u64);
    /// Asset is ready. `from_cache` is true when no download ran.
    fn finish(&mut self, asset: &ModelAsset, from_cache: bool);
}

/// Ignores progress. Used by tests and quiet callers.
#[derive(Debug, Default)]
pub struct NoProgress;

impl Progress for NoProgress {
    fn start(&mut self, _asset: &ModelAsset) {}
    fn bytes(&mut self, _downloaded: u64, _total: u64) {}
    fn finish(&mut self, _asset: &ModelAsset, _from_cache: bool) {}
}

/// Overwrites a single stderr line while a download is in flight.
#[derive(Debug, Default)]
pub struct StderrProgress {
    current: Option<String>,
}

impl StderrProgress {
    /// Write to stderr.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Progress for StderrProgress {
    fn start(&mut self, asset: &ModelAsset) {
        self.current = Some(asset.id.clone());
        let _ = write!(
            io::stderr(),
            "\r\x1b[2K{}",
            format_progress_line(&asset.id, 0, asset.size_bytes)
        );
        let _ = io::stderr().flush();
    }

    fn bytes(&mut self, downloaded: u64, total: u64) {
        if let Some(id) = &self.current {
            let _ = write!(
                io::stderr(),
                "\r\x1b[2K{}",
                format_progress_line(id, downloaded, total)
            );
            let _ = io::stderr().flush();
        }
    }

    fn finish(&mut self, asset: &ModelAsset, from_cache: bool) {
        let line = if from_cache {
            format!("cached {} ({})", asset.id, format_bytes(asset.size_bytes))
        } else {
            format_progress_line(&asset.id, asset.size_bytes, asset.size_bytes)
        };
        let _ = writeln!(io::stderr(), "\r\x1b[2K{line}");
        self.current = None;
    }
}

/// One progress line: `downloading whisper-small  12.4/466.2 MB (2%)`.
pub fn format_progress_line(id: &str, downloaded: u64, total: u64) -> String {
    let pct = if total == 0 {
        100
    } else {
        ((downloaded as u128 * 100) / total as u128) as u64
    };
    format!(
        "downloading {id}  {}/{} ({pct}%)",
        format_bytes(downloaded),
        format_bytes(total)
    )
}

fn format_bytes(n: u64) -> String {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / MB)
    } else if n >= 1_000 {
        format!("{:.1} kB", n as f64 / KB)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::manifest::ModelLayer;

    fn tiny_asset() -> ModelAsset {
        ModelAsset {
            id: "probe".into(),
            layer: ModelLayer::Vad,
            file_name: "probe.bin".into(),
            url: "https://example.invalid/probe.bin".into(),
            sha256: "00".repeat(32),
            size_bytes: 2_000_000,
        }
    }

    #[test]
    fn progress_line_includes_percent() {
        let line = format_progress_line("whisper-small", 12_400_000, 466_200_000);
        assert!(line.contains("whisper-small"), "{line}");
        assert!(line.contains("2%"), "{line}");
        assert!(line.contains("MB"), "{line}");
    }

    #[test]
    fn progress_line_handles_small_and_zero_total() {
        assert!(format_progress_line("a", 12, 100).contains("12 B"));
        assert!(format_progress_line("a", 1_500, 2_000).contains("kB"));
        assert!(format_progress_line("a", 0, 0).contains("100%"));
    }

    #[test]
    fn no_progress_is_a_noop() {
        let asset = tiny_asset();
        let mut p = NoProgress;
        p.start(&asset);
        p.bytes(1, 2);
        p.finish(&asset, true);
    }

    #[test]
    fn stderr_progress_writes_without_panic() {
        let asset = tiny_asset();
        let mut p = StderrProgress::new();
        p.start(&asset);
        p.bytes(1_000, asset.size_bytes);
        p.finish(&asset, false);
        let mut p = StderrProgress::new();
        p.start(&asset);
        p.finish(&asset, true);
    }
}
