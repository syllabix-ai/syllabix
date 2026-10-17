//! Versioned model manifest, checksummed downloads, and on-disk cache.

mod cache;
mod download;
#[cfg(all(not(coverage), feature = "ggml", feature = "cloud"))]
mod goose;
mod manifest;
mod progress;

pub use cache::{cache_root, ModelCache};
pub use download::{BlockedFetcher, Fetcher, HttpFetcher};
pub use manifest::{Manifest, ModelAsset, ModelLayer};
pub use progress::{format_progress_line, NoProgress, Progress, StderrProgress};

#[cfg(all(not(coverage), feature = "ggml", feature = "cloud"))]
pub(crate) use goose::resolve_goose_binary;
