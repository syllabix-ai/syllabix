//! Versioned model manifest, checksummed downloads, and on-disk cache.

mod cache;
mod download;
mod manifest;
mod progress;

pub use cache::{cache_root, ModelCache};
pub use download::{BlockedFetcher, Fetcher, HttpFetcher};
pub use manifest::{Manifest, ModelAsset, ModelLayer};
pub use progress::{format_progress_line, NoProgress, Progress, StderrProgress};
