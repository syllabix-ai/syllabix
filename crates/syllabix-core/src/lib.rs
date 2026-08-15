//! Shared types for the Syllabix runtime.
//!
//! Later pull requests hang pipeline contracts on this crate. This first slice
//! only defines the error type the CLI shell uses.

mod error;

pub use error::{Error, Result};
