//! In-memory pipeline stages for tests and example fixtures.
//!
//! These types are not on the supported crate-root list in `docs/embed.md`.
//! Hosts and this repo's tests that script a loop without loading weights
//! should import them from here (`syllabix_core::testing::FakeLlm`, and so on).

pub use crate::fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeStt,
    FakeTts, FakeVad, LlmCall, ScriptedStt,
};
