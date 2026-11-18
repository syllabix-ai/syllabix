//! In-memory pipeline stages for tests and example fixtures.
//!
//! These types are not on the supported crate-root list in `docs/embed.md`.
//! Hosts and this repo's tests that script a loop without loading weights
//! should import them from here (`syllabix_core::testing::FakeLlm`, and so on).

pub use crate::fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeS2s,
    FakeStt, FakeTts, FakeVad, LlmCall, ScriptedStt,
};
pub use crate::pipeline::{run_s2s_loop, run_s2s_loop_captured, S2sStages};
