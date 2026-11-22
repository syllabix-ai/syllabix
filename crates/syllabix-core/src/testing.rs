//! In-memory pipeline stages for tests and example fixtures.
//!
//! These types are not on the supported crate-root list in `docs/embed.md`.
//! Hosts and this repo's tests that script a loop without loading weights
//! should import them from here (`syllabix_core::testing::FakeLlm`, and so on).

pub use crate::fake::{
    scripted_frames, CollectingSink, FailOnceLlm, FailOnceStt, FailOnceTts, FakeAssistPhase,
    FakeLlm, FakeS2s, FakeS2sCloseRecord, FakeS2sOp, FakeStt, FakeTts, FakeVad, LlmCall,
    ScriptedStt, FAKE_S2S_REPLY, FAKE_S2S_TRANSCRIPT,
};
pub use crate::pipeline::{run_s2s_loop, run_s2s_loop_captured, S2sStages};
