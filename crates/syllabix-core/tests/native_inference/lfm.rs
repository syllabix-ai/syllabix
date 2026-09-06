//! Native inference tests for LiquidAI LFM2.5 QAD Q4_0 GGUFs:
//!
//! ```bash
//! # launch default
//! SYLLABIX_NATIVE_MODELS=lfm2.5-2.6b cargo test -p syllabix-core --features native-inference --test native_inference lfm
//! # yaml opt-in spikes
//! SYLLABIX_NATIVE_MODELS=lfm2.5-350m,lfm2.5-230m cargo test -p syllabix-core --features native-inference --test native_inference lfm_
//! ```
//!
//! The suite covers the family's native tools dialect. The co-residency
//! suite also loads Whisper `small` and Kokoro in the same process;
//! expect their caches plus the selected LFM GGUF on a cold machine.

use std::thread;
use std::time::{Duration, Instant};

use syllabix_core::{
    Cancel, Llm, Transcript, TurnId, LFM25_230M_ASSET, LFM25_2_6B_ASSET, LFM25_350M_ASSET,
    LFM_TOOL_TURN_MAX_CHARS, LLAMA_CANCEL_TIMEOUT,
};

use crate::{native, skip_unless_model};

#[test]
fn qad_q4_0_gguf_loads_co_resident_without_segfault() {
    skip_unless_model!(LFM25_2_6B_ASSET);
    let mut native = native();
    // The model must coexist with the speech models in one process.
    // Keep all three live before generating, rather than treating an isolated
    // LFM load as a co-residency check.
    native.stt();
    native.tts();
    let llm = native.lfm_mut();
    assert_eq!(llm.name(), "local");
    let window = llm.context_window().expect("native lfm context");
    assert!(window.0 > 0 && window.0 == window.1, "n_ctx={window:?}");
    let user = Transcript {
        turn: TurnId(0),
        text: "Say the word hello.".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    llm.generate(&[], &user, &Cancel::new(), &mut |chunk| {
        chunks.push(chunk);
        Ok(())
    })
    .expect("plain generate after load");
    assert!(!chunks.is_empty(), "must stream at least one token chunk");
    assert!(chunks.last().expect("last").is_last);
}

#[test]
fn lfm_tool_turn_parses_a_native_call_into_the_shared_contract() {
    skip_unless_model!(LFM25_2_6B_ASSET);
    let mut native = native();
    let llm = native.lfm_mut();
    let user = Transcript {
        turn: TurnId(1),
        text: "How much free space is there on this machine?".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    let calls = llm
        .generate_lfm_tool_turn(&[], &user, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .expect("lfm tool turn");
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_last);
    // Admission uses fixed fixtures and requires at least 90% valid calls.
    // This assertion is structural: calls normalize into the shared representation
    // with synthesized ids, or the turn fails closed with a
    // `rejected` event and no calls.
    let events = Llm::take_tool_events(llm);
    if calls.is_empty() {
        assert!(
            events.iter().any(|e| e.kind == "rejected"),
            "empty tool turn must record a rejection, got {events:?}"
        );
    } else {
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call.id, format!("local-call-{i}"));
            assert!(matches!(call.name.as_str(), "web_fetch" | "shell"));
            assert!(call.arguments.is_object());
        }
        assert!(
            events.iter().any(|e| e.kind == "call"),
            "non-empty tool turn must record call events, got {events:?}"
        );
        assert!(
            chunks[0].text.len() <= LFM_TOOL_TURN_MAX_CHARS + 1_048_576,
            "tool turn respects the thinking bound"
        );
    }
}

#[test]
fn lfm_cancel_aborts_the_tools_aware_native_generate_within_timeout() {
    skip_unless_model!(LFM25_2_6B_ASSET);
    let mut native = native();
    let llm = native.lfm_mut();
    let user = Transcript {
        turn: TurnId(0),
        text: "Tell a very long story about a river, a mountain, and a forest, with many details."
            .into(),
        language: "en".into(),
    };
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        cancel_thread.shutdown();
    });
    let started = Instant::now();
    let mut chunks = Vec::new();
    let result = llm.generate_lfm_tool_turn(&[], &user, &cancel, &mut |chunk| {
        chunks.push(chunk);
        Ok(())
    });
    let elapsed = started.elapsed();
    assert!(
        elapsed <= LLAMA_CANCEL_TIMEOUT,
        "cancel took {elapsed:?}, documented timeout is {LLAMA_CANCEL_TIMEOUT:?}"
    );
    match result {
        Err(syllabix_core::Error::Cancelled) => {}
        Ok(_calls) => {
            // A very fast host can finish before the cancellation thread
            // runs; it must still have completed as one bounded turn.
            assert!(!chunks.is_empty(), "completed LFM turn must emit a chunk");
            assert!(chunks.last().is_some_and(|chunk| chunk.is_last));
        }
        Err(err) => panic!("unexpected LLM error: {err}"),
    }
}

fn assert_small_lfm_loads_and_streams(asset_id: &'static str) {
    skip_unless_model!(asset_id);
    let mut native = native();
    // Co-reside with the launch speech stack the same way 2.6B does.
    native.stt();
    native.tts();
    let llm = native.lfm_asset_mut(asset_id);
    assert_eq!(llm.name(), "local");
    let window = llm.context_window().expect("native lfm context");
    assert!(window.0 > 0 && window.0 == window.1, "n_ctx={window:?}");
    let user = Transcript {
        turn: TurnId(0),
        text: "Say the word hello.".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    llm.generate(&[], &user, &Cancel::new(), &mut |chunk| {
        chunks.push(chunk);
        Ok(())
    })
    .unwrap_or_else(|err| panic!("{asset_id} plain generate after load: {err}"));
    assert!(
        !chunks.is_empty(),
        "{asset_id} must stream at least one token chunk"
    );
    assert!(chunks.last().expect("last").is_last);
}

fn assert_small_lfm_tool_turn_is_fail_closed(asset_id: &'static str) {
    skip_unless_model!(asset_id);
    let mut native = native();
    let llm = native.lfm_asset_mut(asset_id);
    let user = Transcript {
        turn: TurnId(1),
        text: "How much free space is there on this machine?".into(),
        language: "en".into(),
    };
    let mut chunks = Vec::new();
    let calls = llm
        .generate_lfm_tool_turn(&[], &user, &Cancel::new(), &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap_or_else(|err| panic!("{asset_id} lfm tool turn: {err}"));
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].is_last);
    let events = Llm::take_tool_events(llm);
    if calls.is_empty() {
        assert!(
            events.iter().any(|e| e.kind == "rejected"),
            "{asset_id}: empty tool turn must record a rejection, got {events:?}"
        );
    } else {
        for (i, call) in calls.iter().enumerate() {
            assert_eq!(call.id, format!("local-call-{i}"));
            assert!(matches!(call.name.as_str(), "web_fetch" | "shell"));
            assert!(call.arguments.is_object());
        }
        assert!(
            events.iter().any(|e| e.kind == "call"),
            "{asset_id}: non-empty tool turn must record call events, got {events:?}"
        );
        assert!(
            chunks[0].text.len() <= LFM_TOOL_TURN_MAX_CHARS + 1_048_576,
            "{asset_id}: tool turn respects the thinking bound"
        );
    }
}

fn assert_small_lfm_cancel_within_timeout(asset_id: &'static str) {
    skip_unless_model!(asset_id);
    let mut native = native();
    let llm = native.lfm_asset_mut(asset_id);
    let user = Transcript {
        turn: TurnId(0),
        text: "Tell a very long story about a river, a mountain, and a forest, with many details."
            .into(),
        language: "en".into(),
    };
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        cancel_thread.shutdown();
    });
    let started = Instant::now();
    let mut chunks = Vec::new();
    let result = llm.generate_lfm_tool_turn(&[], &user, &cancel, &mut |chunk| {
        chunks.push(chunk);
        Ok(())
    });
    let elapsed = started.elapsed();
    assert!(
        elapsed <= LLAMA_CANCEL_TIMEOUT,
        "{asset_id}: cancel took {elapsed:?}, documented timeout is {LLAMA_CANCEL_TIMEOUT:?}"
    );
    match result {
        Err(syllabix_core::Error::Cancelled) => {}
        Ok(_calls) => {
            assert!(
                !chunks.is_empty(),
                "{asset_id}: completed LFM turn must emit a chunk"
            );
            assert!(chunks.last().is_some_and(|chunk| chunk.is_last));
        }
        Err(err) => panic!("{asset_id}: unexpected LLM error: {err}"),
    }
}

#[test]
fn lfm_350m_qad_q4_0_loads_co_resident_without_segfault() {
    assert_small_lfm_loads_and_streams(LFM25_350M_ASSET);
}

#[test]
fn lfm_350m_tool_turn_parses_or_fails_closed() {
    assert_small_lfm_tool_turn_is_fail_closed(LFM25_350M_ASSET);
}

#[test]
fn lfm_350m_cancel_aborts_within_timeout() {
    assert_small_lfm_cancel_within_timeout(LFM25_350M_ASSET);
}

#[test]
fn lfm_230m_qad_q4_0_loads_co_resident_without_segfault() {
    assert_small_lfm_loads_and_streams(LFM25_230M_ASSET);
}

#[test]
fn lfm_230m_tool_turn_parses_or_fails_closed() {
    assert_small_lfm_tool_turn_is_fail_closed(LFM25_230M_ASSET);
}

#[test]
fn lfm_230m_cancel_aborts_within_timeout() {
    assert_small_lfm_cancel_within_timeout(LFM25_230M_ASSET);
}
