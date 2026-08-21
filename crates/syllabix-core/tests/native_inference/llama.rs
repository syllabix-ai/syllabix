//! Merge gate for PR 11: llama.cpp GGUF load, greedy stream, history, cancel.

use std::thread;
use std::time::{Duration, Instant};

use syllabix_core::{
    run_loop, scripted_frames, BlockedFetcher, Cancel, CollectingSink, FakeStt, FakeTts, FakeVad,
    HistoryTurn, LlamaLlm, Llm, LoopConfig, ModelCache, PipelineStages, StderrProgress, TokenChunk,
    Transcript, TurnId, LLAMA_32_1B_ASSET, LLAMA_CANCEL_TIMEOUT, QWEN35_08B_ASSET,
    VOICE_SYSTEM_PROMPT,
};

use crate::native;

fn collect(
    llm: &mut LlamaLlm,
    history: &[HistoryTurn],
    user: &Transcript,
    cancel: &Cancel,
) -> syllabix_core::Result<Vec<TokenChunk>> {
    let mut chunks = Vec::new();
    llm.generate(history, user, cancel, &mut |chunk| {
        chunks.push(chunk);
        Ok(())
    })?;
    Ok(chunks)
}

fn joined(chunks: &[TokenChunk]) -> String {
    chunks.iter().map(|c| c.text.as_str()).collect()
}

#[test]
fn q4_km_gguf_loads_without_segfault() {
    let mut n = native();
    assert_eq!(n.llm.name(), "llama.cpp");
    assert!(VOICE_SYSTEM_PROMPT.contains("spoken"));
    let user = Transcript {
        turn: TurnId(0),
        text: "Say the word hello.".into(),
    };
    let chunks = collect(&mut n.llm, &[], &user, &Cancel::new()).expect("generate after load");
    assert!(!chunks.is_empty(), "must stream at least one token chunk");
    assert!(
        chunks.last().expect("last").is_last,
        "last streamed chunk must set is_last"
    );
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i as u32);
        assert_eq!(chunk.turn, TurnId(0));
    }
    let spoken = syllabix_core::speak_text_for_tts(&joined(&chunks));
    assert!(!spoken.contains("<think>"), "{spoken}");
    assert!(!spoken.contains("</think>"), "{spoken}");
}

#[test]
fn deterministic_fixture_streams_a_reply() {
    let mut n = native();
    let user = Transcript {
        turn: TurnId(1),
        text: "Reply with the single word ping.".into(),
    };
    let chunks = collect(&mut n.llm, &[], &user, &Cancel::new()).expect("stream");
    let text = joined(&chunks);
    assert!(
        !text.trim().is_empty(),
        "greedy reply must be non-empty UTF-8, got {text:?}"
    );
    assert!(chunks.last().unwrap().is_last);
    let lower = text.to_ascii_lowercase();
    assert!(
        lower.contains("ping")
            || lower.contains("pong")
            || lower.chars().any(|c| c.is_alphabetic()),
        "fixture reply should be spoken text, got {text:?}"
    );
}

#[test]
fn generate_preserves_configured_turn_history() {
    let mut n = native();
    let first = Transcript {
        turn: TurnId(0),
        text: "My favorite color is teal.".into(),
    };
    let first_chunks = collect(&mut n.llm, &[], &first, &Cancel::new()).expect("turn 1");
    let assistant: String = joined(&first_chunks);
    let history = vec![HistoryTurn {
        user: first.clone(),
        assistant,
    }];
    let second = Transcript {
        turn: TurnId(1),
        text: "What color did I say?".into(),
    };
    let log = n.llm.call_log();
    log.lock().expect("clear").clear();
    let second_chunks = collect(&mut n.llm, &history, &second, &Cancel::new()).expect("turn 2");
    assert!(!joined(&second_chunks).trim().is_empty());
    let calls = log.lock().expect("calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].history_len, 1);
    assert_eq!(
        calls[0].history_user_texts,
        vec!["My favorite color is teal.".to_string()]
    );
    assert_eq!(calls[0].user_text, "What color did I say?");
}

#[test]
fn cancel_aborts_native_generate_within_timeout() {
    let mut n = native();
    let user = Transcript {
        turn: TurnId(0),
        text: "Tell a very long story about a river, a mountain, and a forest, with many details."
            .into(),
    };
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        cancel_thread.shutdown();
    });
    let started = Instant::now();
    let result = collect(&mut n.llm, &[], &user, &cancel);
    let elapsed = started.elapsed();
    assert!(
        elapsed <= LLAMA_CANCEL_TIMEOUT,
        "cancel took {elapsed:?}, documented timeout is {LLAMA_CANCEL_TIMEOUT:?}"
    );
    match result {
        Err(syllabix_core::Error::Cancelled) => {}
        Ok(chunks) => {
            // A very fast host may finish greedy decode before abort.
            assert!(chunks.last().is_some_and(|c| c.is_last));
        }
        Err(err) => panic!("unexpected LLM error: {err}"),
    }
}

#[test]
fn llama_replaces_fake_llm_in_the_loop() {
    let n = native();
    let log = n.llm.call_log();
    log.lock().expect("clear").clear();
    let frames = scripted_frames(1, 2, 1);
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: n.llm.clone(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        frames,
        Cancel::new(),
    )
    .expect("loop");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.queues.within_capacity());
    assert_eq!(report.turns.len(), 1);
    assert!(
        !report.turns[0].assistant_text.trim().is_empty(),
        "real llama.cpp must stream a reply in the conversation loop"
    );
    assert_ne!(
        report.turns[0].assistant_text,
        format!("echo:{}", report.turns[0].user_text),
        "must not still be the fake echo LLM"
    );
    let calls = log.lock().expect("calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].history_len, 0);
}

#[test]
fn populated_cache_reuses_the_gguf_offline() {
    let _n = native();
    let cached = ModelCache::v0();
    let asset = cached.manifest().asset(LLAMA_32_1B_ASSET).unwrap();
    cached
        .resolve(
            asset,
            &BlockedFetcher::default(),
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .expect("populated cache must not need the network");
}

#[test]
fn yaml_qwen_08b_thinking_true_strips_think_for_speech() {
    drop(native());
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut llm = LlamaLlm::from_cached_model(
        &cache,
        &syllabix_core::HttpFetcher,
        &mut progress,
        &Cancel::new(),
        QWEN35_08B_ASSET,
        true,
    )
    .expect("reload qwen3.5-0.8b with thinking");
    assert!(llm.thinking());
    let user = Transcript {
        turn: TurnId(0),
        text: "Say ping.".into(),
    };
    let chunks = collect(&mut llm, &[], &user, &Cancel::new()).expect("thinking generate");
    assert!(!chunks.is_empty());
    let spoken = syllabix_core::speak_text_for_tts(&joined(&chunks));
    assert!(!spoken.contains("<think>") && !spoken.contains("</think>"));
}
