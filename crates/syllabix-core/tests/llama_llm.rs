//! Merge gate for PR 9: llama.cpp GGUF streams a deterministic reply,
//! keeps the configured rolling history, and cancels within `CANCEL_TIMEOUT`.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use syllabix_core::{
    run_loop, scripted_frames, BlockedFetcher, BuiltinDefaults, Cancel, CollectingSink, FakeStt,
    FakeTts, FakeVad, HistoryTurn, HttpFetcher, LlamaLlm, Llm, LlmModel, LoopConfig, LoopMode,
    ModelCache, PipelineStages, StderrProgress, Transcript, TurnId, CANCEL_TIMEOUT,
    MAX_HISTORY_TURNS, VOICE_SYSTEM_PROMPT,
};

/// User text that should yield a one-word spoken reply from Llama-3.2-1B
/// at greedy temperature 0.
const FIXTURE_USER: &str = "Reply with the single word pong and nothing else.";

/// Documented fixture expectation: the streamed reply contains this word.
const FIXTURE_WORD: &str = "pong";

fn llm_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn llama_model_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let cache = ModelCache::v0();
        let asset = cache.manifest().asset("llama-3.2-1b").expect("llm asset");
        cache
            .resolve(
                asset,
                &HttpFetcher,
                &mut StderrProgress::new(),
                &Cancel::new(),
            )
            .expect("resolve llama-3.2-1b")
    })
    .clone()
}

fn load_llama() -> LlamaLlm {
    LlamaLlm::from_model_path(llama_model_path(), LlmModel::Llama32_1b)
        .expect("load llama.cpp GGUF")
}

fn collect_reply(llm: &mut LlamaLlm, history: &[HistoryTurn], text: &str) -> String {
    let user = Transcript {
        turn: TurnId(history.len() as u64),
        text: text.into(),
    };
    let mut reply = String::new();
    llm.generate(history, &user, &Cancel::new(), &mut |chunk| {
        reply.push_str(&chunk.text);
        Ok(())
    })
    .expect("generate");
    reply
}

fn contains_word(text: &str, word: &str) -> bool {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .any(|w| w == word)
}

#[test]
fn deterministic_text_fixture_streams_a_reply() {
    let _guard = llm_lock();
    let mut llm = load_llama();
    let first = collect_reply(&mut llm, &[], FIXTURE_USER);
    assert!(
        contains_word(&first, FIXTURE_WORD),
        "first reply {first:?} did not contain {FIXTURE_WORD:?}"
    );
    let second = collect_reply(&mut llm, &[], FIXTURE_USER);
    assert_eq!(
        first, second,
        "greedy llama.cpp replies must be stable for the documented fixture"
    );

    let cached = ModelCache::v0();
    let asset = cached.manifest().asset("llama-3.2-1b").unwrap();
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
fn configured_turn_history_is_preserved() {
    let _guard = llm_lock();
    let mut llm = load_llama();
    assert_eq!(llm.max_history_turns(), MAX_HISTORY_TURNS);
    assert!(VOICE_SYSTEM_PROMPT.contains("voice assistant"));

    let history = vec![HistoryTurn {
        user: Transcript {
            turn: TurnId(0),
            text: "Your secret code word is willow.".into(),
        },
        assistant: "Okay, I will remember willow.".into(),
    }];
    let reply = collect_reply(
        &mut llm,
        &history,
        "What is the secret code word? Reply with that single word.",
    );
    assert!(
        contains_word(&reply, "willow"),
        "rolling history was not used; reply was {reply:?}"
    );
}

#[test]
fn llama_replaces_fake_llm_in_the_loop() {
    let _guard = llm_lock();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: load_llama(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("loop");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.queues.within_capacity());
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "turn-000");
    assert!(
        !report.turns[0].assistant_text.is_empty(),
        "llama.cpp must stream at least one token through the loop"
    );
    assert!(report.turns[0].token_count > 0);
}

#[test]
fn cancel_returns_within_documented_timeout() {
    let _guard = llm_lock();
    let mut llm = load_llama();
    let cancel = Cancel::new();
    let cancel_thread = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(40));
        cancel_thread.shutdown();
    });
    let started = Instant::now();
    let user = Transcript {
        turn: TurnId(0),
        text: FIXTURE_USER.into(),
    };
    let result = llm.generate(&[], &user, &cancel, &mut |_| Ok(()));
    let elapsed = started.elapsed();
    match result {
        Err(syllabix_core::Error::Cancelled) => {
            assert!(
                elapsed < CANCEL_TIMEOUT,
                "cancel took {elapsed:?}, limit is {CANCEL_TIMEOUT:?}"
            );
        }
        Ok(()) => {
            // A very fast host may finish greedy "pong" before abort.
            assert!(elapsed < CANCEL_TIMEOUT);
        }
        Err(err) => panic!("unexpected LLM error: {err}"),
    }
    drop(llm);
    let _again = load_llama();
}
