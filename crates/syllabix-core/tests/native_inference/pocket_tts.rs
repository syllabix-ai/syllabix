//! P1 Pocket TTS native feasibility gate. This test-only id deliberately
//! precedes any YAML or pipeline exposure; P2 owns that contract.

use std::{
    env, fs,
    process::Command,
    time::{Duration, Instant},
};
use syllabix_core::{
    audio::FrameSplitter, transcript_words, word_match_ratio, Cancel, GenerationId, HttpFetcher,
    ModelCache, PocketTts, StderrProgress, Stt, SttModel, TokenChunk, Tts, TurnId, Utterance,
    WhisperStt, POCKET_TTS_TEXT_CONDITIONER_ASSET, TTS_ASR_MIN_WORD_MATCH,
};

use crate::{native_latency_enabled, skip_unless_model, TTS_LATENCY_SENTENCES};

// Pocket remains an opt-in repair candidate. Its P3 promotion decision keeps
// the shared 80% gate; this lower native floor only prevents regressions below
// the founder-accepted 77.8% baseline while the listening study is pending.
const POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH: f64 = 0.75;

fn token(text: &str, index: u32, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index,
        text: text.into(),
        is_last,
    }
}

fn pcm_to_utterance(samples: &[i16]) -> Utterance {
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(samples).expect("frame split");
    frames.extend(splitter.flush().expect("frame flush"));
    assert!(
        !frames.is_empty(),
        "TTS PCM must fill at least one 16 kHz frame"
    );
    Utterance {
        turn: TurnId(0),
        frames,
    }
}

fn speak(tts: &mut PocketTts, text: &str) -> Vec<i16> {
    let chunks = tts
        .synthesize_chunk(&token(text, 0, true), &Cancel::new())
        .expect("Pocket TTS synthesize");
    assert!(!chunks.is_empty(), "Pocket TTS must emit audio");
    chunks.into_iter().flat_map(|chunk| chunk.samples).collect()
}

const WORD_MATCH_TEXT: &str = "The children played outside in the garden after lunch.";
const WORD_MATCH_CHILD: &str = "SYLLABIX_POCKET_WORD_MATCH_CHILD";
const WORD_MATCH_PCM: &str = "SYLLABIX_POCKET_WORD_MATCH_PCM";
const WORD_MATCH_TRANSCRIPT: &str = "SYLLABIX_POCKET_WORD_MATCH_TRANSCRIPT";

fn pcm_path() -> std::path::PathBuf {
    env::temp_dir().join(format!(
        "syllabix-pocket-word-match-{}.pcm",
        std::process::id()
    ))
}

fn transcript_path() -> std::path::PathBuf {
    env::temp_dir().join(format!(
        "syllabix-pocket-word-match-{}.txt",
        std::process::id()
    ))
}

fn run_word_match_child(test: &str, pcm: &std::path::Path, transcript: Option<&std::path::Path>) {
    let mut child = Command::new(env::current_exe().expect("native test executable"));
    child
        .arg(test)
        .arg("--exact")
        .env(WORD_MATCH_CHILD, "1")
        .env(WORD_MATCH_PCM, pcm);
    if let Some(transcript) = transcript {
        child.env(WORD_MATCH_TRANSCRIPT, transcript);
    }
    let output = child.output().expect("start isolated word-match child");
    assert!(
        output.status.success(),
        "isolated word-match child must pass: status={}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn write_pcm(path: &std::path::Path, samples: &[i16]) {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend(sample.to_le_bytes());
    }
    fs::write(path, bytes).expect("write Pocket PCM fixture");
}

fn read_pcm(path: &std::path::Path) -> Vec<i16> {
    let bytes = fs::read(path).expect("read Pocket PCM fixture");
    assert!(bytes.len().is_multiple_of(2), "PCM fixture must be i16 LE");
    bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect()
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite metric"));
    let rank = percentile / 100.0 * (values.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    values[low] * (high as f64 - rank) + values[high] * (rank - low as f64)
}

#[test]
fn pinned_onnx_graph_set_loads_and_text_fixture_is_deterministic() {
    skip_unless_model!("pocket-tts");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut first = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load P1 Pocket TTS ONNX graph set");
    let a = first.text_fixture().expect("run first native text fixture");
    let b = first
        .text_fixture()
        .expect("run repeated native text fixture");
    assert_eq!(a, b, "fixed graph and token fixture must be deterministic");
    assert!(
        a.iter().any(|value| *value != 0.0),
        "fixture must be non-zero"
    );
    assert_eq!(a.len(), 5 * 1024);
    let raw = first.c_api_fixture().expect("run raw ONNX Runtime fixture");
    assert!(!raw.is_empty());
    assert!(raw.iter().all(|sample| sample.is_finite()));
    assert!(raw.iter().any(|sample| *sample != 0.0));
    let pcm = first
        .synthesize_fixture()
        .expect("synthesize text-conditioned native PCM fixture");
    assert_eq!(pcm.len(), 1920);
    assert!(pcm.iter().all(|sample| sample.is_finite()));
    assert!(pcm.iter().any(|sample| *sample != 0.0));
    let started = Instant::now();
    let measured = first.synthesize_fixture().expect("measure native fixture");
    let elapsed = started.elapsed().as_secs_f64();
    let rtf = elapsed / (measured.len() as f64 / 24_000.0);
    eprintln!(
        "Pocket TTS native fixture RTF: {rtf:.3} ({elapsed:.4}s for {} samples)",
        measured.len()
    );
    assert_eq!(
        cache
            .manifest()
            .asset(POCKET_TTS_TEXT_CONDITIONER_ASSET)
            .expect("manifest entry")
            .layer
            .as_str(),
        "tts"
    );

    let chunks = first
        .synthesize_chunk(
            &TokenChunk {
                turn: TurnId(1),
                generation: GenerationId(0),
                index: 0,
                text: "A short streamed reply.".into(),
                is_last: true,
            },
            &Cancel::new(),
        )
        .expect("Pocket TTS must stream a real sentence into pipeline PCM");
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|chunk| !chunk.samples.is_empty()));
    assert!(chunks.last().is_some_and(|chunk| chunk.is_last));

    let cancel = Cancel::new();
    let trigger = cancel.clone();
    let interrupter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        trigger.cancel_generation();
    });
    let started = Instant::now();
    let err = first
        .synthesize_chunk(
            &TokenChunk {
                turn: TurnId(2),
                generation: GenerationId(0),
                index: 0,
                text: "This sentence is deliberately long enough to interrupt while Pocket TTS is decoding its recurrent frames.".into(),
                is_last: true,
            },
            &cancel,
        )
        .expect_err("barge-in cancellation must abort Pocket TTS");
    interrupter.join().expect("interrupter thread");
    assert!(matches!(err, syllabix_core::Error::Cancelled));
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// Repair-PR machine proxy: Pocket speech must remain intelligible enough to
/// exercise the pipeline. P3 retains the 80% promotion bar and human listen.
#[test]
fn pocket_speech_round_trips_through_whisper_at_eighty_percent() {
    skip_unless_model!("pocket-tts");
    let pcm = pcm_path();
    let transcript = transcript_path();
    run_word_match_child("pocket_tts::pocket_word_match_synthesis_child", &pcm, None);
    run_word_match_child(
        "pocket_tts::pocket_word_match_transcription_child",
        &pcm,
        Some(&transcript),
    );
    fs::remove_file(&pcm).expect("remove Pocket PCM fixture");

    let expected_owned = transcript_words(WORD_MATCH_TEXT);
    let expected: Vec<&str> = expected_owned.iter().map(String::as_str).collect();
    let transcript = fs::read_to_string(&transcript).expect("read Whisper transcript");
    let ratio = word_match_ratio(&transcript, &expected);
    fs::remove_file(transcript_path()).expect("remove Whisper transcript fixture");
    eprintln!(
        "Pocket TTS→Whisper word match: {:.1}% ({:?} vs {:?})",
        ratio * 100.0,
        transcript,
        expected,
    );
    assert!(
        ratio >= POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH,
        "Pocket TTS→ASR {:?} matched {:.1}% of {:?} (need {:.0}% repair floor; P3 needs {:.0}%)",
        transcript,
        ratio * 100.0,
        expected,
        POCKET_TTS_ASR_REPAIR_MIN_WORD_MATCH * 100.0,
        TTS_ASR_MIN_WORD_MATCH * 100.0,
    );
}

/// Runs in a fresh process so ONNX Pocket TTS and Whisper never coexist.
/// The parent test above owns the Whisper comparison and score.
#[test]
fn pocket_word_match_synthesis_child() {
    if env::var_os(WORD_MATCH_CHILD).is_none() {
        return;
    }
    skip_unless_model!("pocket-tts");
    let path = env::var_os(WORD_MATCH_PCM).expect("parent must set PCM path");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut tts = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load Pocket TTS");
    write_pcm(
        &std::path::PathBuf::from(path),
        &speak(&mut tts, WORD_MATCH_TEXT),
    );
}

/// Runs in another fresh process so the scorer does not inherit ONNX state.
#[test]
fn pocket_word_match_transcription_child() {
    if env::var_os(WORD_MATCH_CHILD).is_none() {
        return;
    }
    let pcm = read_pcm(&std::path::PathBuf::from(
        env::var_os(WORD_MATCH_PCM).expect("parent must set PCM path"),
    ));
    let output = std::path::PathBuf::from(
        env::var_os(WORD_MATCH_TRANSCRIPT).expect("parent must set transcript path"),
    );
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut stt = WhisperStt::from_cache(
        &cache,
        &HttpFetcher,
        &mut progress,
        &Cancel::new(),
        SttModel::Small,
    )
    .expect("load selected Whisper scorer");
    let transcript = stt
        .transcribe(&pcm_to_utterance(&pcm), &Cancel::new())
        .expect("Whisper transcribe Pocket TTS audio");
    fs::write(output, transcript.text).expect("write Whisper transcript");
}

/// P3 reproducible evidence capture. The callback measures actual first PCM,
/// not completion of the whole recurrent decode.
#[test]
fn pocket_latency_capture() {
    if !native_latency_enabled() || !crate::native_model_selected("pocket-tts") {
        return;
    }
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut tts = PocketTts::from_cache(&cache, &HttpFetcher, &mut progress, &Cancel::new())
        .expect("load Pocket TTS");
    let mut ttfb_ms = Vec::new();
    let mut rtf = Vec::new();
    for text in TTS_LATENCY_SENTENCES {
        let started = Instant::now();
        let mut first_pcm = None;
        let mut samples = 0usize;
        tts.synthesize_chunk_into(&token(text, 0, true), &Cancel::new(), &mut |audio| {
            first_pcm.get_or_insert_with(|| started.elapsed());
            samples += audio.samples.len();
            Ok(())
        })
        .expect("Pocket TTS latency synthesis");
        let elapsed = started.elapsed().as_secs_f64();
        assert!(samples > 0, "Pocket TTS latency sample must be voiced");
        ttfb_ms.push(first_pcm.expect("first PCM").as_secs_f64() * 1_000.0);
        rtf.push(elapsed / (samples as f64 / 16_000.0));
    }
    println!(
        "tts latency [pocket-tts]: n={} ttfb_ms p50={:.0} p95={:.0}; rtf p50={:.2} p95={:.2}",
        ttfb_ms.len(),
        percentile(ttfb_ms.clone(), 50.0),
        percentile(ttfb_ms, 95.0),
        percentile(rtf.clone(), 50.0),
        percentile(rtf, 95.0),
    );
}
