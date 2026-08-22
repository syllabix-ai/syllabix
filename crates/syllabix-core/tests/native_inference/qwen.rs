//! Row 31 merge gate, row-32 menu: Qwen3-TTS first-sentence audio,
//! TTS→ASR round-trip through whisper.cpp, the numbers gate (`100` →
//! "hundred"), native cancel, and the row-32 self-voice anchor (same seed ⇒
//! identical PCM; the anchor engages on both backbones). Intelligibility is
//! the same ≥80% in-order word match the Kokoro gate uses; the numbers
//! assertion is the machine proxy for the human listening gate in
//! `V0_LAUNCH.md`.

use std::time::{Duration, Instant};

use syllabix_core::{
    audio::FrameSplitter, transcript_words, word_match_ratio, Cancel, GenerationId, HttpFetcher,
    ModelCache, QwenTts, StderrProgress, Stt, TokenChunk, Tts, TtsModel, TurnId, Utterance,
    TTS_ASR_MIN_WORD_MATCH,
};

use crate::native;

/// One shared engine per backbone for this binary; the shared ggml
/// serializes native work anyway. Seed pinned so failures reproduce.
fn qwen(model: TtsModel) -> QwenTts {
    static CELL_06: std::sync::OnceLock<QwenTts> = std::sync::OnceLock::new();
    static CELL_17: std::sync::OnceLock<QwenTts> = std::sync::OnceLock::new();
    let load = || {
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let cancel = Cancel::new();
        let backbone = cache
            .manifest()
            .asset(model.asset_id())
            .expect("manifest lists the qwen3-tts backbone")
            .clone();
        let mmproj = cache
            .manifest()
            .asset(model.mmproj_asset_id().expect("qwen models have an mmproj"))
            .expect("manifest lists the matching mmproj")
            .clone();
        let model_path = cache
            .resolve(&backbone, &HttpFetcher, &mut progress, &cancel)
            .expect("qwen3-tts backbone in cache");
        let mmproj_path = cache
            .resolve(&mmproj, &HttpFetcher, &mut progress, &cancel)
            .expect("qwen3-tts mmproj in cache");
        QwenTts::from_paths_with_seed(model_path, mmproj_path, "en", model, 42)
            .expect("load Qwen3-TTS once")
    };
    match model {
        TtsModel::Qwen06 => CELL_06.get_or_init(load),
        _ => CELL_17.get_or_init(load),
    }
    .clone()
}

/// Every backbone in the yaml menu must pass the same gates.
fn qwen_backbones() -> Vec<TtsModel> {
    vec![TtsModel::Qwen06, TtsModel::Qwen17]
}

fn token(text: &str, index: u32, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index,
        text: text.into(),
        is_last,
    }
}

fn has_energy(samples: &[i16]) -> bool {
    samples.iter().any(|s| s.abs() > 32)
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

fn speak(tts: &mut QwenTts, text: &str) -> Vec<i16> {
    let chunks = tts
        .synthesize_chunk(&token(text, 0, true), &Cancel::new())
        .expect("qwen synthesize");
    assert!(!chunks.is_empty(), "qwen must emit at least one chunk");
    let mut pcm = Vec::new();
    for chunk in &chunks {
        assert!(has_energy(&chunk.samples), "qwen audio must be voiced");
        pcm.extend_from_slice(&chunk.samples);
    }
    pcm
}

#[test]
fn qwen_first_sentence_arrives_before_completion() {
    // Hold the process-global ggml slot for the whole test, like every
    // native module here: parallel test threads share one ggml.
    let _n = native();
    for model in qwen_backbones() {
        let mut tts = qwen(model);
        assert!(
            tts.voice_anchor_engaged(),
            "{model:?}: self-voice anchor must engage at load"
        );
        let first = tts
            .synthesize_chunk(&token("Hello world. ", 0, false), &Cancel::new())
            .expect("first sentence");
        assert_eq!(first.len(), 1, "first sentence must emit before is_last");
        assert!(!first[0].is_last);
        assert!(
            first[0].samples.len() >= 1_000,
            "playable PCM should be more than a few samples, got {}",
            first[0].samples.len()
        );
        assert!(has_energy(&first[0].samples));

        let rest = tts
            .synthesize_chunk(&token("More later.", 1, true), &Cancel::new())
            .expect("remainder");
        assert_eq!(rest.len(), 1);
        assert!(rest[0].is_last);
        assert!(has_energy(&rest[0].samples));
    }
}

/// The intelligibility gate: Qwen3-TTS out, whisper.cpp `small` back in.
#[test]
fn qwen_speech_round_trips_through_whisper_at_eighty_percent() {
    const TEXT: &str = "The quick brown fox jumps over the lazy dog near the river.";
    let expected_owned = transcript_words(TEXT);
    let expected: Vec<&str> = expected_owned.iter().map(String::as_str).collect();

    let mut n = native();
    for model in qwen_backbones() {
        let mut tts = qwen(model);
        let pcm = speak(&mut tts, TEXT);

        let transcript = n
            .stt
            .transcribe(&pcm_to_utterance(&pcm), &Cancel::new())
            .expect("whisper transcribe qwen audio");
        let ratio = word_match_ratio(&transcript.text, &expected);
        assert!(
            ratio >= TTS_ASR_MIN_WORD_MATCH,
            "{model:?}: TTS→ASR {:?} matched {:.1}% of {:?} (need {:.0}%)",
            transcript.text,
            ratio * 100.0,
            expected,
            TTS_ASR_MIN_WORD_MATCH * 100.0
        );
    }
}

/// The numbers gate, automated: `100` must be spoken as words, not a digit
/// string. Kokoro's G2P gap reads "one zero zero"; the LM reads "one hundred".
#[test]
fn qwen_speaks_numbers_like_a_listener_expects() {
    const TEXT: &str = "That costs 100 dollars.";
    let mut n = native();
    for model in qwen_backbones() {
        let mut tts = qwen(model);
        let pcm = speak(&mut tts, TEXT);

        let transcript = n
            .stt
            .transcribe(&pcm_to_utterance(&pcm), &Cancel::new())
            .expect("whisper transcribe numbers audio");
        let normalized = transcript.text.to_lowercase();
        // whisper.cpp writes a correctly spoken "one hundred dollars" back in
        // symbolic form ("$100"), so the machine proxy asserts the LM reading:
        // the number survives as 100/hundred AND no digit-by-digit "zero"
        // sequence leaks. The human listening gate still owns final quality.
        let reads_like_a_number = normalized.contains("hundred")
            || normalized.contains("100")
            || normalized.contains('$');
        assert!(
            reads_like_a_number,
            "{model:?} numbers gate: `100` must survive the round trip, heard {:?}",
            transcript.text
        );
        assert!(
            !normalized.contains("zero"),
            "{model:?} numbers gate: digit-string reading leaked: {:?}",
            transcript.text
        );
    }
}

/// Row 32 voice contract: the anchor is generated from a fixed seed, so two
/// independently loaded engines with the same sampler seed produce
/// byte-identical audio for the same sentence. This is the machine proof
/// that the voice is pinned (the human listening gate still judges speaker
/// consistency across sentences).
#[test]
fn qwen_voice_is_deterministic_under_a_pinned_seed() {
    let _n = native();
    const A_TEXT: &str = "The weather looks clear today.";
    const B_TEXT: &str = "A short reply is a good reply.";
    for model in qwen_backbones() {
        let load = || {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            let backbone = cache
                .manifest()
                .asset(model.asset_id())
                .expect("backbone listed")
                .clone();
            let mmproj = cache
                .manifest()
                .asset(model.mmproj_asset_id().expect("qwen models have an mmproj"))
                .expect("mmproj listed")
                .clone();
            let model_path = cache
                .resolve(&backbone, &HttpFetcher, &mut progress, &cancel)
                .expect("backbone in cache");
            let mmproj_path = cache
                .resolve(&mmproj, &HttpFetcher, &mut progress, &cancel)
                .expect("mmproj in cache");
            QwenTts::from_paths_with_seed(model_path, mmproj_path, "en", model, 42)
                .expect("load independent engine")
        };
        // Two fully separate contexts (the `qwen()` helper shares one engine).
        let mut first = load();
        let mut second = load();
        assert!(first.voice_anchor_engaged());
        assert!(second.voice_anchor_engaged());
        let pcm_a = speak(&mut first, A_TEXT);
        let pcm_b = speak(&mut second, A_TEXT);
        assert_eq!(
            pcm_a, pcm_b,
            "{model:?}: identical seeds must produce identical audio"
        );
        // The pin fixes identity, not every waveform: other text differs.
        let pcm_c = speak(&mut first, B_TEXT);
        assert_ne!(pcm_a, pcm_c, "{model:?}: different text must differ");
    }
}

/// Barge-in contract: shutdown must surface from inside native generation,
/// not after the full render. The 5 s cap matches the LLM cancel bar.
#[test]
fn qwen_native_cancel_surfaces_within_five_seconds() {
    let _n = native();
    for model in qwen_backbones() {
        let mut tts = qwen(model);
        let cancel = Cancel::new();
        let cancel_for_thread = cancel.clone();
        let start = Instant::now();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            cancel_for_thread.shutdown();
        });
        let long_text = "Please interrupt this sentence. ".repeat(12);
        let err = tts
            .synthesize_chunk(&token(&long_text, 0, true), &cancel)
            .expect_err("cancelled synthesis must error");
        assert!(
            matches!(err, syllabix_core::Error::Cancelled),
            "got {err:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "{model:?}: native cancel took {:?}, budget is 5 s",
            start.elapsed()
        );
    }
}

/// G4 evidence for the row-31 provider and the row-32 backbone: silence-end →
/// first-audio (TTFB) and render speed over ≥20 sentences, per backbone size.
/// Opt-in so the default suite stays fast:
/// `SYLLABIX_QWEN_LATENCY=1 cargo test ... qwen_latency_capture -- --nocapture`
#[test]
fn qwen_latency_capture() {
    if std::env::var_os("SYLLABIX_QWEN_LATENCY").is_none() {
        return;
    }
    const SENTENCES: [&str; 20] = [
        "The weather looks clear today.",
        "Remind me to call the dentist tomorrow.",
        "That restaurant opens at six.",
        "A short reply is a good reply.",
        "Please summarize the article in two sentences.",
        "The train leaves before noon.",
        "Coffee first, questions later.",
        "The meeting moved to Thursday afternoon.",
        "Turn left at the next intersection.",
        "This podcast episode runs about an hour.",
        "She finished the marathon in four hours.",
        "The package arrives sometime next week.",
        "Backup files live in the cloud folder.",
        "He plays guitar in a local band.",
        "Dinner smells almost ready.",
        "The report needs one more revision.",
        "Their flight landed late last night.",
        "Sunrise happens earlier in the summer.",
        "The library closes at eight.",
        "Write the note before you forget it.",
    ];
    fn percentile(mut v: Vec<f64>, f: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v[((v.len() as f64) * f).min((v.len() - 1) as f64) as usize]
    }
    for model in qwen_backbones() {
        let mut tts = qwen(model);
        let mut ttfb_ms: Vec<u128> = Vec::new();
        let mut rtf: Vec<f64> = Vec::new();
        for text in SENTENCES {
            let start = Instant::now();
            let chunks = tts
                .synthesize_chunk(&token(text, 0, true), &Cancel::new())
                .expect("latency turn");
            let elapsed = start.elapsed().as_secs_f64();
            let samples: usize = chunks.iter().map(|c| c.samples.len()).sum();
            let audio_s = samples as f64 / 16_000.0; // v0 contract rate
            ttfb_ms.push(start.elapsed().as_millis());
            if audio_s > 0.0 {
                rtf.push(elapsed / audio_s);
            }
            assert!(!chunks.is_empty());
        }
        println!(
            "qwen latency [{}]: n={} ttfb_ms p50={} p95={}; rtf p50={:.2} p95={:.2}",
            model.as_str(),
            ttfb_ms.len(),
            ttfb_ms[((ttfb_ms.len() as f64) * 0.5).min((ttfb_ms.len() - 1) as f64) as usize],
            ttfb_ms[((ttfb_ms.len() as f64) * 0.95).min((ttfb_ms.len() - 1) as f64) as usize],
            percentile(rtf.clone(), 0.50),
            percentile(rtf, 0.95),
        );
    }
}
