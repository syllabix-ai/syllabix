//! Mock-session coverage for the real Moonshine adapter. No weights, no devices.
//!
//! The production ONNX graphs are replaced with scripted [`MockSession`]s
//! behind the shared [`OnnxSession`](crate::onnx::OnnxSession) seam; the
//! tokenizer parses a tiny synthetic vocabulary.

use std::path::{Path, PathBuf};

use super::*;
use crate::onnx::{MockSession, OnnxSession, OnnxTensor};
use crate::{
    AudioFrame, Cancel, Error, Stt, SttModel, TurnId, Utterance, DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};

const SYNTHETIC_VOCAB: &str = r#"{
    "model": {"type": "BPE", "vocab": {
        "<s>": 1, "</s>": 2, "▁hi": 3, "▁there": 4
    }, "merges": []},
    "added_tokens": [
        {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    ],
    "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]}
}"#;

/// Logits over the four-id synthetic vocab peaking at `top`.
fn logits(top: i64) -> OnnxTensor {
    let mut values = vec![-10.0_f32; 5];
    values[top as usize] = 10.0;
    OnnxTensor::f32(vec![1, 1, 5], values)
}

fn kv() -> OnnxTensor {
    OnnxTensor::f32(vec![1, 1, 4], vec![0.0; 4])
}

fn names(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_string()).collect()
}

fn tokenizer() -> MoonshineTokenizer {
    MoonshineTokenizer::from_json(SYNTHETIC_VOCAB).expect("synthetic vocab parses")
}

fn mock_encoder() -> Box<dyn OnnxSession> {
    Box::new(MockSession::with_handler(
        names(&["input_values", "attention_mask"]),
        names(&["encoder_hidden_states"]),
        |_, _| Ok(vec![OnnxTensor::f32(vec![1, 2, 4], vec![0.5; 8])]),
    ))
}

/// Decoder answers the first call with `first_top`, every later call with EOS.
fn mock_decoder(first_top: i64) -> Box<dyn OnnxSession> {
    let calls = std::cell::Cell::new(0_usize);
    Box::new(MockSession::with_handler(
        names(&["decoder_input_ids", "encoder_hidden_states"]),
        names(&["logits", "present_0"]),
        move |_, _| {
            let call = calls.get();
            calls.set(call + 1);
            let top = if call == 0 { first_top } else { EOS };
            Ok(vec![logits(top), kv()])
        },
    ))
}

fn mock_past(top: i64) -> Box<dyn OnnxSession> {
    Box::new(MockSession::with_handler(
        names(&["decoder_input_ids", "encoder_hidden_states", "past_0"]),
        names(&["logits"]),
        move |_, _| Ok(vec![logits(top)]),
    ))
}

/// Adapter that transcribes one frame batch as "hi".
fn mock_adapter() -> MoonshineStt {
    MoonshineStt::from_parts(mock_encoder(), mock_decoder(3), mock_past(EOS), tokenizer())
        .expect("mock contract is valid")
}

fn frame(seq: u64) -> AudioFrame {
    AudioFrame {
        seq,
        sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
        channels: DEFAULT_CHANNELS,
        samples: vec![1000; FRAME_SAMPLES],
        capture_pcm: None,
    }
}

fn utterance(turn: u64, frames: usize) -> Utterance {
    Utterance {
        turn: TurnId(turn),
        frames: (0..frames as u64).map(frame).collect(),
    }
}

#[test]
fn mock_adapter_reports_identity_and_language() {
    let stt = mock_adapter();
    assert_eq!(stt.name(), PROVIDER_NAME);
    assert_eq!(stt.language(), LANGUAGE);
    assert!(stt.supports_partials());
}

#[test]
fn with_language_accepts_en() {
    let stt = mock_adapter().with_language("en").expect("en is accepted");
    assert_eq!(stt.language(), "en");
}

#[test]
fn transcribe_decodes_the_scripted_ids() {
    let mut stt = mock_adapter();
    let transcript = stt
        .transcribe(&utterance(7, 2), &Cancel::new())
        .expect("mock transcription");
    assert_eq!(transcript.turn, TurnId(7));
    assert_eq!(transcript.text, "hi");
    assert_eq!(transcript.language, LANGUAGE);
}

#[test]
fn transcribe_rejects_empty_utterances() {
    let mut stt = mock_adapter();
    let empty = Utterance {
        turn: TurnId(1),
        frames: vec![],
    };
    let err = stt.transcribe(&empty, &Cancel::new()).unwrap_err();
    assert!(err.to_string().contains("utterance has no frames"));
    let silent = Utterance {
        turn: TurnId(1),
        frames: vec![AudioFrame {
            seq: 0,
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
            samples: vec![],
            capture_pcm: None,
        }],
    };
    let err = stt.transcribe(&silent, &Cancel::new()).unwrap_err();
    assert!(err.to_string().contains("utterance has no samples"));
}

#[test]
fn transcribe_observes_shutdown() {
    let mut stt = mock_adapter();
    let cancel = Cancel::new();
    cancel.shutdown();
    assert!(matches!(
        stt.transcribe(&utterance(1, 1), &cancel),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        stt.start_turn(TurnId(1), &cancel),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        stt.push_frame(&frame(0), &cancel),
        Err(Error::Cancelled)
    ));
}

#[test]
fn transcribe_clears_the_active_turn() {
    let mut stt = mock_adapter();
    stt.start_turn(TurnId(3), &Cancel::new()).unwrap();
    stt.transcribe(&utterance(3, 1), &Cancel::new()).unwrap();
    // The finalized turn no longer accepts partial frames.
    assert_eq!(stt.push_frame(&frame(0), &Cancel::new()).unwrap(), None);
}

#[test]
fn partials_arrive_on_cadence_and_stay_monotonic() {
    let mut stt = mock_adapter();
    stt.start_turn(TurnId(1), &Cancel::new()).unwrap();
    for seq in 0..PARTIAL_EVERY_FRAMES - 1 {
        assert_eq!(
            stt.push_frame(&frame(seq as u64), &Cancel::new()).unwrap(),
            None,
            "partials wait for {PARTIAL_EVERY_FRAMES} frames"
        );
    }
    assert_eq!(
        stt.push_frame(&frame(99), &Cancel::new()).unwrap(),
        Some("hi".to_string())
    );
    // Repeating the same hypothesis emits nothing new.
    for seq in 100..100 + PARTIAL_EVERY_FRAMES {
        assert_eq!(
            stt.push_frame(&frame(seq as u64), &Cancel::new()).unwrap(),
            None
        );
    }
}

#[test]
fn push_frame_without_a_turn_is_quiet() {
    let mut stt = mock_adapter();
    assert_eq!(stt.push_frame(&frame(0), &Cancel::new()).unwrap(), None);
}

#[test]
fn cancel_turn_drops_partials() {
    let mut stt = mock_adapter();
    stt.start_turn(TurnId(1), &Cancel::new()).unwrap();
    stt.cancel_turn(TurnId(1));
    assert_eq!(stt.push_frame(&frame(0), &Cancel::new()).unwrap(), None);
    // Cancelling another turn leaves this one alone.
    stt.start_turn(TurnId(2), &Cancel::new()).unwrap();
    stt.cancel_turn(TurnId(9));
    assert!(stt.push_frame(&frame(1), &Cancel::new()).is_ok());
}

#[test]
fn empty_hypothesis_emits_no_partial() {
    // Decoder answers EOS immediately: the hypothesis stays empty.
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        mock_decoder(EOS),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    stt.start_turn(TurnId(1), &Cancel::new()).unwrap();
    for seq in 0..PARTIAL_EVERY_FRAMES as u64 {
        assert_eq!(stt.push_frame(&frame(seq), &Cancel::new()).unwrap(), None);
    }
}

#[test]
fn decode_runs_to_the_token_budget_without_eos() {
    // Every step answers "hi": the loop exits on the token budget, not EOS.
    let mut stt =
        MoonshineStt::from_parts(mock_encoder(), mock_decoder(3), mock_past(3), tokenizer())
            .unwrap();
    let transcript = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .expect("budget-bounded decode");
    assert!(!transcript.text.is_empty());
    assert!(transcript.text.contains("hi"));
}

#[test]
fn shutdown_mid_decode_cancels_the_turn() {
    let cancel = Cancel::new();
    let worker = cancel.clone();
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        mock_decoder(3),
        Box::new(MockSession::with_handler(
            names(&["decoder_input_ids", "encoder_hidden_states", "past_0"]),
            names(&["logits"]),
            move |_, _| {
                worker.shutdown();
                Ok(vec![logits(3)])
            },
        )),
        tokenizer(),
    )
    .unwrap();
    assert!(matches!(
        stt.transcribe(&utterance(1, 1), &cancel),
        Err(Error::Cancelled)
    ));
}

#[test]
fn encoder_failure_aborts_transcription() {
    let mut stt = MoonshineStt::from_parts(
        Box::new(MockSession::failing(
            names(&["input_values", "attention_mask"]),
            names(&["encoder_hidden_states"]),
            "encoder down",
        )),
        mock_decoder(3),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("encoder down"));
}

#[test]
fn empty_encoder_output_is_a_provider_error() {
    let mut stt = MoonshineStt::from_parts(
        Box::new(MockSession::script(
            names(&["input_values", "attention_mask"]),
            names(&["encoder_hidden_states"]),
            vec![Ok(vec![])],
        )),
        mock_decoder(3),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("encoder returned no output"));
}

#[test]
fn non_float_encoder_states_are_rejected() {
    let mut stt = MoonshineStt::from_parts(
        Box::new(MockSession::with_handler(
            names(&["input_values", "attention_mask"]),
            names(&["encoder_hidden_states"]),
            |_, _| Ok(vec![OnnxTensor::i64(vec![1, 2], vec![0, 0])]),
        )),
        mock_decoder(3),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("encoder emitted non-f32 states"));
}

#[test]
fn empty_decoder_output_is_a_provider_error() {
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        Box::new(MockSession::script(
            names(&["decoder_input_ids", "encoder_hidden_states"]),
            names(&["logits", "present_0"]),
            vec![Ok(vec![])],
        )),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("decoder returned no logits"));
}

#[test]
fn decoder_kv_mismatch_is_a_provider_error() {
    // The decoder drops its KV tensor while the map still expects one.
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        Box::new(MockSession::with_handler(
            names(&["decoder_input_ids", "encoder_hidden_states"]),
            names(&["logits", "present_0"]),
            |_, _| Ok(vec![logits(3)]),
        )),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("KV count"));
}

#[test]
fn non_float_logits_are_rejected() {
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        Box::new(MockSession::with_handler(
            names(&["decoder_input_ids", "encoder_hidden_states"]),
            names(&["logits", "present_0"]),
            |_, _| Ok(vec![OnnxTensor::i64(vec![1, 1, 5], vec![0; 5]), kv()]),
        )),
        mock_past(EOS),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("logits are not f32"));
}

#[test]
fn empty_past_output_is_a_provider_error() {
    let mut stt = MoonshineStt::from_parts(
        mock_encoder(),
        mock_decoder(3),
        Box::new(MockSession::script(
            names(&["decoder_input_ids", "encoder_hidden_states", "past_0"]),
            names(&["logits"]),
            vec![Ok(vec![])],
        )),
        tokenizer(),
    )
    .unwrap();
    let err = stt
        .transcribe(&utterance(1, 1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("KV decoder returned no logits"));
}

#[test]
fn contract_validation_rejects_each_wrong_graph() {
    let good_encoder = || mock_encoder();
    let good_decoder = || mock_decoder(3);
    let good_past = || mock_past(EOS);
    let bad_names = |inputs: &[&str], outputs: &[&str]| {
        Box::new(MockSession::script(names(inputs), names(outputs), vec![])) as Box<dyn OnnxSession>
    };
    let parts_error = |encoder: Box<dyn OnnxSession>,
                       decoder: Box<dyn OnnxSession>,
                       past: Box<dyn OnnxSession>| {
        match MoonshineStt::from_parts(encoder, decoder, past, tokenizer()) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("bad contract should fail"),
        }
    };

    let err = parts_error(
        bad_names(&["wrong"], &["encoder_hidden_states"]),
        good_decoder(),
        good_past(),
    );
    assert!(err.contains("unexpected encoder inputs"));

    let err = parts_error(
        bad_names(&["input_values", "attention_mask"], &["wrong"]),
        good_decoder(),
        good_past(),
    );
    assert!(err.contains("unexpected encoder outputs"));

    let err = parts_error(
        good_encoder(),
        bad_names(&["wrong"], &["logits", "present_0"]),
        good_past(),
    );
    assert!(err.contains("unexpected decoder inputs"));

    let err = parts_error(
        good_encoder(),
        bad_names(
            &["decoder_input_ids", "encoder_hidden_states"],
            &["wrong", "present_0"],
        ),
        good_past(),
    );
    assert!(err.contains("decoder must emit logits first"));

    let err = parts_error(
        good_encoder(),
        good_decoder(),
        bad_names(
            &["decoder_input_ids", "encoder_hidden_states", "past_0"],
            &["wrong"],
        ),
    );
    assert!(err.contains("KV decoder must emit logits first"));
}

#[test]
fn kv_map_supports_the_orig_fallback() {
    let map = build_kv_map(
        &["logits".to_string(), "present_cross".to_string()],
        &[
            "decoder_input_ids".to_string(),
            "present_cross_orig".to_string(),
        ],
    )
    .expect("orig fallback maps");
    assert_eq!(map, vec![(1, 1)]);
}

#[test]
fn kv_map_rejects_unmapped_outputs() {
    let err = build_kv_map(
        &["logits".to_string(), "noisy".to_string()],
        &["past_noisy".to_string()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("without present_ prefix"));
    let err = build_kv_map(
        &["logits".to_string(), "present_missing".to_string()],
        &["past_elsewhere".to_string()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("has no past input"));
}

#[test]
fn from_cache_rejects_non_moonshine_models_before_any_fetch() {
    let root = std::env::temp_dir().join(format!(
        "syllabix-moonshine-reject-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let cache = crate::models::ModelCache::new(
        root,
        crate::models::Manifest {
            version: 1,
            assets: vec![],
        },
    );
    let err = match MoonshineStt::from_cache(
        &cache,
        &crate::models::BlockedFetcher::default(),
        &mut crate::models::NoProgress,
        &Cancel::new(),
        SttModel::Small,
    ) {
        Err(err) => err,
        Ok(_) => panic!("non-Moonshine model should fail"),
    };
    assert!(err.to_string().contains("requires a Moonshine model"));
}

#[test]
#[ignore]
fn native_recorded_fixtures_meet_word_match_gate() {
    let dir = std::env::var("SYLLABIX_MOONSHINE_DIR")
        .expect("SYLLABIX_MOONSHINE_DIR must point at the moonshine ONNX directory");
    let root = Path::new(&dir);
    let mut stt = MoonshineStt::from_paths(&[
        root.join("encoder_model_int8.onnx"),
        root.join("decoder_model_int8.onnx"),
        root.join("decoder_with_past_model_int8.onnx"),
        root.join("tokenizer.json"),
    ])
    .expect("moonshine weights load");
    assert_eq!(stt.name(), PROVIDER_NAME);
    assert_eq!(stt.language(), LANGUAGE);
    let fixtures: &[(&str, &[&str], f64)] = &[
        (
            "jfk.wav",
            &[
                "and",
                "so",
                "my",
                "fellow",
                "americans",
                "ask",
                "not",
                "what",
                "your",
                "country",
                "can",
                "do",
                "for",
                "you",
                "ask",
                "what",
                "you",
                "can",
                "do",
                "for",
                "your",
                "country",
            ],
            1.0,
        ),
        (
            "librispeech-1089-134686-0000.wav",
            &[
                "he", "hoped", "there", "would", "be", "stew", "for", "dinner", "turnips", "and",
                "carrots", "and", "bruised", "potatoes", "and", "fat", "mutton", "pieces", "to",
                "be", "ladled", "out", "in", "thick", "peppered", "flour", "fattened", "sauce",
            ],
            0.8,
        ),
        (
            "librispeech-121-127105-0009.wav",
            &["she", "has", "been", "dead", "these", "twenty", "years"],
            0.8,
        ),
        (
            "librispeech-1995-1837-0005.wav",
            &[
                "she", "was", "so", "strange", "and", "human", "a", "creature",
            ],
            0.8,
        ),
    ];
    for (index, (file, expected, minimum)) in fixtures.iter().enumerate() {
        let wav = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/stt")
                .join(file),
        )
        .unwrap_or_else(|err| panic!("{file} fixture: {err}"));
        let pcm = crate::audio::read_wav(std::io::Cursor::new(wav))
            .unwrap_or_else(|err| panic!("{file} parses: {err}"));
        let frames = crate::audio::record_fixture_to_frames(&pcm)
            .unwrap_or_else(|err| panic!("{file} frames: {err}"));
        let transcript = stt
            .transcribe(
                &Utterance {
                    turn: crate::types::TurnId(index as u64),
                    frames,
                },
                &Cancel::new(),
            )
            .unwrap_or_else(|err| panic!("{file} transcribes: {err}"));
        let ratio = crate::stt::word_match_ratio(&transcript.text, expected);
        eprintln!(
            "Moonshine {file}: {:.1}% word match ({:?})",
            ratio * 100.0,
            transcript.text
        );
        assert!(
            ratio >= *minimum,
            "{file} transcript {:?} matched {:.1}% of {:?} (need {:.0}%)",
            transcript.text,
            ratio * 100.0,
            expected,
            minimum * 100.0
        );
        assert_eq!(transcript.language, LANGUAGE);
    }
}

fn relu_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/onnx/relu-tiny.onnx")
}

fn missing() -> PathBuf {
    PathBuf::from("/no/such/moonshine-graph.onnx")
}

fn paths_error(paths: &[PathBuf]) -> String {
    match MoonshineStt::from_paths(paths) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("bad paths should fail"),
    }
}

#[test]
fn from_cache_reports_a_missing_manifest_asset() {
    let root = std::env::temp_dir().join(format!(
        "syllabix-moonshine-cache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let cache = crate::models::ModelCache::new(
        root,
        crate::models::Manifest {
            version: 1,
            assets: vec![],
        },
    );
    let err = match MoonshineStt::from_cache(
        &cache,
        &crate::models::BlockedFetcher::default(),
        &mut crate::models::NoProgress,
        &Cancel::new(),
        SttModel::MoonshineStreamingSmall,
    ) {
        Err(err) => err,
        Ok(_) => panic!("empty manifest should fail"),
    };
    assert!(err.to_string().contains("moonshine-encoder"));
}

#[test]
fn from_cache_surfaces_unresolvable_assets() {
    use crate::models::{Manifest, ModelAsset, ModelCache, ModelLayer};
    let root = std::env::temp_dir().join(format!(
        "syllabix-moonshine-blocked-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let assets = [
        MEDIUM_ENCODER_ASSET,
        MEDIUM_DECODER_ASSET,
        MEDIUM_DECODER_PAST_ASSET,
        MEDIUM_TOKENIZER_ASSET,
    ]
    .iter()
    .map(|id| ModelAsset {
        id: id.to_string(),
        layer: ModelLayer::Stt,
        file_name: format!("{id}.bin"),
        url: "https://example.invalid/model.bin".to_string(),
        sha256: "0".repeat(64),
        size_bytes: 1,
    })
    .collect();
    let cache = ModelCache::new(root, Manifest { version: 1, assets });
    let err = match MoonshineStt::from_cache(
        &cache,
        &crate::models::BlockedFetcher::default(),
        &mut crate::models::NoProgress,
        &Cancel::new(),
        SttModel::MoonshineStreamingMedium,
    ) {
        Err(err) => err,
        Ok(_) => panic!("blocked fetcher should fail"),
    };
    assert!(!err.to_string().is_empty());
}

#[test]
fn from_paths_reports_each_missing_graph_in_order() {
    let relu = relu_fixture();
    let err = paths_error(&[relu.clone(), missing(), missing(), missing()]);
    assert!(err.contains("could not load moonshine decoder"));
    let err = paths_error(&[relu.clone(), relu.clone(), missing(), missing()]);
    assert!(err.contains("could not load moonshine KV decoder"));
    let err = paths_error(&[relu.clone(), relu.clone(), relu.clone(), missing()]);
    assert!(err.contains("could not load tokenizer"));
}

#[test]
fn from_paths_validates_the_loaded_contract() {
    let dir = std::env::temp_dir().join(format!(
        "syllabix-moonshine-tokenizer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let tokenizer_path = dir.join("tokenizer.json");
    std::fs::write(&tokenizer_path, SYNTHETIC_VOCAB).unwrap();
    let relu = relu_fixture();
    // All four files load; the Relu graph is not a Moonshine encoder.
    let err = paths_error(&[relu.clone(), relu.clone(), relu, tokenizer_path]);
    assert!(err.contains("unexpected encoder inputs"));
}
