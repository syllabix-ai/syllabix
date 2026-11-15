//! Extra mock-session coverage for [`super::FluidOnnxStt`].

use super::*;
use crate::Stt;
use syllabix_native::onnx::{MockSession, OnnxTensor};
use syllabix_native::types::{AudioFrame, TurnId, Utterance};
use syllabix_native::Cancel;

fn names(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| (*w).to_string()).collect()
}

fn tiny_vocab() -> Vec<String> {
    let mut v = vec![String::new(); TOKEN_LOGITS];
    v[5] = " test".into();
    v[BLANK as usize] = "<blk>".into();
    v
}

fn states() -> OnnxTensor {
    OnnxTensor::f32(
        vec![STATE_LAYERS, 1, STATE_HIDDEN],
        vec![0.0; (STATE_LAYERS * STATE_HIDDEN) as usize],
    )
}

fn logits_tok_dur(tok: usize, dur: usize) -> OnnxTensor {
    let mut values = vec![-15.0f32; TOTAL_LOGITS];
    values[tok] = 12.0;
    values[TOKEN_LOGITS + dur] = 12.0;
    OnnxTensor::f32(vec![1, 1, 1, TOTAL_LOGITS as i64], values)
}

#[test]
fn ten_symbol_cap_advances_time() {
    let mut decoder_steps = Vec::new();
    for _ in 0..10 {
        decoder_steps.push(Ok(vec![
            logits_tok_dur(5, 0),
            OnnxTensor::i32(vec![1], vec![1]),
            states(),
            states(),
        ]));
    }
    // Eleventh emission on the same frame is blank so time steps forward.
    decoder_steps.push(Ok(vec![
        logits_tok_dur(BLANK as usize, 0),
        OnnxTensor::i32(vec![1], vec![1]),
        states(),
        states(),
    ]));
    decoder_steps.push(Ok(vec![
        logits_tok_dur(BLANK as usize, 0),
        OnnxTensor::i32(vec![1], vec![1]),
        states(),
        states(),
    ]));
    let encoder = MockSession::script(
        names(&["audio_signal", "length"]),
        names(&["outputs", "encoded_lengths"]),
        vec![Ok(vec![
            OnnxTensor::f32(vec![1, ENCODER_DIM, 2], vec![0.0; 2 * ENCODER_DIM as usize]),
            OnnxTensor::i64(vec![1], vec![2]),
        ])],
    );
    let decoder = MockSession::script(
        names(&[
            "encoder_outputs",
            "targets",
            "target_length",
            "input_states_1",
            "input_states_2",
        ]),
        names(&[
            "outputs",
            "prednet_lengths",
            "output_states_1",
            "output_states_2",
        ]),
        decoder_steps,
    );
    let mut stt = FluidOnnxStt::from_parts(Box::new(encoder), Box::new(decoder), tiny_vocab(), 1);
    let text = stt
        .transcribe_pcm(&vec![0.0; 4800], &Cancel::new())
        .expect("capped decode");
    assert_eq!(text.matches("test").count(), 10);
}

#[test]
fn utterance_finalize_path() {
    let encoder = MockSession::script(
        names(&["audio_signal", "length"]),
        names(&["outputs", "encoded_lengths"]),
        vec![Ok(vec![
            OnnxTensor::f32(vec![1, ENCODER_DIM, 1], vec![0.0; ENCODER_DIM as usize]),
            OnnxTensor::i64(vec![1], vec![1]),
        ])],
    );
    let decoder = MockSession::script(
        names(&[
            "encoder_outputs",
            "targets",
            "target_length",
            "input_states_1",
            "input_states_2",
        ]),
        names(&[
            "outputs",
            "prednet_lengths",
            "output_states_1",
            "output_states_2",
        ]),
        vec![Ok(vec![
            logits_tok_dur(BLANK as usize, 0),
            OnnxTensor::i32(vec![1], vec![1]),
            states(),
            states(),
        ])],
    );
    let mut stt = FluidOnnxStt::from_parts(Box::new(encoder), Box::new(decoder), tiny_vocab(), 1);
    let utterance = Utterance {
        turn: TurnId(2),
        frames: vec![AudioFrame {
            seq: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![100; 320],
            capture_pcm: None,
            flush_pad_samples: 0,
        }],
    };
    let transcript = stt.transcribe(&utterance, &Cancel::new()).expect("ok");
    assert_eq!(transcript.turn, TurnId(2));
    assert!(!stt.supports_partials());
}

#[test]
fn name_is_fluid_onnx() {
    let stt = FluidOnnxStt::from_parts(
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        tiny_vocab(),
        1,
    );
    assert_eq!(stt.name(), PROVIDER_NAME);
    assert_eq!(stt.language(), crate::language::LANGUAGE_AUTO);
    assert_eq!(stt.intra_op_threads(), 1);
}

#[test]
fn pcm_without_frame_splitter_pad_strips_only_flush_suffix() {
    use syllabix_native::audio::FrameSplitter;
    use syllabix_native::types::FRAME_SAMPLES;
    let wav_len = 70_480i32;
    let pcm: Vec<i16> = (0..wav_len)
        .map(|i| ((i % 97) as i16).wrapping_mul(120))
        .collect();
    let mut split = FrameSplitter::new();
    let mut frames = split.push(&pcm).expect("push");
    frames.extend(split.flush().expect("flush"));
    let utterance = Utterance {
        turn: TurnId(0),
        frames,
    };
    let trimmed = utterance.pcm_without_splitter_flush_pad();
    let pad = FRAME_SAMPLES - (wav_len as usize % FRAME_SAMPLES);
    assert_eq!(trimmed.len(), wav_len as usize);
    assert_eq!(trimmed, pcm);
    assert_eq!(pad, 176);
    assert_eq!(
        utterance.frames.last().unwrap().flush_pad_samples as usize,
        pad
    );
}

#[test]
fn pcm_without_frame_splitter_pad_keeps_even_frame_trailing_speech_zeros() {
    use syllabix_native::audio::FrameSplitter;
    use syllabix_native::types::FRAME_SAMPLES;
    let n = 2 * FRAME_SAMPLES;
    let mut pcm = vec![42i16; n];
    for sample in pcm.iter_mut().skip(n - 64) {
        *sample = 0;
    }
    let mut split = FrameSplitter::new();
    let mut frames = split.push(&pcm).expect("push");
    frames.extend(split.flush().expect("flush"));
    let utterance = Utterance {
        turn: TurnId(0),
        frames,
    };
    assert_eq!(utterance.frames.last().unwrap().flush_pad_samples, 0);
    let trimmed = utterance.pcm_without_splitter_flush_pad();
    assert_eq!(trimmed.len(), n);
    assert_eq!(trimmed, pcm);
}

fn relu_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../syllabix-native/tests/fixtures/onnx/relu-tiny.onnx")
}

fn missing() -> std::path::PathBuf {
    std::path::PathBuf::from("/no/such/fluid-graph.onnx")
}

#[test]
fn set_intra_threads_is_observed_via_from_parts() {
    set_intra_threads(3);
    let stt = FluidOnnxStt::from_parts(
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        tiny_vocab(),
        3,
    );
    assert_eq!(stt.intra_op_threads(), 3);
    set_intra_threads(0);
}

#[test]
fn from_cache_reports_a_missing_manifest_asset() {
    use syllabix_native::models::{Manifest, ModelCache};
    let root =
        std::env::temp_dir().join(format!("fluid-cache-{}-{}", std::process::id(), rand_dir()));
    std::fs::create_dir_all(&root).unwrap();
    let cache = ModelCache::new(
        root,
        Manifest {
            version: 1,
            assets: vec![],
        },
    );
    let err = match FluidOnnxStt::from_cache(
        &cache,
        &syllabix_native::models::BlockedFetcher::default(),
        &mut syllabix_native::models::NoProgress,
        &Cancel::new(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("empty manifest should fail"),
    };
    assert!(err.to_string().contains(ENCODER_ASSET));
}

#[test]
fn from_cache_surfaces_unresolvable_assets() {
    use syllabix_native::models::{Manifest, ModelAsset, ModelCache, ModelLayer};
    let root = std::env::temp_dir().join(format!(
        "fluid-blocked-{}-{}",
        std::process::id(),
        rand_dir()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let assets = [
        ENCODER_ASSET,
        ENCODER_DATA_ASSET,
        DECODER_JOINT_ASSET,
        DECODER_JOINT_DATA_ASSET,
        VOCAB_ASSET,
    ]
    .iter()
    .map(|id| ModelAsset {
        id: (*id).to_string(),
        layer: ModelLayer::Stt,
        file_name: format!("{id}.bin"),
        url: "https://example.invalid/model.bin".to_string(),
        sha256: "0".repeat(64),
        size_bytes: 1,
    })
    .collect();
    let cache = ModelCache::new(root, Manifest { version: 1, assets });
    let err = match FluidOnnxStt::from_cache(
        &cache,
        &syllabix_native::models::BlockedFetcher::default(),
        &mut syllabix_native::models::NoProgress,
        &Cancel::new(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("blocked fetch should fail"),
    };
    assert!(!err.to_string().is_empty());
}

fn rand_dir() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

#[test]
fn from_paths_reports_missing_graphs_and_vocab() {
    let relu = relu_fixture();
    let err = match FluidOnnxStt::from_paths(relu.clone(), missing(), missing(), 1) {
        Err(err) => err,
        Ok(_) => panic!("missing decoder_joint should fail"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("could not load") || msg.contains("No such file"),
        "{msg}"
    );
    let dir = std::env::temp_dir().join(format!("fluid-paths-{}", rand_dir()));
    std::fs::create_dir_all(&dir).unwrap();
    let err = match FluidOnnxStt::from_paths(relu.clone(), relu, dir.join("missing-vocab.txt"), 1) {
        Err(err) => err,
        Ok(_) => panic!("missing vocab should fail"),
    };
    assert!(err.to_string().contains("could not read"));
}

#[test]
fn transcribe_rejects_empty_frames_and_all_silence() {
    let encoder = MockSession::script(
        names(&["audio_signal", "length"]),
        names(&["outputs", "encoded_lengths"]),
        vec![],
    );
    let decoder = MockSession::script(names(&["encoder_outputs"]), names(&["outputs"]), vec![]);
    let mut stt = FluidOnnxStt::from_parts(Box::new(encoder), Box::new(decoder), tiny_vocab(), 1);
    let empty = Utterance {
        turn: TurnId(0),
        frames: vec![],
    };
    assert!(stt.transcribe(&empty, &Cancel::new()).is_err());
    let silence = Utterance {
        turn: TurnId(1),
        frames: vec![AudioFrame {
            seq: 0,
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0; 320],
            capture_pcm: None,
            flush_pad_samples: 0,
        }],
    };
    assert!(stt.transcribe(&silence, &Cancel::new()).is_err());
}

#[test]
fn with_language_round_trips() {
    let stt = FluidOnnxStt::from_parts(
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        Box::new(MockSession::script(names(&["a"]), names(&["b"]), vec![])),
        tiny_vocab(),
        2,
    )
    .with_language("fr")
    .expect("fr");
    assert_eq!(stt.language(), "fr");
    assert!(validate_stt_language("zz").is_err());
}

#[test]
fn encoder_and_decoder_errors_surface() {
    let encoder = MockSession::script(
        names(&["audio_signal", "length"]),
        names(&["outputs", "encoded_lengths"]),
        vec![Err(syllabix_native::error::Error::Provider {
            provider: "mock",
            message: "encoder down".into(),
        })],
    );
    let decoder = MockSession::script(names(&["x"]), names(&["y"]), vec![]);
    let mut stt = FluidOnnxStt::from_parts(Box::new(encoder), Box::new(decoder), tiny_vocab(), 1);
    assert!(stt
        .transcribe_pcm(&vec![0.0; 8000], &Cancel::new())
        .unwrap_err()
        .to_string()
        .contains("encoder down"));
}
