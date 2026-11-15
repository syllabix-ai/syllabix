//! Mock-session coverage for the Hima-ASR adapter. No weights, no devices.
//!
//! The production ONNX graphs are replaced with scripted [`MockSession`]s
//! behind the shared [`OnnxSession`](syllabix_native::onnx::OnnxSession) seam.

use super::*;
use crate::Stt;
use syllabix_native::onnx::{MockSession, OnnxData, OnnxSession, OnnxTensor};
use syllabix_native::types::{AudioFrame, TurnId, Utterance};
use syllabix_native::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};
use syllabix_native::{Cancel, Error};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const SYNTHETIC_VOCAB: &str = r#"{
    "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": {
        "<unk>": 0, "</s>": 1, "<s>": 2,
        "a": 3, "b": 4, "c": 7, "d": 8, "e": 9, "f": 11,
        "hello": 13, "world": 14
    }},
    "added_tokens": [
        {"id": 10, "content": "<english native>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 12, "content": "<english mixed>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
    ],
    "decoder": {"type": "Fuse"}
}"#;

fn names(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_string()).collect()
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_bytes(SYNTHETIC_VOCAB.as_bytes()).expect("synthetic vocab parses")
}

fn keep_ids() -> Vec<i64> {
    // Dense 0..15 so the synthetic tokenizer's remapped added-token ids (5, 6)
    // still sit in the keep table next to BOS=2, EOS=1, hello=13, world=14.
    (0..16).collect()
}

fn packed_embed(rows: usize) -> Vec<u16> {
    vec![0u16; rows * EMBED_DIM]
}

fn sot() -> Vec<f32> {
    vec![0.25; EMBED_DIM]
}

fn filters() -> Vec<f32> {
    load_mel_filters(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/hima/mel_filters.npy"),
    )
    .expect("fixture filters")
}

fn logits(top: usize, vocab: usize) -> OnnxTensor {
    let mut values = vec![-10.0_f32; vocab];
    values[top] = 10.0;
    OnnxTensor::f32(vec![vocab as i64], values)
}

fn kv(seq: i64) -> OnnxTensor {
    let n = KV_LAYERS * seq.max(0) as usize * KV_DIM;
    OnnxTensor::f32(
        vec![KV_LAYERS as i64, 1, seq.max(0), KV_DIM as i64],
        vec![0.0; n],
    )
}

fn hidden() -> OnnxTensor {
    OnnxTensor::f32(vec![19, 1, EMBED_DIM as i64], vec![0.0; 19 * EMBED_DIM])
}

fn mock_encoder() -> Box<dyn OnnxSession> {
    Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        move |inputs, _| {
            let n = inputs
                .iter()
                .find(|(name, _)| *name == "n_frames")
                .and_then(|(_, tensor)| match &tensor.data {
                    OnnxData::I64(values) => values.first().copied(),
                    _ => None,
                })
                .unwrap_or(7) as usize;
            let n = n.max(1);
            Ok(vec![OnnxTensor::f32(
                vec![n as i64, EMBED_DIM as i64],
                vec![0.1; n * EMBED_DIM],
            )])
        },
    ))
}

fn mock_decoder_script(tops: Vec<usize>) -> Box<dyn OnnxSession> {
    let calls = AtomicUsize::new(0);
    let vocab = keep_ids().len();
    Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |inputs, _| {
            let call = calls.fetch_add(1, Ordering::SeqCst);
            let top = tops.get(call).copied().unwrap_or(1);
            let seq = inputs
                .iter()
                .find(|(name, _)| *name == "past_k")
                .and_then(|(_, t)| t.shape.get(2).copied())
                .unwrap_or(0);
            let x_len = inputs
                .iter()
                .find(|(name, _)| *name == "x")
                .and_then(|(_, t)| t.shape.first().copied())
                .unwrap_or(1);
            Ok(vec![
                logits(top, vocab),
                kv(seq + x_len),
                kv(seq + x_len),
                hidden(),
            ])
        },
    ))
}

fn adapter(
    encoder: Box<dyn OnnxSession>,
    decoder: Box<dyn OnnxSession>,
    config: HimaConfig,
) -> HimaStt {
    let keep = keep_ids();
    let rows = keep.len();
    HimaStt::from_parts(
        encoder,
        decoder,
        HimaWeights {
            embed: packed_embed(rows),
            embed_rows: rows,
            keep_ids: keep,
            sot: sot(),
            filters: filters(),
            tokenizer: tokenizer(),
        },
        config,
    )
    .expect("mock contract is valid")
}

fn frame_with_samples(seq: u64, samples: Vec<i16>) -> AudioFrame {
    AudioFrame {
        seq,
        sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
        channels: DEFAULT_CHANNELS,
        samples,
        capture_pcm: None,
    }
}

fn utterance_from_pcm(turn: u64, pcm: &[i16]) -> Utterance {
    let mut frames = Vec::new();
    for (seq, chunk) in pcm.chunks(FRAME_SAMPLES).enumerate() {
        let mut samples = chunk.to_vec();
        if samples.len() < FRAME_SAMPLES {
            samples.resize(FRAME_SAMPLES, 0);
        }
        frames.push(frame_with_samples(seq as u64, samples));
    }
    if frames.is_empty() {
        frames.push(frame_with_samples(0, vec![0; FRAME_SAMPLES]));
    }
    Utterance {
        turn: TurnId(turn),
        frames,
    }
}

fn short_utterance(turn: u64) -> Utterance {
    utterance_from_pcm(turn, &vec![1000i16; MIN_SAMPLES])
}

#[test]
fn prompt_width_omits_tokens_when_language_is_none() {
    let stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        HimaConfig::default(),
    );
    assert_eq!(stt.prompt_width(7).unwrap(), 9);
    let en_native = adapter(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        HimaConfig {
            language: Some(HimaLanguage::En),
            style: HimaStyle::Native,
        },
    );
    // Added-token `"<english native>"` is one extra row after SOT.
    assert_eq!(en_native.prompt_width(7).unwrap(), 10);
    let en_mixed = en_native.with_config(HimaConfig {
        language: Some(HimaLanguage::En),
        style: HimaStyle::Mixed,
    });
    assert_eq!(en_mixed.prompt_width(7).unwrap(), 10);
    assert_eq!(
        language_prompt(HimaLanguage::En, HimaStyle::Mixed),
        "<english mixed>"
    );
}

#[test]
fn prompt_construction_is_visible_on_the_first_decoder_call() {
    let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |inputs, _| {
            if let Some((_, tensor)) = inputs.iter().find(|(name, _)| *name == "x") {
                log.lock().unwrap().push(tensor.shape[0]);
            }
            Ok(vec![logits(1, 16), kv(1), kv(1), hidden()])
        },
    ));
    let mut stt = adapter(
        mock_encoder(),
        decoder,
        HimaConfig {
            language: Some(HimaLanguage::En),
            style: HimaStyle::Native,
        },
    );
    stt.transcribe(&short_utterance(1), &Cancel::new())
        .expect("eos");
    let shapes = seen.lock().unwrap().clone();
    assert_eq!(shapes.first().copied(), Some(10));
}

#[test]
fn keep_ids_map_through_argmax() {
    // keep_ids[13] == 13 ("hello"); decoder peaks at 13 then EOS (row 1).
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![13, 1]),
        HimaConfig::default(),
    );
    let transcript = stt
        .transcribe(&short_utterance(3), &Cancel::new())
        .expect("hello");
    assert_eq!(transcript.text, "hello");
    assert_eq!(transcript.turn, TurnId(3));
    assert_eq!(transcript.language, "auto");
    assert!(!stt.supports_partials());
    assert_eq!(stt.name(), PROVIDER_NAME);
}

#[test]
fn decode_stops_at_eos() {
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        HimaConfig::default(),
    );
    let transcript = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .expect("immediate eos");
    assert_eq!(transcript.text, "");
}

#[test]
fn decode_stops_at_max_pieces() {
    // 0.5 s pad → n_sound=7, limit = min(512, 10+8)=18. Cycle letter tokens
    // so the repeat-guard never fires (period 6 needs 24 copies) and the budget does.
    let cycle: Vec<usize> = [3, 4, 7, 8, 9, 11].into_iter().cycle().take(40).collect();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let vocab = keep_ids().len();
    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |_, _| {
            let call = seen.fetch_add(1, Ordering::SeqCst);
            let top = cycle[call % cycle.len()];
            Ok(vec![logits(top, vocab), kv(8), kv(8), hidden()])
        },
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let transcript = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .expect("budget");
    assert!(!transcript.text.is_empty());
    // limit+1 decoder steps, last one still emits because the break is after
    // append when len >= limit (18 tokens, 18 calls).
    assert_eq!(calls.load(Ordering::SeqCst), 18);
}

#[test]
fn decode_stops_on_the_loop_guard() {
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![13; 8]),
        HimaConfig::default(),
    );
    let transcript = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .expect("loop cut");
    // Six "hello" ids trip size-1 × 6; _cut_loop leaves one copy.
    assert_eq!(transcript.text, "hello");
}

#[test]
fn decode_stops_on_token_budget_when_tokens_do_not_repeat() {
    // Distinct keep rows 2.. so loop_length stays 0. n_sound from a real
    // 0.5 s pad is 7, limit = min(512, 10+8) = 18.
    let cycle: Vec<usize> = [3, 4, 7, 8, 9, 11].into_iter().cycle().take(40).collect();
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(cycle),
        HimaConfig::default(),
    );
    let transcript = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .expect("budget");
    assert!(!transcript.text.is_empty());
}

#[test]
fn transcribe_observes_cancel_between_decode_steps() {
    let cancel = Cancel::new();
    let flag = cancel.clone();
    let calls = AtomicUsize::new(0);
    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |_, _| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                flag.shutdown();
            }
            Ok(vec![logits(13, 16), kv(1), kv(1), hidden()])
        },
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let err = stt.transcribe(&short_utterance(1), &cancel).unwrap_err();
    assert!(matches!(err, Error::Cancelled));

    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        HimaConfig::default(),
    );
    let dead = Cancel::new();
    dead.shutdown();
    assert!(matches!(
        stt.transcribe(&short_utterance(1), &dead),
        Err(Error::Cancelled)
    ));
}

#[test]
fn short_clip_path_pads_and_decodes() {
    let pcm = vec![500i16; (0.3 * SAMPLE_RATE_HZ as f32) as usize];
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![13, 1]),
        HimaConfig::default(),
    );
    let transcript = stt
        .transcribe(&utterance_from_pcm(9, &pcm), &Cancel::new())
        .expect("short");
    assert_eq!(transcript.text, "hello");
}

#[test]
fn long_clip_splits_and_joins_transcripts() {
    let encodes = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&encodes);
    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        move |inputs, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            let n = inputs
                .iter()
                .find(|(name, _)| *name == "n_frames")
                .and_then(|(_, tensor)| match &tensor.data {
                    OnnxData::I64(values) => values.first().copied(),
                    _ => None,
                })
                .unwrap_or(7) as usize;
            Ok(vec![OnnxTensor::f32(
                vec![n as i64, EMBED_DIM as i64],
                vec![0.1; n * EMBED_DIM],
            )])
        },
    ));
    let chunk = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&chunk);
    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |inputs, _| {
            let is_prompt = inputs
                .iter()
                .find(|(name, _)| *name == "x")
                .and_then(|(_, t)| t.shape.first().copied())
                .unwrap_or(1)
                > 1;
            if is_prompt {
                seen.fetch_add(1, Ordering::SeqCst);
            }
            let piece = seen.load(Ordering::SeqCst);
            let top = if is_prompt {
                if piece == 1 {
                    13
                } else {
                    14
                }
            } else {
                1
            };
            Ok(vec![logits(top, 16), kv(8), kv(8), hidden()])
        },
    ));
    let mut stt = adapter(encoder, decoder, HimaConfig::default());
    let pcm = vec![1000i16; MAX_CHUNK_SAMPLES + SAMPLE_RATE_HZ as usize];
    let transcript = stt
        .transcribe(&utterance_from_pcm(1, &pcm), &Cancel::new())
        .expect("long");
    assert_eq!(encodes.load(Ordering::SeqCst), 2);
    assert_eq!(transcript.text, "hello world");
}

#[test]
fn transcribe_rejects_empty_utterances() {
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        HimaConfig::default(),
    );
    let empty = Utterance {
        turn: TurnId(1),
        frames: vec![],
    };
    let err = stt.transcribe(&empty, &Cancel::new()).unwrap_err();
    assert!(err.to_string().contains("utterance has no frames"));
}

#[test]
fn missing_model_dir_fails_before_decode() {
    let err = match HimaStt::from_dir(
        std::path::Path::new("/no/such/hima-dir"),
        HimaConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("missing weights should fail"),
    };
    assert!(matches!(err, Error::Provider { .. }));
    assert!(err.to_string().contains("not found"), "{err}");
}

fn relu_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../syllabix-native/tests/fixtures/onnx/relu-tiny.onnx")
}

fn npy_header(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}\n");
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    let len = u16::try_from(dict.len()).expect("header");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend_from_slice(data);
    out
}

fn write_shared_tables(dir: &std::path::Path, embed_shape: &str, embed: &[u16], sot_len: usize) {
    let mut embed_bytes = Vec::new();
    for value in embed {
        embed_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(
        dir.join("embed.npy"),
        npy_header("<f2", embed_shape, &embed_bytes),
    )
    .unwrap();
    std::fs::write(
        dir.join("keep_ids.json"),
        serde_json::to_vec(&keep_ids()).unwrap(),
    )
    .unwrap();
    let mut sot_bytes = Vec::new();
    for _ in 0..sot_len {
        sot_bytes.extend_from_slice(&0f32.to_le_bytes());
    }
    std::fs::write(
        dir.join("start_of_text.npy"),
        npy_header("<f4", &format!("({sot_len},)"), &sot_bytes),
    )
    .unwrap();
    std::fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/hima/mel_filters.npy"),
        dir.join("mel_filters.npy"),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), SYNTHETIC_VOCAB).unwrap();
}

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "syllabix-hima-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn from_dir_err(dir: &std::path::Path) -> String {
    match HimaStt::from_dir(dir, HimaConfig::default()) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("from_dir should fail"),
    }
}

#[test]
fn from_dir_loads_relu_fixture_then_rejects_the_contract() {
    let dir = scratch_dir("direct");
    std::fs::copy(relu_fixture(), dir.join("encoder_model_int8.onnx")).unwrap();
    std::fs::copy(relu_fixture(), dir.join("decoder_model_int8.onnx")).unwrap();
    write_shared_tables(&dir, "(16, 640)", &vec![0u16; 16 * EMBED_DIM], EMBED_DIM);
    let err = from_dir_err(&dir);
    assert!(err.contains("unexpected encoder inputs"), "{err}");

    let nested = scratch_dir("nested");
    std::fs::create_dir_all(nested.join("onnx")).unwrap();
    std::fs::copy(
        relu_fixture(),
        nested.join("onnx").join("encoder_model_int8.onnx"),
    )
    .unwrap();
    std::fs::copy(
        relu_fixture(),
        nested.join("onnx").join("decoder_model_int8.onnx"),
    )
    .unwrap();
    write_shared_tables(&nested, "(16, 640)", &vec![0u16; 16 * EMBED_DIM], EMBED_DIM);
    let err = from_dir_err(&nested);
    assert!(err.contains("unexpected encoder inputs"), "{err}");

    let walked = scratch_dir("walk");
    let onnx = walked.join("onnx");
    std::fs::create_dir_all(&onnx).unwrap();
    std::fs::copy(relu_fixture(), onnx.join("encoder_model_int8.onnx")).unwrap();
    std::fs::copy(relu_fixture(), onnx.join("decoder_model_int8.onnx")).unwrap();
    write_shared_tables(&walked, "(16, 640)", &vec![0u16; 16 * EMBED_DIM], EMBED_DIM);
    let err = from_dir_err(&onnx);
    assert!(err.contains("unexpected encoder inputs"), "{err}");
}

#[test]
fn from_dir_rejects_wrong_embed_shape_and_missing_keep_ids() {
    let dir = scratch_dir("embed");
    std::fs::copy(relu_fixture(), dir.join("encoder_model_int8.onnx")).unwrap();
    std::fs::copy(relu_fixture(), dir.join("decoder_model_int8.onnx")).unwrap();
    write_shared_tables(&dir, "(2, 8)", &[0u16; 16], EMBED_DIM);
    let err = from_dir_err(&dir);
    assert!(err.contains("embed.npy shape"), "{err}");

    let missing = scratch_dir("nokeep");
    std::fs::copy(relu_fixture(), missing.join("encoder_model_int8.onnx")).unwrap();
    std::fs::copy(relu_fixture(), missing.join("decoder_model_int8.onnx")).unwrap();
    write_shared_tables(
        &missing,
        "(16, 640)",
        &vec![0u16; 16 * EMBED_DIM],
        EMBED_DIM,
    );
    std::fs::remove_file(missing.join("keep_ids.json")).unwrap();
    let err = from_dir_err(&missing);
    assert!(err.contains("keep_ids.json not found"), "{err}");
}

#[test]
fn unexpected_encoder_contract_is_rejected() {
    let encoder = Box::new(MockSession::script(
        names(&["wrong"]),
        names(&["sound"]),
        vec![],
    )) as Box<dyn OnnxSession>;
    let decoder = mock_decoder_script(vec![1]);
    let err = match HimaStt::from_parts(
        encoder,
        decoder,
        HimaWeights {
            embed: packed_embed(16),
            embed_rows: 16,
            keep_ids: keep_ids(),
            sot: sot(),
            filters: filters(),
            tokenizer: tokenizer(),
        },
        HimaConfig::default(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("bad contract should fail"),
    };
    assert!(err.to_string().contains("unexpected encoder inputs"));
}

fn parts_err(
    encoder: Box<dyn OnnxSession>,
    decoder: Box<dyn OnnxSession>,
    weights: HimaWeights,
) -> String {
    match HimaStt::from_parts(encoder, decoder, weights, HimaConfig::default()) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("from_parts should fail"),
    }
}

fn default_weights() -> HimaWeights {
    HimaWeights {
        embed: packed_embed(16),
        embed_rows: 16,
        keep_ids: keep_ids(),
        sot: sot(),
        filters: filters(),
        tokenizer: tokenizer(),
    }
}

#[test]
fn from_parts_rejects_bad_tables_and_contracts() {
    let err = parts_err(
        Box::new(MockSession::script(
            names(&["mel", "n_frames"]),
            names(&["layers"]),
            vec![],
        )),
        mock_decoder_script(vec![1]),
        default_weights(),
    );
    assert!(err.contains("encoder must emit sound"), "{err}");

    let err = parts_err(
        mock_encoder(),
        Box::new(MockSession::script(
            names(&["x", "pos0", "past_k"]),
            names(&["logits", "present_k", "present_v"]),
            vec![],
        )),
        default_weights(),
    );
    assert!(err.contains("unexpected decoder inputs"), "{err}");

    let err = parts_err(
        mock_encoder(),
        Box::new(MockSession::script(
            names(&["x", "pos0", "past_k", "past_v"]),
            names(&["hidden"]),
            vec![],
        )),
        default_weights(),
    );
    assert!(err.contains("decoder must emit logits"), "{err}");

    let mut weights = default_weights();
    weights.embed_rows = 3;
    let err = parts_err(mock_encoder(), mock_decoder_script(vec![1]), weights);
    assert!(err.contains("embed is"), "{err}");

    let mut weights = default_weights();
    weights.keep_ids = vec![1, 2];
    let err = parts_err(mock_encoder(), mock_decoder_script(vec![1]), weights);
    assert!(err.contains("keep_ids has"), "{err}");

    let mut weights = default_weights();
    weights.sot = vec![0.0; 8];
    let err = parts_err(mock_encoder(), mock_decoder_script(vec![1]), weights);
    assert!(err.contains("start_of_text"), "{err}");

    let mut weights = default_weights();
    weights.filters = vec![0.0; 4];
    let err = parts_err(mock_encoder(), mock_decoder_script(vec![1]), weights);
    assert!(err.contains("filterbank"), "{err}");

    let mut weights = default_weights();
    weights.keep_ids = (3..19).collect();
    let err = parts_err(mock_encoder(), mock_decoder_script(vec![1]), weights);
    assert!(err.contains("BOS=2 and EOS=1"), "{err}");
}

#[test]
fn transcribe_rejects_empty_samples_and_sets_language() {
    let mut stt = adapter(
        mock_encoder(),
        mock_decoder_script(vec![13, 1]),
        HimaConfig {
            language: Some(HimaLanguage::En),
            style: HimaStyle::Mixed,
        },
    );
    assert_eq!(stt.language(), Some(HimaLanguage::En));
    assert_eq!(stt.style(), HimaStyle::Mixed);
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

    let transcript = stt
        .transcribe(&short_utterance(2), &Cancel::new())
        .expect("en");
    assert_eq!(transcript.language, "en");
    assert_eq!(transcript.text, "hello");
    let stt = stt.with_config(HimaConfig::default());
    assert_eq!(stt.language(), None);
    assert_eq!(stt.style(), HimaStyle::Native);
}

#[test]
fn encoder_and_decoder_output_errors_surface() {
    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        |_, _| {
            Ok(vec![OnnxTensor::f32(
                vec![1, EMBED_DIM as i64],
                vec![0.0; EMBED_DIM],
            )])
        },
    ));
    let mut stt = adapter(encoder, mock_decoder_script(vec![1]), HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("expected at least"), "{err}");

    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        |_, _| {
            Ok(vec![OnnxTensor::i64(
                vec![7, EMBED_DIM as i64],
                vec![0; 7 * EMBED_DIM],
            )])
        },
    ));
    let mut stt = adapter(encoder, mock_decoder_script(vec![1]), HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("not f32"), "{err}");

    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        |_, _| {
            Ok(vec![OnnxTensor::f32(
                vec![1, EMBED_DIM as i64],
                vec![0.0; EMBED_DIM + 1],
            )])
        },
    ));
    let mut stt = adapter(encoder, mock_decoder_script(vec![1]), HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("not a multiple of"), "{err}");

    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        |_, _| Ok(vec![]),
    ));
    let mut stt = adapter(encoder, mock_decoder_script(vec![1]), HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("no sound"), "{err}");

    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        |_, _| Ok(vec![logits(1, 16)]),
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("too few outputs"), "{err}");

    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        |_, _| {
            Ok(vec![
                OnnxTensor::i64(vec![16], vec![0; 16]),
                kv(1),
                kv(1),
                hidden(),
            ])
        },
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("logits are not f32"), "{err}");

    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        |_, _| {
            Ok(vec![
                OnnxTensor::f32(vec![0], Vec::new()),
                kv(1),
                kv(1),
                hidden(),
            ])
        },
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("logits are not f32"), "{err}");

    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        |_, _| Ok(vec![logits(20, 21), kv(1), kv(1), hidden()]),
    ));
    let mut stt = adapter(mock_encoder(), decoder, HimaConfig::default());
    let err = stt
        .transcribe(&short_utterance(1), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("keep_ids missing row"), "{err}");
}

#[test]
fn cancel_between_chunks_is_honoured() {
    let cancel = Cancel::new();
    let flag = cancel.clone();
    let encodes = AtomicUsize::new(0);
    let encoder = Box::new(MockSession::with_handler(
        names(&["mel", "n_frames"]),
        names(&["sound", "layers"]),
        move |inputs, _| {
            let n = inputs
                .iter()
                .find(|(name, _)| *name == "n_frames")
                .and_then(|(_, tensor)| match &tensor.data {
                    OnnxData::I64(values) => values.first().copied(),
                    _ => None,
                })
                .unwrap_or(7) as usize;
            Ok(vec![OnnxTensor::f32(
                vec![n as i64, EMBED_DIM as i64],
                vec![0.1; n.max(1) * EMBED_DIM],
            )])
        },
    ));
    let decoder = Box::new(MockSession::with_handler(
        names(&["x", "pos0", "past_k", "past_v"]),
        names(&["logits", "present_k", "present_v", "hidden"]),
        move |inputs, _| {
            let is_prompt = inputs
                .iter()
                .find(|(name, _)| *name == "x")
                .and_then(|(_, t)| t.shape.first().copied())
                .unwrap_or(1)
                > 1;
            if !is_prompt {
                flag.shutdown();
            }
            let _ = encodes.fetch_add(1, Ordering::SeqCst);
            let top = if is_prompt { 13 } else { 1 };
            Ok(vec![logits(top, 16), kv(8), kv(8), hidden()])
        },
    ));
    let mut stt = adapter(encoder, decoder, HimaConfig::default());
    let pcm = vec![1000i16; MAX_CHUNK_SAMPLES + SAMPLE_RATE_HZ as usize];
    let err = stt
        .transcribe(&utterance_from_pcm(1, &pcm), &cancel)
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}

#[test]
fn prompt_token_missing_from_keep_ids_is_a_provider_error() {
    let mut weights = default_weights();
    weights.keep_ids = (0i64..16)
        .map(|i| if i == 1 || i == 2 { i } else { 1_000 + i })
        .collect();
    // Only BOS=2 and EOS=1 remain in the keep table.
    let stt = HimaStt::from_parts(
        mock_encoder(),
        mock_decoder_script(vec![1]),
        weights,
        HimaConfig {
            language: Some(HimaLanguage::En),
            style: HimaStyle::Native,
        },
    )
    .expect("tables otherwise valid");
    let err = stt.prompt_width(7).unwrap_err();
    assert!(err.to_string().contains("not in keep_ids"), "{err}");
}

fn fixture_frames(
    wav: &syllabix_native::audio::WavPcm,
) -> syllabix_native::Result<Vec<AudioFrame>> {
    use syllabix_native::audio::{f32_to_i16, i16_to_f32, FrameSplitter, PcmConverter, PcmFormat};
    let mut conv = PcmConverter::new(wav.format, PcmFormat::v0())?;
    let mut split = FrameSplitter::new();
    let mut mono = conv.push(&i16_to_f32(&wav.samples));
    mono.extend(conv.flush());
    let mut frames = split.push(&f32_to_i16(&mono))?;
    frames.extend(split.flush()?);
    Ok(frames)
}

/// Smoke test against a real int8 model directory. Does not claim Indic quality.
/// Gated by `SYLLABIX_HIMA_DIR` (directory of encoder/decoder int8 graphs plus
/// `embed.npy` / `keep_ids.json` / `start_of_text.npy` / `mel_filters.npy` /
/// `tokenizer.json`). Hima drops the opening "and so" on JFK, so the gate is
/// 0.8 word-match on every English fixture rather than a JFK-exact check.
#[test]
#[ignore]
fn native_recorded_fixtures_meet_word_match_gate() {
    let dir = std::env::var("SYLLABIX_HIMA_DIR")
        .expect("SYLLABIX_HIMA_DIR must point at the hima-asr-ONNX int8 directory");
    let mut stt = HimaStt::from_dir(
        std::path::Path::new(&dir),
        HimaConfig {
            language: Some(HimaLanguage::En),
            style: HimaStyle::Native,
        },
    )
    .expect("hima weights load");
    assert_eq!(stt.name(), PROVIDER_NAME);
    let fixtures: &[(&str, &[&str])] = &[
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
        ),
        (
            "librispeech-1089-134686-0000.wav",
            &[
                "he", "hoped", "there", "would", "be", "stew", "for", "dinner", "turnips", "and",
                "carrots", "and", "bruised", "potatoes", "and", "fat", "mutton", "pieces", "to",
                "be", "ladled", "out", "in", "thick", "peppered", "flour", "fattened", "sauce",
            ],
        ),
        (
            "librispeech-121-127105-0009.wav",
            &["she", "has", "been", "dead", "these", "twenty", "years"],
        ),
        (
            "librispeech-1995-1837-0005.wav",
            &[
                "she", "was", "so", "strange", "and", "human", "a", "creature",
            ],
        ),
    ];
    for (index, (file, expected)) in fixtures.iter().enumerate() {
        let wav = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../syllabix-core/tests/fixtures/stt")
                .join(file),
        )
        .unwrap_or_else(|err| panic!("{file} fixture: {err}"));
        let pcm = syllabix_native::audio::read_wav(std::io::Cursor::new(wav))
            .unwrap_or_else(|err| panic!("{file} parses: {err}"));
        let frames = fixture_frames(&pcm).unwrap_or_else(|err| panic!("{file} frames: {err}"));
        let audio_secs = pcm.samples.len() as f64 / f64::from(pcm.format.sample_rate_hz);
        let started = std::time::Instant::now();
        let transcript = stt
            .transcribe(
                &Utterance {
                    turn: TurnId(index as u64),
                    frames,
                },
                &Cancel::new(),
            )
            .unwrap_or_else(|err| panic!("{file} transcribes: {err}"));
        let elapsed = started.elapsed().as_secs_f64().max(1e-9);
        let ratio = crate::word_match_ratio(&transcript.text, expected);
        let rtf = audio_secs / elapsed;
        eprintln!(
            "Hima {file}: {:.1}% word match, RTF {rtf:.1}x ({:?})",
            ratio * 100.0,
            transcript.text
        );
        assert!(
            ratio >= 0.8,
            "{file} transcript {:?} matched {:.1}% of {:?} (need 80%)",
            transcript.text,
            ratio * 100.0,
            expected
        );
        assert_eq!(transcript.language, "en");
    }
}
