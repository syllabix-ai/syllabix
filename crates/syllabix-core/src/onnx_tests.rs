//! Unit tests for the shared ONNX seam. No weights, no devices.

use super::*;
use crate::Error;

fn names(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_string()).collect()
}

#[test]
fn tensor_constructors_round_trip_shapes_and_data() {
    let f32_tensor = OnnxTensor::f32(vec![1, 2], vec![1.0, 2.0]);
    assert_eq!(f32_tensor.shape, vec![1, 2]);
    assert_eq!(f32_tensor.f32_data(), Some(&[1.0, 2.0][..]));
    let i64_tensor = OnnxTensor::i64(vec![1], vec![7]);
    assert!(i64_tensor.f32_data().is_none());
    assert!(matches!(i64_tensor.data, OnnxData::I64(_)));
    let bool_tensor = OnnxTensor::bool(vec![2], vec![1, 0]);
    assert!(bool_tensor.f32_data().is_none());
    assert!(matches!(bool_tensor.data, OnnxData::Bool(_)));
}

#[test]
fn mock_reports_its_graph_contract() {
    let session = MockSession::script(names(&["a"]), names(&["b"]), vec![]);
    assert_eq!(session.input_names(), vec!["a".to_string()]);
    assert_eq!(session.output_names(), vec!["b".to_string()]);
    assert!(session.calls.is_empty());
}

#[test]
fn script_replies_in_order_then_reports_exhaustion() {
    let mut session = MockSession::script(
        names(&["in"]),
        names(&["out"]),
        vec![
            Ok(vec![OnnxTensor::f32(vec![1], vec![1.0])]),
            Err(Error::Cancelled),
        ],
    );
    let input = OnnxTensor::f32(vec![1], vec![0.0]);
    let first = session.run(&[("in", &input)], &["out"]).unwrap();
    assert_eq!(first[0].f32_data(), Some(&[1.0][..]));
    assert!(matches!(
        session.run(&[("in", &input)], &["out"]),
        Err(Error::Cancelled)
    ));
    let exhausted = session.run(&[("in", &input)], &["out"]).unwrap_err();
    assert!(exhausted.to_string().contains("mock script exhausted"));
    assert_eq!(session.calls.len(), 3);
    assert_eq!(session.calls[0], vec!["in".to_string()]);
}

#[test]
fn handler_sees_inputs_and_outputs() {
    let mut session =
        MockSession::with_handler(names(&["in"]), names(&["out"]), |inputs, outputs| {
            assert_eq!(outputs, &["out"]);
            let tensor = inputs
                .iter()
                .find(|(name, _)| *name == "in")
                .map(|(_, tensor)| (*tensor).clone())
                .expect("handler sees the named input");
            Ok(vec![tensor])
        });
    let input = OnnxTensor::i64(vec![1, 1], vec![3]);
    let echoed = session.run(&[("in", &input)], &["out"]).unwrap();
    assert!(matches!(echoed[0].data, OnnxData::I64(_)));
}

#[test]
fn failing_mock_reports_the_configured_message() {
    let mut session = MockSession::failing(names(&["in"]), names(&["out"]), "boom");
    let input = OnnxTensor::f32(vec![1], vec![0.0]);
    let err = session.run(&[("in", &input)], &["out"]).unwrap_err();
    assert!(err.to_string().contains("boom"));
}

#[test]
fn ort_load_rejects_a_missing_graph() {
    let err = match OrtSession::load(
        Path::new("/no/such/graph.onnx"),
        "mock-provider",
        "mock graph",
    ) {
        Err(err) => err,
        Ok(_) => panic!("missing graph should fail"),
    };
    assert!(matches!(err, Error::Provider { .. }));
    assert!(err.to_string().contains("mock graph"));
}

#[test]
fn null_status_is_success() {
    check(ptr::null_mut()).expect("null status means success");
}

fn relu_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/onnx/relu-tiny.onnx")
}

fn load_relu() -> OrtSession {
    OrtSession::load(&relu_fixture(), "mock-provider", "relu fixture")
        .expect("checked-in fixture loads")
}

#[test]
fn ort_session_reports_fixture_contract() {
    let session = load_relu();
    assert_eq!(session.input_names(), vec!["x".to_string()]);
    assert_eq!(session.output_names(), vec!["y".to_string()]);
}

#[test]
fn ort_session_runs_the_fixture_end_to_end() {
    let mut session = load_relu();
    let input = OnnxTensor::f32(vec![1, 2], vec![-1.0, 2.0]);
    let outputs = session.run(&[("x", &input)], &["y"]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![1, 2]);
    assert_eq!(outputs[0].f32_data(), Some(&[0.0, 2.0][..]));
}

#[test]
fn ort_session_rejects_nul_names_before_touching_weights() {
    let mut session = load_relu();
    let input = OnnxTensor::f32(vec![1, 2], vec![1.0, 2.0]);
    let err = session.run(&[("x\0", &input)], &["y"]).unwrap_err();
    assert!(err.to_string().contains("invalid input name"));
    let err = session.run(&[("x", &input)], &["y\0"]).unwrap_err();
    assert!(err.to_string().contains("invalid output name"));
}

#[test]
fn ort_session_surfaces_runtime_shape_errors() {
    let mut session = load_relu();
    let input = OnnxTensor::f32(vec![1, 3], vec![1.0, 2.0, 3.0]);
    let err = session.run(&[("x", &input)], &["y"]).unwrap_err();
    assert!(matches!(err, Error::Provider { .. }));
}

fn mixed_fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/onnx/mixed-types.onnx")
}

#[test]
fn ort_session_reads_every_element_type() {
    let mut session = OrtSession::load(&mixed_fixture(), "mock-provider", "mixed fixture")
        .expect("fixture loads");
    assert_eq!(
        session.output_names(),
        vec![
            "y_f32".to_string(),
            "y_i64".to_string(),
            "y_bool".to_string()
        ]
    );
    let input = OnnxTensor::f32(vec![1, 2], vec![-1.0, 2.5]);
    let outputs = session
        .run(&[("x", &input)], &["y_f32", "y_i64", "y_bool"])
        .expect("mixed run");
    assert_eq!(outputs[0].f32_data(), Some(&[0.0, 2.5][..]));
    assert!(matches!(outputs[1].data, OnnxData::I64(_)));
    assert!(matches!(outputs[2].data, OnnxData::Bool(_)));
}

#[test]
fn ort_session_builds_non_float_inputs() {
    let mut session = load_relu();
    // The graph rejects the mistyped input at Run, after the C-API value is built.
    let int_input = OnnxTensor::i64(vec![1, 2], vec![1, 2]);
    assert!(session.run(&[("x", &int_input)], &["y"]).is_err());
    let bool_input = OnnxTensor::bool(vec![1, 2], vec![1, 0]);
    assert!(session.run(&[("x", &bool_input)], &["y"]).is_err());
    // Empty tensors skip the host-to-device copy (Run still validates dims).
    let empty = OnnxTensor::f32(vec![1, 0], vec![]);
    assert!(session.run(&[("x", &empty)], &["y"]).is_err());
}
