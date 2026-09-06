//! Shared ONNX Runtime inference seam for file-level coverage.
//!
//! Pocket TTS and Moonshine both drive ONNX graphs through the same two
//! mechanisms: fixed-shape runs (Pocket text conditioner) and dynamic-shape
//! runs through the C API (recurrent state, KV cache, variable-length audio).
//! [`OnnxSession`] abstracts both so unit tests can inject [`MockSession`]
//! instead of downloading weights. [`OrtSession`] is the production wrapper
//! over `ort::session::Session`; a 75-byte checked-in Relu fixture exercises
//! its full load/run path with no downloads (see `onnx_tests.rs`).

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    path::Path,
    ptr,
};

use ort::{session::Session, AsPointer};

use crate::{Error, Result};

/// Owned dense tensor exchanged with an [`OnnxSession`].
#[derive(Clone, Debug)]
pub(crate) struct OnnxTensor {
    pub(crate) shape: Vec<i64>,
    pub(crate) data: OnnxData,
}

/// Element storage for [`OnnxTensor`].
#[derive(Clone, Debug)]
pub(crate) enum OnnxData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bool(Vec<u8>),
}

impl OnnxTensor {
    pub(crate) fn f32(shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            shape,
            data: OnnxData::F32(data),
        }
    }

    pub(crate) fn i64(shape: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            shape,
            data: OnnxData::I64(data),
        }
    }

    pub(crate) fn bool(shape: Vec<i64>, data: Vec<u8>) -> Self {
        Self {
            shape,
            data: OnnxData::Bool(data),
        }
    }

    /// Borrow float elements, or `None` when the tensor holds another type.
    pub(crate) fn f32_data(&self) -> Option<&[f32]> {
        match &self.data {
            OnnxData::F32(data) => Some(data),
            _ => None,
        }
    }
}

/// Narrow inference seam: run named tensors, inspect graph I/O names.
///
/// Production code uses [`OrtSession`]; coverage tests inject [`MockSession`].
/// Neither side downloads weights through this trait.
pub(crate) trait OnnxSession: Send {
    fn input_names(&self) -> Vec<String>;
    fn output_names(&self) -> Vec<String>;
    fn run(&mut self, inputs: &[(&str, &OnnxTensor)], outputs: &[&str]) -> Result<Vec<OnnxTensor>>;
}

/// Production wrapper over `ort::session::Session`.
///
/// The C-API path handles dynamic-shape tensors (recurrent state, KV cache,
/// variable-length audio) that the high-level `ort::inputs!` macro cannot
/// name. Fixed-shape graphs run through the same entry point.
pub(crate) struct OrtSession {
    session: Session,
}

impl OrtSession {
    pub(crate) fn load(path: &Path, provider: &'static str, label: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|err| Error::Provider {
                provider,
                message: format!("ONNX Runtime: {err}"),
            })?
            .commit_from_file(path)
            .map_err(|err| Error::Provider {
                provider,
                message: format!("could not load {label} {}: {err}", path.display()),
            })?;
        Ok(Self { session })
    }
}

impl OnnxSession for OrtSession {
    fn input_names(&self) -> Vec<String> {
        self.session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    fn output_names(&self) -> Vec<String> {
        self.session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    // The C-API path handles dynamic-shape tensors (recurrent state, KV
    // cache, variable-length audio) that the high-level `ort::inputs!` macro
    // cannot name. A checked-in Relu fixture covers this path with no weights.
    fn run(&mut self, inputs: &[(&str, &OnnxTensor)], outputs: &[&str]) -> Result<Vec<OnnxTensor>> {
        let mut allocator = ptr::null_mut();
        check(unsafe { ort::ortsys!(GetAllocatorWithDefaultOptions)(&mut allocator) })?;
        let names = inputs
            .iter()
            .map(|(name, _)| CString::new(*name))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::Provider {
                provider: "onnx",
                message: "invalid input name".into(),
            })?;
        let out_names = outputs
            .iter()
            .map(|name| CString::new(*name))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::Provider {
                provider: "onnx",
                message: "invalid output name".into(),
            })?;
        let values = inputs
            .iter()
            .map(|(_, tensor)| create_value(allocator, tensor))
            .collect::<Result<Vec<_>>>()?;
        let in_names = names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<*const c_char>>();
        let in_values = values
            .iter()
            .map(|value| value.0.cast_const())
            .collect::<Vec<_>>();
        let out_ptrs = out_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut raw_outputs = vec![ptr::null_mut(); outputs.len()];
        check(unsafe {
            ort::ortsys!(Run)(
                self.session.ptr_mut(),
                ptr::null(),
                in_names.as_ptr(),
                in_values.as_ptr(),
                in_values.len(),
                out_ptrs.as_ptr(),
                out_ptrs.len(),
                raw_outputs.as_mut_ptr(),
            )
        })?;
        raw_outputs
            .into_iter()
            .map(|ptr| read_value(RawValue(ptr)))
            .collect()
    }
}

struct RawValue(*mut ort_sys::OrtValue);
// The value is freed on drop; individual reads copy out before that.
impl Drop for RawValue {
    fn drop(&mut self) {
        unsafe { ort::ortsys!(ReleaseValue)(self.0) };
    }
}

// Covered through the checked-in Relu fixture (no weight downloads).
fn create_value(allocator: *mut ort_sys::OrtAllocator, tensor: &OnnxTensor) -> Result<RawValue> {
    use ort_sys::ONNXTensorElementDataType as ElementType;
    let (element_type, byte_len, source): (ElementType, usize, *const c_void) = match &tensor.data {
        OnnxData::F32(data) => (
            ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            data.len() * size_of::<f32>(),
            data.as_ptr().cast(),
        ),
        OnnxData::I64(data) => (
            ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            data.len() * size_of::<i64>(),
            data.as_ptr().cast(),
        ),
        OnnxData::Bool(data) => (
            ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            data.len(),
            data.as_ptr().cast(),
        ),
    };
    let mut value = ptr::null_mut();
    check(unsafe {
        ort::ortsys!(CreateTensorAsOrtValue)(
            allocator,
            tensor.shape.as_ptr(),
            tensor.shape.len(),
            element_type,
            &mut value,
        )
    })?;
    let value = RawValue(value);
    if byte_len != 0 {
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { ort::ortsys!(GetTensorMutableData)(value.0, &mut data) })?;
        unsafe {
            ptr::copy_nonoverlapping(source.cast::<u8>(), data.cast::<u8>(), byte_len);
        }
    }
    Ok(value)
}

// Covered through the checked-in Relu fixture (no weight downloads).
fn read_value(value: RawValue) -> Result<OnnxTensor> {
    use ort_sys::ONNXTensorElementDataType as ElementType;
    let mut info = ptr::null_mut();
    check(unsafe { ort::ortsys!(GetTensorTypeAndShape)(value.0, &mut info) })?;
    let mut rank = 0;
    check(unsafe { ort::ortsys!(GetDimensionsCount)(info, &mut rank) })?;
    let mut element_type = ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check(unsafe { ort::ortsys!(GetTensorElementType)(info, &mut element_type) })?;
    let mut shape = vec![0_i64; rank];
    check(unsafe { ort::ortsys!(GetDimensions)(info, shape.as_mut_ptr(), rank) })?;
    unsafe { ort::ortsys!(ReleaseTensorTypeAndShapeInfo)(info) };
    let len = shape
        .iter()
        .try_fold(1_usize, |count, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| count.checked_mul(dim))
        })
        .ok_or_else(|| Error::Provider {
            provider: "onnx",
            message: "invalid output shape".into(),
        })?;
    let mut data: *mut c_void = ptr::null_mut();
    if len != 0 {
        check(unsafe { ort::ortsys!(GetTensorMutableData)(value.0, &mut data) })?;
    }
    match element_type {
        ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
            let mut out = vec![0_f32; len];
            if len != 0 {
                unsafe {
                    ptr::copy_nonoverlapping(data.cast::<f32>(), out.as_mut_ptr(), len);
                }
            }
            Ok(OnnxTensor::f32(shape, out))
        }
        ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
            let mut out = vec![0_i64; len];
            if len != 0 {
                unsafe {
                    ptr::copy_nonoverlapping(data.cast::<i64>(), out.as_mut_ptr(), len);
                }
            }
            Ok(OnnxTensor::i64(shape, out))
        }
        ElementType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => {
            let mut out = vec![0_u8; len];
            if len != 0 {
                unsafe {
                    ptr::copy_nonoverlapping(data.cast::<u8>(), out.as_mut_ptr(), len);
                }
            }
            Ok(OnnxTensor::bool(shape, out))
        }
        _ => Err(Error::Provider {
            provider: "onnx",
            message: "unsupported ONNX output type".into(),
        }),
    }
}

pub(crate) fn check(status: ort_sys::OrtStatusPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let message = unsafe {
        CStr::from_ptr(ort::ortsys!(GetErrorMessage)(status))
            .to_string_lossy()
            .into_owned()
    };
    unsafe { ort::ortsys!(ReleaseStatus)(status) };
    Err(Error::Provider {
        provider: "onnx",
        message: format!("ONNX Runtime C API: {message}"),
    })
}

/// Scripted [`OnnxSession`] for coverage tests. No weights, no devices.
///
/// Either reply from a per-call script ([`MockSession::script`]) or from a
/// stateful handler ([`MockSession::with_handler`]). Every call records its
/// input names so tests can assert graph wiring.
#[cfg(test)]
pub(crate) struct MockSession {
    inputs: Vec<String>,
    outputs: Vec<String>,
    behavior: MockBehavior,
    /// Input names observed on each `run`, in call order.
    pub(crate) calls: Vec<Vec<String>>,
}

#[cfg(test)]
type RunHandler = Box<dyn FnMut(&[(&str, &OnnxTensor)], &[&str]) -> Result<Vec<OnnxTensor>> + Send>;

#[cfg(test)]
enum MockBehavior {
    Script(std::collections::VecDeque<Result<Vec<OnnxTensor>>>),
    Handler(RunHandler),
}

#[cfg(test)]
impl MockSession {
    /// Reply with one scripted output vector per `run`, in order.
    pub(crate) fn script(
        inputs: Vec<String>,
        outputs: Vec<String>,
        steps: Vec<Result<Vec<OnnxTensor>>>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            behavior: MockBehavior::Script(steps.into()),
            calls: Vec::new(),
        }
    }

    /// Reply from a stateful closure (loops, EOS-vs-frame logic, cancels).
    pub(crate) fn with_handler(
        inputs: Vec<String>,
        outputs: Vec<String>,
        handler: impl FnMut(&[(&str, &OnnxTensor)], &[&str]) -> Result<Vec<OnnxTensor>> + Send + 'static,
    ) -> Self {
        Self {
            inputs,
            outputs,
            behavior: MockBehavior::Handler(Box::new(handler)),
            calls: Vec::new(),
        }
    }

    /// Fail every `run` with a provider error (contract/cancel paths).
    pub(crate) fn failing(
        inputs: Vec<String>,
        outputs: Vec<String>,
        message: impl Into<String>,
    ) -> Self {
        let message = message.into();
        Self::with_handler(inputs, outputs, move |_, _| {
            Err(Error::Provider {
                provider: "mock-onnx",
                message: message.clone(),
            })
        })
    }
}

#[cfg(test)]
impl OnnxSession for MockSession {
    fn input_names(&self) -> Vec<String> {
        self.inputs.clone()
    }

    fn output_names(&self) -> Vec<String> {
        self.outputs.clone()
    }

    fn run(&mut self, inputs: &[(&str, &OnnxTensor)], outputs: &[&str]) -> Result<Vec<OnnxTensor>> {
        self.calls
            .push(inputs.iter().map(|(name, _)| (*name).to_string()).collect());
        match &mut self.behavior {
            MockBehavior::Script(steps) => steps.pop_front().unwrap_or_else(|| {
                Err(Error::Provider {
                    provider: "mock-onnx",
                    message: "mock script exhausted".into(),
                })
            }),
            MockBehavior::Handler(handler) => handler(inputs, outputs),
        }
    }
}

#[cfg(test)]
#[path = "onnx_tests.rs"]
mod onnx_tests;
