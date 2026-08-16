//! Shared `ggml` plus the whisper.cpp and llama.cpp frontends.
//!
//! This crate does **not** load a GGUF or generate tokens. That is PR 11.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::Once;

mod ffi {
    use super::*;

    #[repr(C)]
    pub struct WhisperContext {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn syllabix_native_hush_logs();
        pub fn syllabix_native_link_anchor() -> c_int;
        pub fn syllabix_llama_system_info() -> *const c_char;
        pub fn syllabix_llama_backend_init();
        pub fn syllabix_llama_backend_free();
        pub fn syllabix_whisper_load(path: *const c_char) -> *mut WhisperContext;
        pub fn syllabix_whisper_free(ctx: *mut WhisperContext);
        pub fn syllabix_whisper_decode(
            ctx: *mut WhisperContext,
            pcm: *const f32,
            n_samples: c_int,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
    }
}

/// True when both frontends resolved against the single `ggml`.
pub fn frontends_linked() -> bool {
    hush_logs();
    unsafe { ffi::syllabix_native_link_anchor() != 0 }
}

/// llama.cpp CPU system-info string. Does not load a GGUF.
pub fn llama_system_info() -> String {
    hush_logs();
    unsafe {
        ffi::syllabix_llama_backend_init();
        let ptr = ffi::syllabix_llama_system_info();
        let text = if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        ffi::syllabix_llama_backend_free();
        text
    }
}

/// In-process whisper.cpp context loaded from a GGML weight file.
pub struct WhisperContext {
    raw: *mut ffi::WhisperContext,
}

unsafe impl Send for WhisperContext {}

impl WhisperContext {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        hush_logs();
        let path = path.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| format!("model path is not valid UTF-8: {}", path.display()))?;
        let c_path = CString::new(path).map_err(|_| "model path contains an interior NUL")?;
        let raw = unsafe { ffi::syllabix_whisper_load(c_path.as_ptr()) };
        if raw.is_null() {
            return Err(format!("failed to load whisper.cpp model at {path}"));
        }
        Ok(Self { raw })
    }

    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn decode(
        &mut self,
        pcm: &[f32],
        n_threads: i32,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<String, DecodeError> {
        if pcm.is_empty() {
            return Err(DecodeError::Failed("no samples".into()));
        }
        let mut out = vec![0u8; 32 * 1024];
        let rc = unsafe {
            ffi::syllabix_whisper_decode(
                self.raw,
                pcm.as_ptr(),
                pcm.len() as c_int,
                n_threads,
                abort,
                abort_user,
                out.as_mut_ptr().cast::<c_char>(),
                out.len() as c_int,
            )
        };
        match rc {
            0 => {
                let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
                Ok(String::from_utf8_lossy(&out[..end]).into_owned())
            }
            1 => Err(DecodeError::Cancelled),
            _ => Err(DecodeError::Failed("whisper.cpp decode failed".into())),
        }
    }
}

impl Drop for WhisperContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::syllabix_whisper_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Cancelled,
    Failed(String),
}

fn hush_logs() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { ffi::syllabix_native_hush_logs() });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_frontends_are_linked() {
        assert!(frontends_linked());
        let info = llama_system_info();
        assert!(
            !info.is_empty(),
            "llama.cpp frontend must report system info"
        );
        assert!(
            !info.to_ascii_lowercase().contains("cuda")
                || info.to_ascii_lowercase().contains("cpu"),
            "unexpected llama system info: {info}"
        );
    }
}
