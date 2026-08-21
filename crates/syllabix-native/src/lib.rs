//! Shared `ggml` plus the whisper.cpp and llama.cpp frontends.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, Once, OnceLock};

/// Whisper and Llama share one `ggml`. Concurrent `whisper_full` / `llama_decode`
/// in the pipeline (STT of turn N+1 overlapping LLM of turn N) is not safe.
fn ggml_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[allow(dead_code)]
mod ffi {
    use super::*;

    #[repr(C)]
    pub struct WhisperContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct LlamaHandle {
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
            language: *const c_char,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
        pub fn syllabix_llama_load(
            path: *const c_char,
            n_ctx: c_int,
            n_threads: c_int,
        ) -> *mut LlamaHandle;
        pub fn syllabix_llama_free(llm: *mut LlamaHandle);
        pub fn syllabix_llama_generate(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            n_predict: c_int,
            thinking: c_int,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            token_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            token_user: *mut c_void,
        ) -> c_int;
    }
}

/// One chat message passed into llama.cpp's template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
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
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
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
        let _ggml = ggml_lock();
        let raw = unsafe { ffi::syllabix_whisper_load(c_path.as_ptr()) };
        if raw.is_null() {
            return Err(format!("failed to load whisper.cpp model at {path}"));
        }
        Ok(Self { raw })
    }

    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    /// `language` is a whisper.cpp id (`en`) and must remain valid UTF-8 without interior NULs.
    pub unsafe fn decode(
        &mut self,
        pcm: &[f32],
        n_threads: i32,
        language: &str,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<String, DecodeError> {
        if pcm.is_empty() {
            return Err(DecodeError::Failed("no samples".into()));
        }
        let lang = CString::new(language)
            .map_err(|_| DecodeError::Failed("language contains NUL".into()))?;
        let mut out = vec![0u8; 32 * 1024];
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_whisper_decode(
                self.raw,
                pcm.as_ptr(),
                pcm.len() as c_int,
                n_threads,
                lang.as_ptr(),
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
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_whisper_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Options for one greedy llama.cpp generate.
pub struct LlamaGenerate {
    /// Token budget (`n_predict`).
    pub n_predict: i32,
    /// When false, append an empty Qwen think closer so the model skips CoT.
    pub thinking: bool,
    /// llama.cpp thread count.
    pub n_threads: i32,
}

/// In-process llama.cpp context loaded from a GGUF.
pub struct LlamaContext {
    raw: *mut ffi::LlamaHandle,
}

unsafe impl Send for LlamaContext {}

impl LlamaContext {
    /// Load `Llama-3.2-1B-Instruct` (or any instruct GGUF) for CPU greedy decode.
    pub fn load(path: impl AsRef<Path>, n_ctx: i32, n_threads: i32) -> Result<Self, String> {
        hush_logs();
        unsafe { ffi::syllabix_llama_backend_init() };
        let path = path.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| format!("model path is not valid UTF-8: {}", path.display()))?;
        let c_path = CString::new(path).map_err(|_| "model path contains an interior NUL")?;
        let _ggml = ggml_lock();
        let raw = unsafe { ffi::syllabix_llama_load(c_path.as_ptr(), n_ctx, n_threads) };
        if raw.is_null() {
            return Err(format!("failed to load llama.cpp GGUF at {path}"));
        }
        Ok(Self { raw })
    }

    /// Stream greedy pieces. `on_piece` is invoked in order; the last call has `is_last`.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn generate(
        &mut self,
        messages: &[ChatMessage],
        opts: LlamaGenerate,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<(), LlamaError>,
    ) -> Result<(), LlamaError> {
        if messages.is_empty() {
            return Err(LlamaError::Failed("no chat messages".into()));
        }
        let roles: Result<Vec<CString>, LlamaError> = messages
            .iter()
            .map(|m| {
                CString::new(m.role.as_str()).map_err(|_| LlamaError::Failed("role NUL".into()))
            })
            .collect();
        let roles = roles?;
        let contents: Result<Vec<CString>, LlamaError> = messages
            .iter()
            .map(|m| {
                CString::new(m.content.as_str())
                    .map_err(|_| LlamaError::Failed("content NUL".into()))
            })
            .collect();
        let contents = contents?;
        let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();

        let mut sink = TokenSink { on_piece };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_llama_generate(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                opts.n_predict,
                if opts.thinking { 1 } else { 0 },
                opts.n_threads,
                abort,
                abort_user,
                Some(on_token_piece),
                (&mut sink as *mut TokenSink).cast(),
            )
        };
        match rc {
            0 => Ok(()),
            1 => Err(LlamaError::Cancelled),
            _ => Err(LlamaError::Failed("llama.cpp generate failed".into())),
        }
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_llama_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

struct TokenSink<'a> {
    on_piece: &'a mut dyn FnMut(&str, bool) -> Result<(), LlamaError>,
}

unsafe extern "C" fn on_token_piece(
    piece: *const c_char,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() {
        return -1;
    }
    let text = if piece.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(piece) }
            .to_string_lossy()
            .into_owned()
    };
    let sink = unsafe { &mut *(user as *mut TokenSink) };
    match (sink.on_piece)(&text, is_last != 0) {
        Ok(()) => 0,
        Err(LlamaError::Cancelled) => 1,
        Err(_) => -1,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Cancelled,
    Failed(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum LlamaError {
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

    #[test]
    fn missing_whisper_weights_do_not_load() {
        let err = match WhisperContext::load("/no/such/ggml-small.bin") {
            Err(err) => err,
            Ok(_) => panic!("missing whisper weights should fail"),
        };
        assert!(err.contains("failed to load whisper.cpp"));
    }

    #[test]
    fn missing_llama_weights_do_not_load() {
        let err = match LlamaContext::load("/no/such/model.gguf", 2048, 1) {
            Err(err) => err,
            Ok(_) => panic!("missing llama weights should fail"),
        };
        assert!(err.contains("failed to load llama.cpp"));
    }

    #[test]
    fn decode_and_generate_errors_are_distinct() {
        assert_ne!(DecodeError::Cancelled, DecodeError::Failed("x".into()));
        assert_ne!(LlamaError::Cancelled, LlamaError::Failed("x".into()));
        let msg = ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        };
        assert_eq!(msg.role, "user");
    }
}
