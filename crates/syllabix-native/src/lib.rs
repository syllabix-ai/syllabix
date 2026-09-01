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

    #[repr(C)]
    pub struct QwenTtsHandle {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn syllabix_native_hush_logs();
        pub fn syllabix_native_link_anchor() -> c_int;
        pub fn syllabix_llama_system_info() -> *const c_char;
        pub fn syllabix_llama_backend_init();
        pub fn syllabix_llama_backend_free();
        pub fn syllabix_llama_n_gpu_layers() -> c_int;
        pub fn syllabix_whisper_use_gpu() -> c_int;
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
            out_lang: *mut c_char,
            out_lang_cap: c_int,
        ) -> c_int;
        pub fn syllabix_llama_load(path: *const c_char, n_threads: c_int) -> *mut LlamaHandle;
        pub fn syllabix_llama_free(llm: *mut LlamaHandle);
        pub fn syllabix_llama_n_ctx(llm: *const LlamaHandle) -> c_int;
        pub fn syllabix_llama_n_ctx_train(llm: *const LlamaHandle) -> c_int;
        pub fn syllabix_llama_count_prompt_tokens(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
        ) -> c_int;
        pub fn syllabix_llama_generate(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            token_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            token_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_qwen_tts_load(
            model_path: *const c_char,
            mmproj_path: *const c_char,
            n_threads: c_int,
            seed: u32,
        ) -> *mut QwenTtsHandle;
        pub fn syllabix_qwen_tts_free(tts: *mut QwenTtsHandle);
        pub fn syllabix_qwen_tts_has_voice(tts: *const QwenTtsHandle) -> c_int;
        pub fn syllabix_qwen_tts_synthesize(
            tts: *mut QwenTtsHandle,
            text: *const c_char,
            lang: *const c_char,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            out_sample_rate: *mut i32,
            out_pcm: *mut *mut i16,
            out_n_samples: *mut i64,
        ) -> c_int;
        pub fn syllabix_qwen_tts_synthesize_streaming(
            tts: *mut QwenTtsHandle,
            text: *const c_char,
            lang: *const c_char,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            pcm_cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
            pcm_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_qwen_tts_pcm_free(pcm: *mut i16);
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

/// llama.cpp system-info string. Does not load a GGUF.
///
/// Memoized: `llama_print_system_info` builds into a shared static
/// `std::string` (`clear()` then append) that is not thread-safe, so
/// concurrent callers (the test harness runs tests in parallel threads)
/// can observe a wiped string. System info never changes for the process,
/// so the first copy wins.
pub fn llama_system_info() -> String {
    static INFO: OnceLock<String> = OnceLock::new();
    INFO.get_or_init(|| {
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
    })
    .clone()
}

/// Layers offloaded to Metal. `-1` on Darwin (all), `0` on Linux/Windows.
pub fn llama_n_gpu_layers() -> i32 {
    unsafe { ffi::syllabix_llama_n_gpu_layers() }
}

/// Whisper encoder uses Metal on Darwin. CPU everywhere else.
pub fn whisper_use_gpu() -> bool {
    unsafe { ffi::syllabix_whisper_use_gpu() != 0 }
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
    /// `language` is a whisper.cpp id (`en`) or `auto`, valid UTF-8 without interior NULs.
    ///
    /// Returns `(text, language)`: the transcript plus the effective language
    /// code — the requested id, or the code whisper.cpp detected when
    /// `language` was `auto`.
    pub unsafe fn decode(
        &mut self,
        pcm: &[f32],
        n_threads: i32,
        language: &str,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<(String, String), DecodeError> {
        if pcm.is_empty() {
            return Err(DecodeError::Failed("no samples".into()));
        }
        let lang = CString::new(language)
            .map_err(|_| DecodeError::Failed("language contains NUL".into()))?;
        let mut out = vec![0u8; 32 * 1024];
        let mut lang_out = vec![0u8; 16];
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
                lang_out.as_mut_ptr().cast::<c_char>(),
                lang_out.len() as c_int,
            )
        };
        match rc {
            0 => {
                let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
                let text = String::from_utf8_lossy(&out[..end]).into_owned();
                let lang_end = lang_out
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(lang_out.len());
                let detected = String::from_utf8_lossy(&lang_out[..lang_end]).into_owned();
                let detected = if detected.is_empty() {
                    language.to_string()
                } else {
                    detected
                };
                Ok((text, detected))
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
    /// Append an empty think block to disable thinking where the model's chat
    /// format supports that convention.
    pub append_thinking_off_suffix: bool,
    /// llama.cpp thread count.
    pub n_threads: i32,
}

/// In-process llama.cpp context loaded from a GGUF.
pub struct LlamaContext {
    raw: *mut ffi::LlamaHandle,
}

unsafe impl Send for LlamaContext {}

impl LlamaContext {
    /// Load an instruct GGUF. Context length is the model's trained window.
    pub fn load(path: impl AsRef<Path>, n_threads: i32) -> Result<Self, String> {
        hush_logs();
        unsafe { ffi::syllabix_llama_backend_init() };
        let path = path.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| format!("model path is not valid UTF-8: {}", path.display()))?;
        let c_path = CString::new(path).map_err(|_| "model path contains an interior NUL")?;
        let _ggml = ggml_lock();
        let raw = unsafe { ffi::syllabix_llama_load(c_path.as_ptr(), n_threads) };
        if raw.is_null() {
            return Err(format!("failed to load llama.cpp GGUF at {path}"));
        }
        Ok(Self { raw })
    }

    /// llama.cpp context size after load (`n_ctx_train` when `n_ctx` was 0).
    pub fn n_ctx(&self) -> i32 {
        unsafe { ffi::syllabix_llama_n_ctx(self.raw) }
    }

    /// GGUF trained context length.
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { ffi::syllabix_llama_n_ctx_train(self.raw) }
    }

    /// Count the exact chat-template prompt tokens used by generation.
    pub fn prompt_token_count(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
    ) -> Result<usize, LlamaError> {
        let roles: Result<Vec<CString>, LlamaError> = messages
            .iter()
            .map(|m| {
                CString::new(m.role.as_str()).map_err(|_| LlamaError::Failed("role NUL".into()))
            })
            .collect();
        let contents: Result<Vec<CString>, LlamaError> = messages
            .iter()
            .map(|m| {
                CString::new(m.content.as_str())
                    .map_err(|_| LlamaError::Failed("content NUL".into()))
            })
            .collect();
        let roles = roles?;
        let contents = contents?;
        let role_ptrs: Vec<*const c_char> = roles.iter().map(|s| s.as_ptr()).collect();
        let content_ptrs: Vec<*const c_char> = contents.iter().map(|s| s.as_ptr()).collect();
        let _ggml = ggml_lock();
        let count = unsafe {
            ffi::syllabix_llama_count_prompt_tokens(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                if append_thinking_off_suffix { 0 } else { 1 },
            )
        };
        if count < 1 {
            return Err(LlamaError::Failed(
                "llama.cpp prompt tokenization failed".into(),
            ));
        }
        Ok(count as usize)
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
                if opts.append_thinking_off_suffix {
                    0
                } else {
                    1
                },
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

#[derive(Debug, PartialEq, Eq)]
pub enum QwenTtsError {
    Cancelled,
    Failed(String),
}

type QwenPcmCallback<'a> = dyn FnMut(i32, &[f32], bool) -> Result<(), QwenTtsError> + 'a;

struct QwenPcmStream<'a> {
    on_pcm: &'a mut QwenPcmCallback<'a>,
    error: Option<QwenTtsError>,
}

unsafe extern "C" fn qwen_pcm_callback(
    sample_rate: i32,
    pcm: *const f32,
    n_samples: i64,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() || n_samples < 0 || (n_samples > 0 && pcm.is_null()) {
        return -1;
    }
    // Safety: `synthesize_streaming` keeps this stack value alive for the C call.
    let stream = unsafe { &mut *(user as *mut QwenPcmStream<'_>) };
    let samples = if n_samples == 0 {
        &[]
    } else {
        // Safety: native owns this PCM until the callback returns.
        unsafe { std::slice::from_raw_parts(pcm, n_samples as usize) }
    };
    match (stream.on_pcm)(sample_rate, samples, is_last != 0) {
        Ok(()) => 0,
        Err(err) => {
            stream.error = Some(err);
            1
        }
    }
}

/// In-process Qwen3-TTS context: backbone GGUF + mmproj through the shared
/// ggml. One instance synthesizes sentences sequentially; each sentence is an
/// independent generation on the underlying llama.cpp context.
pub struct QwenTtsContext {
    raw: *mut ffi::QwenTtsHandle,
}

unsafe impl Send for QwenTtsContext {}

impl QwenTtsContext {
    /// Load the backbone GGUF and mmproj (speech tokenizer). `seed` pins the
    /// semantic-token sampler; pass a fixed value in tests, `u32::MAX`
    /// (llama.cpp `LLAMA_DEFAULT_SEED`) for runtime randomness.
    pub fn load(
        model_path: impl AsRef<Path>,
        mmproj_path: impl AsRef<Path>,
        n_threads: i32,
        seed: u32,
    ) -> Result<Self, String> {
        hush_logs();
        unsafe { ffi::syllabix_llama_backend_init() };
        let to_c = |p: &Path| {
            let s = p
                .to_str()
                .ok_or_else(|| format!("path is not valid UTF-8: {}", p.display()))?;
            CString::new(s).map_err(|_| "path contains an interior NUL".to_string())
        };
        let c_model = to_c(model_path.as_ref())?;
        let c_mmproj = to_c(mmproj_path.as_ref())?;
        let _ggml = ggml_lock();
        let raw = unsafe {
            ffi::syllabix_qwen_tts_load(c_model.as_ptr(), c_mmproj.as_ptr(), n_threads, seed)
        };
        if raw.is_null() {
            return Err(format!(
                "failed to load Qwen3-TTS at {} (mmproj {})",
                model_path.as_ref().display(),
                mmproj_path.as_ref().display()
            ));
        }
        Ok(Self { raw })
    }

    /// 1 when the self-voice anchor engaged at load; 0 means generation fell
    /// back to unconditioned sampling. Diagnostics and native tests.
    pub fn has_voice(&self) -> bool {
        if self.raw.is_null() {
            return false;
        }
        let _ggml = ggml_lock();
        unsafe { ffi::syllabix_qwen_tts_has_voice(self.raw) == 1 }
    }

    /// Synthesize one sentence into mono i16 PCM plus its sample rate.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn synthesize(
        &mut self,
        text: &str,
        lang: &str,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<(i32, Vec<i16>), QwenTtsError> {
        if text.is_empty() {
            return Err(QwenTtsError::Failed("empty speak text".into()));
        }
        let c_text =
            CString::new(text).map_err(|_| QwenTtsError::Failed("text contains NUL".into()))?;
        let c_lang =
            CString::new(lang).map_err(|_| QwenTtsError::Failed("language contains NUL".into()))?;
        let mut rate: i32 = 0;
        let mut pcm: *mut i16 = std::ptr::null_mut();
        let mut n_samples: i64 = 0;
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_qwen_tts_synthesize(
                self.raw,
                c_text.as_ptr(),
                c_lang.as_ptr(),
                abort,
                abort_user,
                &mut rate,
                &mut pcm,
                &mut n_samples,
            )
        };
        match rc {
            0 => {
                if pcm.is_null() || n_samples <= 0 || rate <= 0 {
                    return Err(QwenTtsError::Failed(
                        "native synth returned no audio".into(),
                    ));
                }
                let samples =
                    unsafe { std::slice::from_raw_parts(pcm, n_samples as usize) }.to_vec();
                unsafe { ffi::syllabix_qwen_tts_pcm_free(pcm) };
                Ok((rate, samples))
            }
            1 => Err(QwenTtsError::Cancelled),
            _ => Err(QwenTtsError::Failed("Qwen3-TTS synthesis failed".into())),
        }
    }

    /// Stream native-rate f32 PCM windows while Qwen is still generating.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn synthesize_streaming(
        &mut self,
        text: &str,
        lang: &str,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_pcm: &mut QwenPcmCallback<'_>,
    ) -> Result<(), QwenTtsError> {
        if text.is_empty() {
            return Err(QwenTtsError::Failed("empty speak text".into()));
        }
        let c_text =
            CString::new(text).map_err(|_| QwenTtsError::Failed("text contains NUL".into()))?;
        let c_lang =
            CString::new(lang).map_err(|_| QwenTtsError::Failed("language contains NUL".into()))?;
        let mut stream = QwenPcmStream {
            on_pcm,
            error: None,
        };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_qwen_tts_synthesize_streaming(
                self.raw,
                c_text.as_ptr(),
                c_lang.as_ptr(),
                abort,
                abort_user,
                Some(qwen_pcm_callback),
                (&mut stream as *mut QwenPcmStream<'_>).cast(),
            )
        };
        if let Some(err) = stream.error {
            return Err(err);
        }
        match rc {
            0 => Ok(()),
            1 => Err(QwenTtsError::Cancelled),
            _ => Err(QwenTtsError::Failed(
                "Qwen3-TTS streaming synthesis failed".into(),
            )),
        }
    }
}

impl Drop for QwenTtsContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_qwen_tts_free(self.raw) };
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
        let lower = info.to_ascii_lowercase();
        assert!(
            !lower.contains("cuda") || lower.contains("cpu"),
            "unexpected llama system info: {info}"
        );
    }

    #[test]
    fn n_gpu_layers_matches_os() {
        #[cfg(not(target_vendor = "apple"))]
        {
            assert_eq!(llama_n_gpu_layers(), 0);
            assert!(!whisper_use_gpu());
            let lower = llama_system_info().to_ascii_lowercase();
            assert!(
                !lower.contains("metal") && !lower.contains("mtl"),
                "Linux/Windows ggml must stay CPU-only: {lower}"
            );
        }
        #[cfg(target_vendor = "apple")]
        {
            assert_eq!(llama_n_gpu_layers(), -1);
            assert!(whisper_use_gpu());
            let lower = llama_system_info().to_ascii_lowercase();
            assert!(
                lower.contains("metal") || lower.contains("mtl"),
                "Darwin ggml must compile Metal: {lower}"
            );
        }
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
        let err = match LlamaContext::load("/no/such/model.gguf", 1) {
            Err(err) => err,
            Ok(_) => panic!("missing llama weights should fail"),
        };
        assert!(err.contains("failed to load llama.cpp"));
    }

    #[test]
    fn missing_qwen_tts_weights_do_not_load() {
        let err =
            match QwenTtsContext::load("/no/such/qwen3-tts.gguf", "/no/such/mmproj.gguf", 1, 0) {
                Err(err) => err,
                Ok(_) => panic!("missing Qwen3-TTS weights should fail"),
            };
        assert!(err.contains("failed to load Qwen3-TTS"));
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
