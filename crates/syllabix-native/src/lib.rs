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

    #[cfg(not(coverage))]
    extern "C" {
        pub fn syllabix_native_hush_logs();
        pub fn syllabix_native_link_anchor() -> c_int;
        pub fn syllabix_llama_system_info() -> *const c_char;
        pub fn syllabix_llama_backend_init();
        pub fn syllabix_llama_n_gpu_layers() -> c_int;
        pub fn syllabix_whisper_use_gpu() -> c_int;
        pub fn syllabix_vk_device_count_or_zero() -> c_int;
        pub fn syllabix_vk_device0_description(out: *mut c_char, out_cap: usize) -> c_int;
        pub fn syllabix_vk_device0_vram_bytes(total_bytes: *mut u64) -> c_int;
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
        pub fn syllabix_llama_generate_with_tools(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
            tools_json: *const c_char,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            token_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            token_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_llama_render_qwen(
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            tools_json: *const c_char,
            thinking: c_int,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
        pub fn syllabix_llama_count_prompt_tokens_with_tools(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
            tools_json: *const c_char,
        ) -> c_int;
        pub fn syllabix_llama_generate_with_lfm_tools(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
            tools_json: *const c_char,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            token_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            token_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_llama_render_lfm(
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            tools_json: *const c_char,
            thinking: c_int,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
        pub fn syllabix_llama_count_prompt_tokens_with_lfm_tools(
            llm: *mut LlamaHandle,
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            thinking: c_int,
            tools_json: *const c_char,
        ) -> c_int;
        pub fn syllabix_qwen_tts_load(
            model_path: *const c_char,
            mmproj_path: *const c_char,
            n_threads: c_int,
            seed: u32,
            backend: c_int,
        ) -> *mut QwenTtsHandle;
        pub fn syllabix_qwen_tts_free(tts: *mut QwenTtsHandle);
        pub fn syllabix_qwen_tts_backend(tts: *const QwenTtsHandle) -> c_int;
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

    // The coverage build must exercise the Rust safety/translation layer
    // without loading multi-gigabyte native models. Keep prompt rendering on
    // the real shim (it is pure and already weight-free); fake the calls that
    // require a live native context.
    #[cfg(coverage)]
    extern "C" {
        pub fn syllabix_llama_render_qwen(
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            tools_json: *const c_char,
            thinking: c_int,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
        pub fn syllabix_llama_render_lfm(
            roles: *const *const c_char,
            contents: *const *const c_char,
            n_messages: c_int,
            tools_json: *const c_char,
            thinking: c_int,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
    }

    #[cfg(coverage)]
    fn text(ptr: *const c_char) -> String {
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    #[cfg(coverage)]
    fn fail(text: &str, needle: &str) -> bool {
        text.contains(needle)
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_native_hush_logs() {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_native_link_anchor() -> c_int {
        1
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_system_info() -> *const c_char {
        if cfg!(target_os = "macos") {
            c"coverage fake backend Metal".as_ptr()
        } else {
            c"coverage fake backend CPU".as_ptr()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_backend_init() {}

    #[cfg(coverage)]
    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_n_gpu_layers() -> c_int {
        if cfg!(target_os = "macos") {
            -1
        } else {
            0
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_whisper_use_gpu() -> c_int {
        i32::from(cfg!(target_os = "macos"))
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_vk_device_count_or_zero() -> c_int {
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_vk_device0_description(out: *mut c_char, out_cap: usize) -> c_int {
        if !out.is_null() && out_cap > 0 {
            *out = 0;
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_vk_device0_vram_bytes(total_bytes: *mut u64) -> c_int {
        if !total_bytes.is_null() {
            *total_bytes = 0;
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_whisper_load(path: *const c_char) -> *mut WhisperContext {
        if fail(&text(path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<WhisperContext>()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_whisper_free(_ctx: *mut WhisperContext) {}

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_whisper_decode(
        _ctx: *mut WhisperContext,
        _pcm: *const f32,
        _n_samples: c_int,
        _n_threads: c_int,
        language: *const c_char,
        _abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        out: *mut c_char,
        _out_cap: c_int,
        out_lang: *mut c_char,
        _out_lang_cap: c_int,
    ) -> c_int {
        let lang = text(language);
        if lang == "cancel" {
            return 1;
        }
        if lang == "error" {
            return 2;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(c"hello from fake whisper".as_ptr(), out, 24);
            if lang == "auto" {
                std::ptr::copy_nonoverlapping(c"es".as_ptr(), out_lang, 3);
            }
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_load(path: *const c_char, _n_threads: c_int) -> *mut LlamaHandle {
        if fail(&text(path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<LlamaHandle>()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_free(_llm: *mut LlamaHandle) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_n_ctx(_llm: *const LlamaHandle) -> c_int {
        4096
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_n_ctx_train(_llm: *const LlamaHandle) -> c_int {
        4096
    }

    #[cfg(coverage)]
    unsafe fn message_text(contents: *const *const c_char, n: c_int) -> String {
        if n <= 0 || contents.is_null() {
            return String::new();
        }
        text(*contents)
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_count_prompt_tokens(
        _llm: *mut LlamaHandle,
        _roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        _thinking: c_int,
    ) -> c_int {
        if message_text(contents, n_messages).contains("count-fail") {
            0
        } else {
            7
        }
    }

    #[cfg(coverage)]
    unsafe fn emit_token(
        contents: *const *const c_char,
        n_messages: c_int,
        cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        let message = message_text(contents, n_messages);
        if message.contains("cancel") {
            return 1;
        }
        if message.contains("error") {
            return 2;
        }
        let piece = c"fake reply";
        match cb {
            Some(cb) => {
                let rc = cb(piece.as_ptr(), 0, user);
                if rc != 0 {
                    return rc;
                }
                cb(c"".as_ptr(), 1, user)
            }
            None => 0,
        }
    }

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_llama_generate(
        _llm: *mut LlamaHandle,
        _roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        _thinking: c_int,
        _n_threads: c_int,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        emit_token(contents, n_messages, cb, user)
    }

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_llama_generate_with_tools(
        llm: *mut LlamaHandle,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        thinking: c_int,
        _tools: *const c_char,
        n_threads: c_int,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        syllabix_llama_generate(
            llm, roles, contents, n_messages, thinking, n_threads, abort, abort_user, cb, user,
        )
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_count_prompt_tokens_with_tools(
        llm: *mut LlamaHandle,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        thinking: c_int,
        _tools: *const c_char,
    ) -> c_int {
        syllabix_llama_count_prompt_tokens(llm, roles, contents, n_messages, thinking)
    }

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_llama_generate_with_lfm_tools(
        llm: *mut LlamaHandle,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        thinking: c_int,
        _tools: *const c_char,
        n_threads: c_int,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        syllabix_llama_generate(
            llm, roles, contents, n_messages, thinking, n_threads, abort, abort_user, cb, user,
        )
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_llama_count_prompt_tokens_with_lfm_tools(
        llm: *mut LlamaHandle,
        roles: *const *const c_char,
        contents: *const *const c_char,
        n_messages: c_int,
        thinking: c_int,
        _tools: *const c_char,
    ) -> c_int {
        syllabix_llama_count_prompt_tokens(llm, roles, contents, n_messages, thinking)
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_load(
        model_path: *const c_char,
        _mmproj_path: *const c_char,
        _n_threads: c_int,
        _seed: u32,
        _backend: c_int,
    ) -> *mut QwenTtsHandle {
        if fail(&text(model_path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<QwenTtsHandle>()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_free(_tts: *mut QwenTtsHandle) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_backend(_tts: *const QwenTtsHandle) -> c_int {
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_has_voice(_tts: *const QwenTtsHandle) -> c_int {
        1
    }

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_qwen_tts_synthesize(
        _tts: *mut QwenTtsHandle,
        text_ptr: *const c_char,
        _lang: *const c_char,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        rate: *mut i32,
        pcm: *mut *mut i16,
        n_samples: *mut i64,
    ) -> c_int {
        let value = text(text_ptr);
        if value.contains("cancel") {
            return 1;
        }
        if value.contains("error") {
            return 2;
        }
        unsafe {
            *rate = 24_000;
            *n_samples = 2;
            *pcm = Box::into_raw(Box::new([100i16, -100i16])).cast::<i16>();
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_synthesize_streaming(
        _tts: *mut QwenTtsHandle,
        text_ptr: *const c_char,
        _lang: *const c_char,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        let value = text(text_ptr);
        if value.contains("cancel") {
            return 1;
        }
        if value.contains("error") {
            return 2;
        }
        let first = [0.1f32, 0.2];
        let second = [0.3f32];
        if let Some(cb) = cb {
            if cb(24_000, first.as_ptr(), 2, 0, user) != 0 {
                return 2;
            }
            if cb(24_000, second.as_ptr(), 1, 1, user) != 0 {
                return 2;
            }
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_tts_pcm_free(pcm: *mut i16) {
        if !pcm.is_null() {
            drop(Box::from_raw(pcm.cast::<[i16; 2]>()));
        }
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

/// Layers offloaded to GPU. `-1` on Darwin (Metal) and on Vulkan Linux builds
/// with at least one device; `0` on the default portable CPU artifact.
pub fn llama_n_gpu_layers() -> i32 {
    unsafe { ffi::syllabix_llama_n_gpu_layers() }
}

/// Whisper encoder uses Metal on Darwin, or Vulkan when that backend is
/// compiled in and a device is present. CPU everywhere else.
pub fn whisper_use_gpu() -> bool {
    unsafe { ffi::syllabix_whisper_use_gpu() != 0 }
}

/// `true` when this binary was built with `SYLLABIX_GGML_VULKAN=1` (Linux).
pub const fn ggml_vulkan_compiled() -> bool {
    cfg!(syllabix_ggml_vulkan)
}

/// Vulkan devices ggml can use; `0` when Vulkan is not compiled in or instance
/// init fails (no ICD, API < 1.2). Never unwinds a C++ exception into Rust.
pub fn vulkan_device_count() -> usize {
    usize::try_from(unsafe { ffi::syllabix_vk_device_count_or_zero() }).unwrap_or(0)
}

/// ggml placement used for whisper/llama: `cpu`, `metal`, or `vulkan`.
///
/// Matches runtime offload (`llama_n_gpu_layers` / `whisper_use_gpu`): Darwin
/// Metal when GPU is on; Vulkan only on a Vulkan-enabled Linux build with a
/// usable device; otherwise CPU.
pub fn ggml_backend_id() -> &'static str {
    if !whisper_use_gpu() {
        return "cpu";
    }
    if cfg!(target_os = "macos") {
        "metal"
    } else if ggml_vulkan_compiled() {
        "vulkan"
    } else {
        // Today's shim never enables GPU outside Darwin Metal or Vulkan Linux.
        // Fail loudly in debug if a future backend forgets to extend this map.
        debug_assert!(
            false,
            "whisper_use_gpu without macOS Metal or a Vulkan build"
        );
        "cpu"
    }
}

/// Marketing name for Vulkan device 0 when known.
pub fn vulkan_device0_name() -> Option<String> {
    let mut buf = vec![0u8; 256];
    let ok = unsafe {
        ffi::syllabix_vk_device0_description(buf.as_mut_ptr().cast::<c_char>(), buf.len())
    };
    if ok == 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..end]).trim().to_owned();
    (!name.is_empty()).then_some(name)
}

/// Total VRAM bytes for Vulkan device 0 when known.
pub fn vulkan_device0_vram_bytes() -> Option<u64> {
    let mut total = 0u64;
    let ok = unsafe { ffi::syllabix_vk_device0_vram_bytes(&mut total) };
    (ok != 0 && total > 0).then_some(total)
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
#[derive(Clone, Copy)]
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

    /// Render a Qwen tools-aware prompt without loading weights.
    ///
    /// Pure unit-test seam for the `<tools>` preamble and `tool`-role
    /// grouping. `tools_json` is the JSON array of tool
    /// definitions; empty means the tool-free Qwen framing.
    pub fn render_qwen_prompt(
        messages: &[ChatMessage],
        tools_json: &str,
        thinking_off_suffix: bool,
    ) -> Result<String, LlamaError> {
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
        // Measure, then render. The renderer is pure (no model handle), so no
        // ggml lock is needed here.
        let len = unsafe {
            ffi::syllabix_llama_render_qwen(
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                c_tools.as_ptr(),
                if thinking_off_suffix { 0 } else { 1 },
                std::ptr::null_mut(),
                0,
            )
        };
        if len < 1 {
            return Err(LlamaError::Failed("qwen prompt rendering failed".into()));
        }
        let mut out = vec![0u8; (len as usize) + 1];
        let rc = unsafe {
            ffi::syllabix_llama_render_qwen(
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                c_tools.as_ptr(),
                if thinking_off_suffix { 0 } else { 1 },
                out.as_mut_ptr().cast::<c_char>(),
                out.len() as c_int,
            )
        };
        if rc != len {
            return Err(LlamaError::Failed("qwen prompt rendering failed".into()));
        }
        out.pop();
        String::from_utf8(out).map_err(|_| LlamaError::Failed("qwen prompt is not UTF-8".into()))
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

    /// Stream greedy pieces through the tools-aware entry.
    ///
    /// With empty `tools_json` the shim delegates to the plain path, so
    /// tool-free generation stays byte-identical.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn generate_with_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;

        let mut sink = TokenSink { on_piece };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_llama_generate_with_tools(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                if opts.append_thinking_off_suffix {
                    0
                } else {
                    1
                },
                c_tools.as_ptr(),
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

    /// Return the prompt-token count for a tools-aware prompt.
    pub fn prompt_token_count_with_tools(
        &mut self,
        messages: &[ChatMessage],
        append_thinking_off_suffix: bool,
        tools_json: &str,
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
        let _ggml = ggml_lock();
        let count = unsafe {
            ffi::syllabix_llama_count_prompt_tokens_with_tools(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                if append_thinking_off_suffix { 0 } else { 1 },
                c_tools.as_ptr(),
            )
        };
        if count < 1 {
            return Err(LlamaError::Failed(
                "llama.cpp prompt tokenization failed".into(),
            ));
        }
        Ok(count as usize)
    }

    /// Render an LFM tools-aware prompt without loading weights.
    ///
    /// Pure unit-test seam for the `List of tools:` preamble and
    /// `tool`-role turns. `tools_json` is the JSON array of tool
    /// definitions; empty means the tool-free LFM framing. Separate from
    /// the Qwen renderer (second-dialect exception).
    pub fn render_lfm_prompt(
        messages: &[ChatMessage],
        tools_json: &str,
    ) -> Result<String, LlamaError> {
        Self::render_lfm_prompt_inner(messages, tools_json)
    }

    fn render_lfm_prompt_inner(
        messages: &[ChatMessage],
        tools_json: &str,
    ) -> Result<String, LlamaError> {
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
        let len = unsafe {
            ffi::syllabix_llama_render_lfm(
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                c_tools.as_ptr(),
                1,
                std::ptr::null_mut(),
                0,
            )
        };
        if len < 1 {
            return Err(LlamaError::Failed("lfm prompt rendering failed".into()));
        }
        let mut out = vec![0u8; (len as usize) + 1];
        let rc = unsafe {
            ffi::syllabix_llama_render_lfm(
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                c_tools.as_ptr(),
                1,
                out.as_mut_ptr().cast::<c_char>(),
                out.len() as c_int,
            )
        };
        if rc != len {
            return Err(LlamaError::Failed("lfm prompt rendering failed".into()));
        }
        out.pop();
        String::from_utf8(out).map_err(|_| LlamaError::Failed("lfm prompt is not UTF-8".into()))
    }

    /// Stream greedy pieces through the LFM tools-aware entry.
    ///
    /// With empty `tools_json` the shim delegates to the plain path, so
    /// tool-free generation stays byte-identical.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn generate_with_lfm_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;

        let mut sink = TokenSink { on_piece };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_llama_generate_with_lfm_tools(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                if opts.append_thinking_off_suffix {
                    0
                } else {
                    1
                },
                c_tools.as_ptr(),
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

    /// Return the prompt-token count for an LFM tools-aware prompt.
    pub fn prompt_token_count_with_lfm_tools(
        &mut self,
        messages: &[ChatMessage],
        tools_json: &str,
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
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
        let _ggml = ggml_lock();
        let count = unsafe {
            ffi::syllabix_llama_count_prompt_tokens_with_lfm_tools(
                self.raw,
                role_ptrs.as_ptr(),
                content_ptrs.as_ptr(),
                messages.len() as c_int,
                1,
                c_tools.as_ptr(),
            )
        };
        if count < 1 {
            return Err(LlamaError::Failed(
                "llama.cpp prompt tokenization failed".into(),
            ));
        }
        Ok(count as usize)
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

/// Compute placement selected for a loaded Qwen3-TTS context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTtsBackend {
    Cpu,
    Metal,
    Vulkan,
}

impl QwenTtsBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }

    fn as_c_int(self) -> c_int {
        match self {
            Self::Cpu => 0,
            Self::Metal => 1,
            Self::Vulkan => 2,
        }
    }

    fn from_c_int(value: c_int) -> Self {
        match value {
            1 => Self::Metal,
            2 => Self::Vulkan,
            _ => Self::Cpu,
        }
    }
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
        backend: QwenTtsBackend,
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
            ffi::syllabix_qwen_tts_load(
                c_model.as_ptr(),
                c_mmproj.as_ptr(),
                n_threads,
                seed,
                backend.as_c_int(),
            )
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

    /// Compute backend confirmed by the native context after its full
    /// voice-anchor warm-up succeeds.
    pub fn backend(&self) -> QwenTtsBackend {
        let _ggml = ggml_lock();
        QwenTtsBackend::from_c_int(unsafe { ffi::syllabix_qwen_tts_backend(self.raw) })
    }

    /// 1 when the self-voice anchor and complete placement warm-up engaged.
    /// A successfully loaded context always returns 1; exposed for diagnostics
    /// and native tests.
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
            let lower = llama_system_info().to_ascii_lowercase();
            assert!(
                !lower.contains("metal") && !lower.contains("mtl"),
                "Linux/Windows ggml must not enable Metal: {lower}"
            );
            #[cfg(syllabix_ggml_vulkan)]
            {
                // Device-less hosts stay at CPU layers; a present Vulkan device
                // offloads and should appear in system info.
                if llama_n_gpu_layers() == -1 {
                    assert!(whisper_use_gpu());
                    assert!(
                        lower.contains("vulkan"),
                        "Vulkan offload should appear in system info: {lower}"
                    );
                } else {
                    assert_eq!(llama_n_gpu_layers(), 0);
                    assert!(!whisper_use_gpu());
                }
            }
            #[cfg(not(syllabix_ggml_vulkan))]
            {
                assert_eq!(llama_n_gpu_layers(), 0);
                assert!(!whisper_use_gpu());
            }
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
    fn ggml_backend_id_matches_offload() {
        let id = ggml_backend_id();
        assert!(
            matches!(id, "cpu" | "metal" | "vulkan"),
            "unexpected ggml backend {id}"
        );
        if whisper_use_gpu() {
            #[cfg(target_os = "macos")]
            assert_eq!(id, "metal");
            #[cfg(all(not(target_os = "macos"), syllabix_ggml_vulkan))]
            assert_eq!(id, "vulkan");
        } else {
            assert_eq!(id, "cpu");
            assert!(vulkan_device0_name().is_none());
            assert!(vulkan_device0_vram_bytes().is_none());
        }
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_whisper_weights_do_not_load() {
        let err = match WhisperContext::load("/no/such/ggml-small.bin") {
            Err(err) => err,
            Ok(_) => panic!("missing whisper weights should fail"),
        };
        assert!(err.contains("failed to load whisper.cpp"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_llama_weights_do_not_load() {
        let err = match LlamaContext::load("/no/such/model.gguf", 1) {
            Err(err) => err,
            Ok(_) => panic!("missing llama weights should fail"),
        };
        assert!(err.contains("failed to load llama.cpp"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_qwen_tts_weights_do_not_load() {
        let err = match QwenTtsContext::load(
            "/no/such/qwen3-tts.gguf",
            "/no/such/mmproj.gguf",
            1,
            0,
            QwenTtsBackend::Cpu,
        ) {
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

    fn qwen_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What time is it?".into(),
            },
        ]
    }

    #[test]
    fn qwen_tool_free_prompt_matches_chatml_framing() {
        let prompt = LlamaContext::render_qwen_prompt(&qwen_messages(), "", true).expect("render");
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nWhat time is it?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n</think>\n"));
        assert!(!prompt.contains("<tools>"));
    }

    #[test]
    fn qwen_tools_preamble_folds_the_system_message() {
        let tools = r#"[{"type":"function","function":{"name":"shell"}}]"#;
        let prompt =
            LlamaContext::render_qwen_prompt(&qwen_messages(), tools, true).expect("render");
        assert!(prompt.contains("<tools>\n"));
        assert!(prompt.contains(tools));
        assert!(prompt.contains("</tools>"));
        assert!(prompt.contains("<tool_call>"));
        // The system text joins the preamble; it must not repeat as a block.
        assert_eq!(prompt.matches("You are helpful.").count(), 1);
        assert!(prompt.contains("<|im_start|>user\nWhat time is it?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n</think>\n"));
    }

    #[test]
    fn qwen_tool_results_group_into_one_user_turn() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "How much space?".into(),
            },
            ChatMessage {
                role: "tool".into(),
                content: "first result".into(),
            },
            ChatMessage {
                role: "tool".into(),
                content: "second result".into(),
            },
        ];
        let prompt = LlamaContext::render_qwen_prompt(&messages, "", true).expect("render");
        assert!(prompt.contains(
            "<|im_start|>user\n<tool_response>\nfirst result\n</tool_response>\n<tool_response>\nsecond result\n</tool_response>\n<|im_end|>\n"
        ));
        assert!(!prompt.contains("<|im_start|>tool"));
    }

    #[test]
    fn qwen_render_rejects_empty_messages() {
        assert!(LlamaContext::render_qwen_prompt(&[], "", true).is_err());
    }

    fn lfm_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "What time is it?".into(),
            },
        ]
    }

    #[test]
    fn lfm_tool_free_prompt_uses_im_framing_without_tools_list() {
        let prompt = LlamaContext::render_lfm_prompt(&lfm_messages(), "").expect("render");
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nWhat time is it?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!prompt.contains("List of tools:"));
        assert!(!prompt.contains("tool_call_start"));
    }

    #[test]
    fn lfm_tools_preamble_lists_tools_and_folds_the_system_message() {
        let tools = r#"[{"name":"shell"}]"#;
        let prompt = LlamaContext::render_lfm_prompt(&lfm_messages(), tools).expect("render");
        assert!(prompt.contains("<|im_start|>system\nList of tools: "));
        assert!(prompt.contains(tools));
        assert!(prompt.contains("<|im_end|>\n"));
        // The system text joins the preamble; it must not repeat as a block.
        assert_eq!(prompt.matches("You are helpful.").count(), 1);
        assert!(prompt.contains("<|im_start|>user\nWhat time is it?<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn lfm_tool_results_render_as_native_tool_turns() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "How much space?".into(),
            },
            ChatMessage {
                role: "tool".into(),
                content: "first result".into(),
            },
            ChatMessage {
                role: "tool".into(),
                content: "second result".into(),
            },
        ];
        let prompt = LlamaContext::render_lfm_prompt(&messages, "").expect("render");
        assert!(prompt.contains("<|im_start|>tool\nfirst result<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>tool\nsecond result<|im_end|>\n"));
        assert!(!prompt.contains("<tool_response>"));
    }

    #[test]
    fn lfm_render_rejects_empty_messages() {
        assert!(LlamaContext::render_lfm_prompt(&[], "").is_err());
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_whisper_wrapper_without_weights() {
        assert!(WhisperContext::load("fail-whisper").is_err());
        let mut ctx = WhisperContext::load("fake-whisper").expect("fake context");
        assert!(unsafe { ctx.decode(&[], 1, "en", None, std::ptr::null_mut()) }.is_err());
        assert!(unsafe { ctx.decode(&[0.0], 1, "bad\0lang", None, std::ptr::null_mut()) }.is_err());
        assert_eq!(
            unsafe { ctx.decode(&[0.0], 1, "en", None, std::ptr::null_mut()) }
                .unwrap()
                .0,
            "hello from fake whisper"
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], 1, "auto", None, std::ptr::null_mut()) }
                .unwrap()
                .1,
            "es"
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], 1, "cancel", None, std::ptr::null_mut()) },
            Err(DecodeError::Cancelled)
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], 1, "error", None, std::ptr::null_mut()) },
            Err(DecodeError::Failed("whisper.cpp decode failed".into()))
        );
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_llama_wrapper_without_weights() {
        assert!(LlamaContext::load("fail-llama", 1).is_err());
        let mut ctx = LlamaContext::load("fake-llama", 1).expect("fake context");
        assert_eq!(ctx.n_ctx(), 4096);
        assert_eq!(ctx.n_ctx_train(), 4096);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        assert_eq!(ctx.prompt_token_count(&messages, true).unwrap(), 7);
        assert!(ctx
            .prompt_token_count(
                &[ChatMessage {
                    role: "user".into(),
                    content: "count-fail".into()
                }],
                false
            )
            .is_err());
        assert!(ctx
            .prompt_token_count(
                &[ChatMessage {
                    role: "bad\0role".into(),
                    content: "x".into()
                }],
                false
            )
            .is_err());
        let opts = LlamaGenerate {
            append_thinking_off_suffix: true,
            n_threads: 1,
        };
        let mut pieces = Vec::new();
        unsafe {
            ctx.generate(
                &messages,
                opts,
                None,
                std::ptr::null_mut(),
                &mut |s, last| {
                    pieces.push((s.to_string(), last));
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(
            pieces,
            vec![("fake reply".into(), false), ("".into(), true)]
        );
        assert_eq!(
            unsafe {
                ctx.generate(
                    &[ChatMessage {
                        role: "user".into(),
                        content: "cancel".into(),
                    }],
                    opts,
                    None,
                    std::ptr::null_mut(),
                    &mut |_, _| Ok(()),
                )
            },
            Err(LlamaError::Cancelled)
        );
        assert_eq!(
            unsafe {
                ctx.generate(
                    &[ChatMessage {
                        role: "user".into(),
                        content: "error".into(),
                    }],
                    opts,
                    None,
                    std::ptr::null_mut(),
                    &mut |_, _| Ok(()),
                )
            },
            Err(LlamaError::Failed("llama.cpp generate failed".into()))
        );
        assert!(
            unsafe { ctx.generate(&[], opts, None, std::ptr::null_mut(), &mut |_, _| Ok(())) }
                .is_err()
        );
        assert!(unsafe {
            ctx.generate(
                &[ChatMessage {
                    role: "bad\0role".into(),
                    content: "x".into(),
                }],
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());
        assert!(ctx
            .prompt_token_count_with_tools(&messages, false, "[]")
            .is_ok());
        assert!(unsafe {
            ctx.generate_with_tools(
                &messages,
                "[]",
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
        }
        .is_ok());
        assert!(unsafe {
            ctx.generate_with_tools(
                &messages,
                "bad\0tools",
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());
        assert!(ctx
            .prompt_token_count_with_lfm_tools(&messages, "[]")
            .is_ok());
        assert!(LlamaContext::render_lfm_prompt(&messages, "[]").is_ok());
        assert!(unsafe {
            ctx.generate_with_lfm_tools(
                &messages,
                "[]",
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
        }
        .is_ok());
        assert!(unsafe {
            ctx.generate_with_lfm_tools(
                &messages,
                "bad\0tools",
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_qwen_wrapper_without_weights() {
        assert!(
            QwenTtsContext::load("fail-qwen", "fake-mmproj", 1, 0, QwenTtsBackend::Cpu,).is_err()
        );
        let mut ctx = QwenTtsContext::load("fake-qwen", "fake-mmproj", 1, 0, QwenTtsBackend::Cpu)
            .expect("fake context");
        assert!(ctx.has_voice());
        assert!(unsafe { ctx.synthesize("", "en", None, std::ptr::null_mut()) }.is_err());
        assert!(unsafe { ctx.synthesize("bad\0text", "en", None, std::ptr::null_mut()) }.is_err());
        assert!(
            !unsafe { ctx.synthesize("hello", "en", None, std::ptr::null_mut()) }
                .unwrap()
                .1
                .is_empty()
        );
        assert_eq!(
            unsafe { ctx.synthesize("cancel", "en", None, std::ptr::null_mut()) },
            Err(QwenTtsError::Cancelled)
        );
        assert_eq!(
            unsafe { ctx.synthesize("error", "en", None, std::ptr::null_mut()) },
            Err(QwenTtsError::Failed("Qwen3-TTS synthesis failed".into()))
        );
        let mut windows = Vec::new();
        unsafe {
            ctx.synthesize_streaming(
                "hello",
                "en",
                None,
                std::ptr::null_mut(),
                &mut |rate, pcm, last| {
                    windows.push((rate, pcm.to_vec(), last));
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(windows.len(), 2);
        assert_eq!(
            unsafe {
                ctx.synthesize_streaming(
                    "cancel",
                    "en",
                    None,
                    std::ptr::null_mut(),
                    &mut |_, _, _| Ok(()),
                )
            },
            Err(QwenTtsError::Cancelled)
        );
        assert_eq!(
            unsafe {
                ctx.synthesize_streaming(
                    "error",
                    "en",
                    None,
                    std::ptr::null_mut(),
                    &mut |_, _, _| Ok(()),
                )
            },
            Err(QwenTtsError::Failed(
                "Qwen3-TTS streaming synthesis failed".into()
            ))
        );
        assert!(unsafe {
            ctx.synthesize_streaming(
                "bad\0text",
                "en",
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
            )
        }
        .is_err());
        assert_eq!(
            unsafe { qwen_pcm_callback(1, std::ptr::null(), 1, 0, std::ptr::null_mut()) },
            -1
        );
    }
}
