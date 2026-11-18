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

    #[repr(C)]
    pub struct QwenAsrHandle {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct S2sHandle {
        _private: [u8; 0],
    }

    #[cfg(feature = "parakeet")]
    #[repr(C)]
    pub struct ParakeetHandle {
        _private: [u8; 0],
    }

    #[cfg(feature = "cpp")]
    #[repr(C)]
    pub struct MioHandle {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct ChatSessionHandle {
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
        pub fn syllabix_llama_chat_lfm_sampler_preset(
            llm: *const LlamaHandle,
            temp_out: *mut f32,
            top_k_out: *mut c_int,
            repeat_penalty_out: *mut f32,
        ) -> c_int;
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
        pub fn syllabix_chat_string_free(s: *mut c_char);
        pub fn syllabix_chat_session_new(
            llm: *mut LlamaHandle,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> *mut ChatSessionHandle;
        pub fn syllabix_chat_session_free(session: *mut ChatSessionHandle);
        pub fn syllabix_chat_tools_supported(session: *const ChatSessionHandle) -> c_int;
        pub fn syllabix_chat_last_format_name(session: *const ChatSessionHandle) -> *const c_char;
        pub fn syllabix_chat_session_from_template(
            template_src: *const c_char,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> *mut ChatSessionHandle;
        pub fn syllabix_chat_apply(
            session: *mut ChatSessionHandle,
            messages_json: *const c_char,
            tools_json: *const c_char,
            enable_thinking: c_int,
            prompt_out: *mut *mut c_char,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn syllabix_chat_parse(
            session: *mut ChatSessionHandle,
            text: *const c_char,
            is_partial: c_int,
            result_out: *mut *mut c_char,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> c_int;
        pub fn syllabix_chat_generate(
            session: *mut ChatSessionHandle,
            messages_json: *const c_char,
            tools_json: *const c_char,
            enable_thinking: c_int,
            thinking: c_int,
            n_threads: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            token_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            token_user: *mut c_void,
            err_out: *mut c_char,
            err_cap: usize,
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
        pub fn syllabix_qwen_asr_load(
            model_path: *const c_char,
            mmproj_path: *const c_char,
            n_threads: c_int,
            backend: c_int,
        ) -> *mut QwenAsrHandle;
        pub fn syllabix_qwen_asr_free(asr: *mut QwenAsrHandle);
        pub fn syllabix_qwen_asr_backend(asr: *const QwenAsrHandle) -> c_int;
        pub fn syllabix_qwen_asr_decode(
            asr: *mut QwenAsrHandle,
            pcm: *const f32,
            n_samples: c_int,
            language: *const c_char,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
        pub fn syllabix_s2s_load(
            model_path: *const c_char,
            mmproj_path: *const c_char,
            n_threads: c_int,
            seed: u32,
            n_ctx: c_int,
            voice_chat_system_prompt: *const c_char,
        ) -> *mut S2sHandle;
        pub fn syllabix_s2s_free(s2s: *mut S2sHandle);
        pub fn syllabix_s2s_in_rate(s2s: *const S2sHandle) -> c_int;
        pub fn syllabix_s2s_out_rate(s2s: *const S2sHandle) -> c_int;
        pub fn syllabix_s2s_n_gpu_layers(s2s: *const S2sHandle) -> c_int;
        pub fn syllabix_s2s_last_text(s2s: *const S2sHandle) -> *const c_char;
        pub fn syllabix_s2s_last_error(s2s: *const S2sHandle) -> *const c_char;
        pub fn syllabix_s2s_reset(s2s: *mut S2sHandle);
        pub fn syllabix_s2s_n_past(s2s: *const S2sHandle) -> c_int;
        pub fn syllabix_s2s_generate(
            s2s: *mut S2sHandle,
            pcm_in: *const f32,
            n_samples: c_int,
            max_out_samples: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            pcm_cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
            pcm_user: *mut c_void,
            text_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            text_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_s2s_transcribe(
            s2s: *mut S2sHandle,
            pcm_in: *const f32,
            n_samples: c_int,
            language: *const c_char,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            text_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
            text_user: *mut c_void,
        ) -> c_int;
        pub fn syllabix_s2s_speak(
            s2s: *mut S2sHandle,
            text: *const c_char,
            language: *const c_char,
            voice: *const c_char,
            max_out_samples: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            pcm_cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
            pcm_user: *mut c_void,
        ) -> c_int;
    }

    #[cfg(all(not(coverage), feature = "parakeet"))]
    extern "C" {
        pub fn syllabix_parakeet_load(path: *const c_char, n_threads: c_int)
            -> *mut ParakeetHandle;
        pub fn syllabix_parakeet_free(pk: *mut ParakeetHandle);
        pub fn syllabix_parakeet_set_threads(n_threads: c_int);
        pub fn syllabix_parakeet_model_loaded(pk: *const ParakeetHandle) -> c_int;
        pub fn syllabix_parakeet_transcribe(
            pk: *mut ParakeetHandle,
            pcm: *const f32,
            n_samples: c_int,
            out: *mut c_char,
            out_cap: c_int,
        ) -> c_int;
    }

    #[cfg(all(not(coverage), feature = "cpp"))]
    extern "C" {
        pub fn syllabix_mio_load(
            path: *const c_char,
            n_threads: c_int,
            n_gpu_layers: c_int,
        ) -> *mut MioHandle;
        pub fn syllabix_mio_free(mio: *mut MioHandle);
        pub fn syllabix_mio_generate(
            mio: *mut MioHandle,
            text: *const c_char,
            seed: u32,
            max_tokens: c_int,
            abort_cb: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
            abort_user: *mut c_void,
            code_cb: Option<unsafe extern "C" fn(i32, *mut c_void)>,
            code_user: *mut c_void,
        ) -> c_int;
    }

    // The coverage build must exercise the Rust safety/translation layer
    // without loading multi-gigabyte native models. Keep prompt rendering on
    // the real shim (it is pure and already weight-free); fake the calls that
    // require a live native context.
    #[cfg(coverage)]
    extern "C" {}

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
    static COVERAGE_S2S_MODEL: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

    #[cfg(coverage)]
    static COVERAGE_S2S_N_PAST: std::sync::Mutex<i32> = std::sync::Mutex::new(0);

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
    pub unsafe fn syllabix_llama_chat_lfm_sampler_preset(
        _llm: *const LlamaHandle,
        _temp_out: *mut f32,
        _top_k_out: *mut c_int,
        _repeat_penalty_out: *mut f32,
    ) -> c_int {
        0
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
    pub unsafe fn syllabix_chat_string_free(_s: *mut c_char) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_session_from_template(
        _template_src: *const c_char,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut ChatSessionHandle {
        if !err_out.is_null() && err_cap > 0 {
            *err_out = 0;
        }
        std::ptr::dangling_mut::<ChatSessionHandle>()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_session_new(
        _llm: *mut LlamaHandle,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut ChatSessionHandle {
        if !err_out.is_null() && err_cap > 0 {
            *err_out = 0;
        }
        std::ptr::dangling_mut::<ChatSessionHandle>()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_session_free(_session: *mut ChatSessionHandle) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_tools_supported(_session: *const ChatSessionHandle) -> c_int {
        1
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_last_format_name(
        _session: *const ChatSessionHandle,
    ) -> *const c_char {
        c"CONTENT_ONLY".as_ptr()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_apply(
        _session: *mut ChatSessionHandle,
        messages_json: *const c_char,
        _tools_json: *const c_char,
        _enable_thinking: c_int,
        prompt_out: *mut *mut c_char,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> c_int {
        if !err_out.is_null() && err_cap > 0 {
            *err_out = 0;
        }
        if prompt_out.is_null() {
            return -1;
        }
        let msg = text(messages_json);
        if msg.contains("apply-fail") {
            return -1;
        }
        let prompt = format!("fake-chat-prompt:{msg}");
        let out = std::ffi::CString::new(prompt).unwrap().into_raw();
        *prompt_out = out;
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_chat_parse(
        _session: *mut ChatSessionHandle,
        input: *const c_char,
        _is_partial: c_int,
        result_out: *mut *mut c_char,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> c_int {
        if !err_out.is_null() && err_cap > 0 {
            *err_out = 0;
        }
        if result_out.is_null() {
            return -1;
        }
        let body = text(input);
        if body.contains("parse-fail") {
            return -1;
        }
        let json = if body.contains("tool_call") {
            r#"{"content":"","tool_calls":[{"name":"shell","arguments":"{\"command\":\"pwd\"}","id":"local-call-0"}]}"#
        } else {
            r#"{"content":"fake reply","tool_calls":[]}"#
        };
        *result_out = std::ffi::CString::new(json).unwrap().into_raw();
        0
    }

    #[cfg(coverage)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_chat_generate(
        session: *mut ChatSessionHandle,
        messages_json: *const c_char,
        tools_json: *const c_char,
        _enable_thinking: c_int,
        thinking: c_int,
        n_threads: c_int,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> c_int {
        if !err_out.is_null() && err_cap > 0 {
            *err_out = 0;
        }
        let _ = (session, tools_json, thinking, n_threads);
        let msg = text(messages_json);
        if msg.contains("chat-error") {
            return -1;
        }
        if msg.contains("cancel") {
            return 1;
        }
        match cb {
            Some(cb) => {
                let piece = c"fake chat reply";
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

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_asr_load(
        model_path: *const c_char,
        _mmproj_path: *const c_char,
        _n_threads: c_int,
        _backend: c_int,
    ) -> *mut QwenAsrHandle {
        if fail(&text(model_path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<QwenAsrHandle>()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_asr_free(_asr: *mut QwenAsrHandle) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_asr_backend(_asr: *const QwenAsrHandle) -> c_int {
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_qwen_asr_decode(
        _asr: *mut QwenAsrHandle,
        pcm: *const f32,
        n_samples: c_int,
        language: *const c_char,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        out: *mut c_char,
        out_cap: c_int,
    ) -> c_int {
        if pcm.is_null() || n_samples < 1 || out.is_null() || out_cap < 2 {
            return -1;
        }
        let lang = text(language);
        if lang.contains("cancel") {
            return 1;
        }
        if lang.contains("error") {
            return -1;
        }
        let reply = if lang == "auto" {
            "language Spanish<asr_text>hola"
        } else {
            "hello from fake qwen asr"
        };
        let bytes = reply.as_bytes();
        let n = bytes.len().min(out_cap as usize - 1);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_load(
        model_path: *const c_char,
        _mmproj_path: *const c_char,
        _n_threads: c_int,
        _seed: u32,
        _n_ctx: c_int,
        _voice_chat_system_prompt: *const c_char,
    ) -> *mut S2sHandle {
        let path = text(model_path);
        if let Ok(mut model) = COVERAGE_S2S_MODEL.lock() {
            *model = path.clone();
        }
        if fail(&path, "fail") {
            std::ptr::null_mut()
        } else {
            if let Ok(mut past) = COVERAGE_S2S_N_PAST.lock() {
                *past = 0;
            }
            std::ptr::dangling_mut::<S2sHandle>()
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_free(_s2s: *mut S2sHandle) {}

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_in_rate(_s2s: *const S2sHandle) -> c_int {
        let path = COVERAGE_S2S_MODEL
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();
        if path.contains("in24k") {
            24_000
        } else {
            16_000
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_out_rate(_s2s: *const S2sHandle) -> c_int {
        24_000
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_n_gpu_layers(_s2s: *const S2sHandle) -> c_int {
        syllabix_llama_n_gpu_layers()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_last_text(_s2s: *const S2sHandle) -> *const c_char {
        static REPLY: &[u8] = b"fake s2s reply\0";
        REPLY.as_ptr().cast()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_last_error(_s2s: *const S2sHandle) -> *const c_char {
        std::ptr::null()
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_generate(
        _s2s: *mut S2sHandle,
        pcm: *const f32,
        n_samples: c_int,
        _max_out_samples: c_int,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
        text_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        text_user: *mut c_void,
    ) -> c_int {
        if pcm.is_null() || n_samples < 1 {
            return -1;
        }
        let path = COVERAGE_S2S_MODEL
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();
        if path.contains("nativeerr") {
            return -1;
        }
        if let Some(abort) = abort {
            if abort(abort_user) {
                return 1;
            }
        }
        let pcm_rate = if path.contains("badrate") {
            16_000
        } else {
            24_000
        };
        let first = [0.05f32, -0.05];
        if let Some(cb) = cb {
            if cb(pcm_rate, first.as_ptr(), 2, 0, user) != 0 {
                return 1;
            }
            if let Some(abort) = abort {
                if abort(abort_user) {
                    return 1;
                }
            }
            if cb(pcm_rate, std::ptr::null(), 0, 1, user) != 0 {
                return 1;
            }
        }
        if let Some(text_cb) = text_cb {
            const PIECES: &[&str] = &["fake ", "s2s ", "reply"];
            for piece in PIECES {
                let c_piece = CString::new(*piece).unwrap();
                if text_cb(c_piece.as_ptr(), 0, text_user) != 0 {
                    return 1;
                }
            }
            if text_cb(std::ptr::null(), 1, text_user) != 0 {
                return 1;
            }
        }
        if let Ok(mut past) = COVERAGE_S2S_N_PAST.lock() {
            *past += 37;
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_reset(_s2s: *mut S2sHandle) {
        if let Ok(mut past) = COVERAGE_S2S_N_PAST.lock() {
            *past = 0;
        }
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_n_past(_s2s: *const S2sHandle) -> c_int {
        COVERAGE_S2S_N_PAST.lock().map(|p| *p).unwrap_or(0)
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_transcribe(
        _s2s: *mut S2sHandle,
        pcm: *const f32,
        n_samples: c_int,
        language: *const c_char,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        text_cb: Option<unsafe extern "C" fn(*const c_char, c_int, *mut c_void) -> c_int>,
        text_user: *mut c_void,
    ) -> c_int {
        if pcm.is_null() || n_samples < 1 {
            return -1;
        }
        let path = COVERAGE_S2S_MODEL
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();
        if path.contains("nativeerr") {
            return -1;
        }
        if !language.is_null() {
            let lang = text(language);
            if lang.contains('\0') {
                return -1;
            }
        }
        if let Some(abort) = abort {
            if abort(abort_user) {
                return 1;
            }
        }
        if let Some(text_cb) = text_cb {
            const PIECES: &[&str] = &["fake ", "s2s ", "transcript"];
            for piece in PIECES {
                let c_piece = CString::new(*piece).unwrap();
                if text_cb(c_piece.as_ptr(), 0, text_user) != 0 {
                    return 1;
                }
            }
            if text_cb(std::ptr::null(), 1, text_user) != 0 {
                return 1;
            }
        }
        if let Ok(mut past) = COVERAGE_S2S_N_PAST.lock() {
            *past = 0;
        }
        0
    }

    #[cfg(coverage)]
    pub unsafe fn syllabix_s2s_speak(
        _s2s: *mut S2sHandle,
        prompt_ptr: *const c_char,
        language: *const c_char,
        voice: *const c_char,
        _max_out_samples: c_int,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        cb: Option<unsafe extern "C" fn(i32, *const f32, i64, c_int, *mut c_void) -> c_int>,
        user: *mut c_void,
    ) -> c_int {
        if prompt_ptr.is_null() || cb.is_none() {
            return -1;
        }
        let spoken = text(prompt_ptr);
        if spoken.is_empty() || spoken.contains('\0') {
            return -1;
        }
        if !language.is_null() && text(language).contains('\0') {
            return -1;
        }
        if !voice.is_null() && text(voice).contains('\0') {
            return -1;
        }
        let path = COVERAGE_S2S_MODEL
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default();
        if path.contains("nativeerr") {
            return -1;
        }
        if let Some(abort) = abort {
            if abort(abort_user) {
                return 1;
            }
        }
        let _ = (language, voice);
        let first = [0.12f32, -0.12];
        if let Some(cb) = cb {
            if cb(24_000, first.as_ptr(), 2, 0, user) != 0 {
                return 1;
            }
            if let Some(abort) = abort {
                if abort(abort_user) {
                    return 1;
                }
            }
            if cb(24_000, std::ptr::null(), 0, 1, user) != 0 {
                return 1;
            }
        }
        if let Ok(mut past) = COVERAGE_S2S_N_PAST.lock() {
            *past = 0;
        }
        0
    }

    #[cfg(all(coverage, feature = "parakeet"))]
    pub unsafe fn syllabix_parakeet_load(
        model_path: *const c_char,
        _n_threads: c_int,
    ) -> *mut ParakeetHandle {
        if fail(&text(model_path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<ParakeetHandle>()
        }
    }

    #[cfg(all(coverage, feature = "parakeet"))]
    pub unsafe fn syllabix_parakeet_free(_pk: *mut ParakeetHandle) {}

    #[cfg(all(coverage, feature = "parakeet"))]
    pub unsafe fn syllabix_parakeet_set_threads(_n_threads: c_int) {}

    #[cfg(all(coverage, feature = "parakeet"))]
    pub unsafe fn syllabix_parakeet_model_loaded(pk: *const ParakeetHandle) -> c_int {
        if pk.is_null() {
            0
        } else {
            1
        }
    }

    #[cfg(all(coverage, feature = "parakeet"))]
    pub unsafe fn syllabix_parakeet_transcribe(
        _pk: *mut ParakeetHandle,
        pcm: *const f32,
        n_samples: c_int,
        out: *mut c_char,
        out_cap: c_int,
    ) -> c_int {
        if pcm.is_null() || n_samples < 1 || out.is_null() || out_cap < 2 {
            return -1;
        }
        let reply = "hello from fake parakeet";
        let bytes = reply.as_bytes();
        let n = bytes.len().min(out_cap as usize - 1);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.cast::<u8>(), n);
        *out.add(n) = 0;
        0
    }

    #[cfg(all(coverage, feature = "cpp"))]
    pub unsafe fn syllabix_mio_load(
        path: *const c_char,
        _n_threads: c_int,
        _n_gpu_layers: c_int,
    ) -> *mut MioHandle {
        if fail(&text(path), "fail") {
            std::ptr::null_mut()
        } else {
            std::ptr::dangling_mut::<MioHandle>()
        }
    }

    #[cfg(all(coverage, feature = "cpp"))]
    pub unsafe fn syllabix_mio_free(_mio: *mut MioHandle) {}

    #[cfg(all(coverage, feature = "cpp"))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn syllabix_mio_generate(
        _mio: *mut MioHandle,
        text_ptr: *const c_char,
        _seed: u32,
        _max_tokens: c_int,
        _abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        _abort_user: *mut c_void,
        code_cb: Option<unsafe extern "C" fn(i32, *mut c_void)>,
        code_user: *mut c_void,
    ) -> c_int {
        let value = text(text_ptr);
        if value.contains("cancel") {
            return 1;
        }
        if value.contains("error") {
            return -1;
        }
        if let Some(cb) = code_cb {
            cb(0, code_user);
            cb(1, code_user);
        }
        0
    }
}

/// Resolved LFM chat sampler preset (template or GGUF metadata).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LfmChatSamplerPreset {
    /// Sampling temperature.
    pub temp: f32,
    /// Top-k limit.
    pub top_k: i32,
    /// Repeat penalty.
    pub repeat_penalty: f32,
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
    chat: Option<ChatSession>,
}

/// llama.cpp `common` chat session (Jinja apply + tool-call parse).
pub struct ChatSession {
    raw: *mut ffi::ChatSessionHandle,
}

/// One tool call parsed from model output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedChatToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

/// Assistant turn after `common_chat_parse`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ParsedChatOutput {
    pub content: String,
    pub tool_calls: Vec<ParsedChatToolCall>,
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
        Ok(Self { raw, chat: None })
    }

    fn chat_session(&mut self) -> Result<&mut ChatSession, LlamaError> {
        if self.chat.is_none() {
            let mut err = vec![0u8; 512];
            let _ggml = ggml_lock();
            let raw = unsafe {
                ffi::syllabix_chat_session_new(self.raw, err.as_mut_ptr().cast(), err.len())
            };
            if raw.is_null() {
                let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
                let message = String::from_utf8_lossy(&err[..end]).into_owned();
                return Err(LlamaError::Failed(if message.is_empty() {
                    "failed to open chat session".into()
                } else {
                    message
                }));
            }
            self.chat = Some(ChatSession { raw });
        }
        Ok(self.chat.as_mut().expect("chat session"))
    }

    pub fn chat_tools_supported(&mut self) -> Result<bool, LlamaError> {
        let session = self.chat_session()?;
        let _ggml = ggml_lock();
        Ok(unsafe { ffi::syllabix_chat_tools_supported(session.raw) != 0 })
    }

    pub fn chat_apply_prompt(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        enable_thinking: bool,
    ) -> Result<String, LlamaError> {
        let session = self.chat_session()?;
        chat_apply_prompt_raw(session.raw, messages_json, tools_json, enable_thinking)
    }

    pub fn chat_parse_output(
        &mut self,
        text: &str,
        is_partial: bool,
    ) -> Result<ParsedChatOutput, LlamaError> {
        let session = self.chat_session()?;
        chat_parse_output_raw(session.raw, text, is_partial)
    }

    pub fn chat_last_format_name(&mut self) -> Result<String, LlamaError> {
        let session = self.chat_session()?;
        let _ggml = ggml_lock();
        let name = unsafe { ffi::syllabix_chat_last_format_name(session.raw) };
        if name.is_null() {
            return Ok(String::new());
        }
        Ok(unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned())
    }

    #[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
    pub unsafe fn chat_generate(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        enable_thinking: bool,
        append_thinking_off_suffix: bool,
        opts: LlamaGenerate,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_piece: &mut dyn FnMut(&str, bool) -> Result<(), LlamaError>,
    ) -> Result<(), LlamaError> {
        let c_messages = CString::new(messages_json)
            .map_err(|_| LlamaError::Failed("messages_json NUL".into()))?;
        let c_tools =
            CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
        let session = self.chat_session()?;
        let mut err = vec![0u8; 512];
        let mut sink = TokenSink { on_piece };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_chat_generate(
                session.raw,
                c_messages.as_ptr(),
                c_tools.as_ptr(),
                i32::from(enable_thinking),
                if append_thinking_off_suffix { 0 } else { 1 },
                opts.n_threads,
                abort,
                abort_user,
                Some(on_token_piece),
                (&mut sink as *mut TokenSink).cast(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        match rc {
            0 => Ok(()),
            1 => Err(LlamaError::Cancelled),
            _ => {
                let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
                Err(LlamaError::Failed(
                    String::from_utf8_lossy(&err[..end]).into_owned(),
                ))
            }
        }
    }

    /// llama.cpp context size after load (`n_ctx_train` when `n_ctx` was 0).
    pub fn n_ctx(&self) -> i32 {
        unsafe { ffi::syllabix_llama_n_ctx(self.raw) }
    }

    /// GGUF trained context length.
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { ffi::syllabix_llama_n_ctx_train(self.raw) }
    }

    /// LiquidAI / GGUF `general.sampling.*` chat preset when present.
    pub fn chat_lfm_sampler_preset(&self) -> Option<LfmChatSamplerPreset> {
        let mut temp = 0.0f32;
        let mut top_k = 0;
        let mut repeat_penalty = 0.0f32;
        let ok = unsafe {
            ffi::syllabix_llama_chat_lfm_sampler_preset(
                self.raw,
                &mut temp,
                &mut top_k,
                &mut repeat_penalty,
            )
        };
        if ok == 0 {
            None
        } else {
            Some(LfmChatSamplerPreset {
                temp,
                top_k,
                repeat_penalty,
            })
        }
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

/// Chat apply/parse without loading GGUF weights (Jinja template string only).
pub struct ChatTemplateSession {
    raw: *mut ffi::ChatSessionHandle,
}

impl ChatTemplateSession {
    pub fn from_template(template: &str) -> Result<Self, LlamaError> {
        let c_template =
            CString::new(template).map_err(|_| LlamaError::Failed("template NUL".into()))?;
        let mut err = vec![0u8; 512];
        let _ggml = ggml_lock();
        let raw = unsafe {
            ffi::syllabix_chat_session_from_template(
                c_template.as_ptr(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if raw.is_null() {
            let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
            return Err(LlamaError::Failed(
                String::from_utf8_lossy(&err[..end]).into_owned(),
            ));
        }
        Ok(Self { raw })
    }

    pub fn apply_prompt(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        enable_thinking: bool,
    ) -> Result<String, LlamaError> {
        chat_apply_prompt_raw(self.raw, messages_json, tools_json, enable_thinking)
    }

    pub fn parse_output(
        &mut self,
        text: &str,
        is_partial: bool,
    ) -> Result<ParsedChatOutput, LlamaError> {
        chat_parse_output_raw(self.raw, text, is_partial)
    }
}

impl Drop for ChatTemplateSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_chat_session_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

fn chat_apply_prompt_raw(
    session: *mut ffi::ChatSessionHandle,
    messages_json: &str,
    tools_json: &str,
    enable_thinking: bool,
) -> Result<String, LlamaError> {
    let c_messages =
        CString::new(messages_json).map_err(|_| LlamaError::Failed("messages_json NUL".into()))?;
    let c_tools =
        CString::new(tools_json).map_err(|_| LlamaError::Failed("tools_json NUL".into()))?;
    let mut prompt_out: *mut c_char = std::ptr::null_mut();
    let mut err = vec![0u8; 512];
    let _ggml = ggml_lock();
    let rc = unsafe {
        ffi::syllabix_chat_apply(
            session,
            c_messages.as_ptr(),
            c_tools.as_ptr(),
            i32::from(enable_thinking),
            &mut prompt_out,
            err.as_mut_ptr().cast(),
            err.len(),
        )
    };
    if rc != 0 {
        let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
        return Err(LlamaError::Failed(
            String::from_utf8_lossy(&err[..end]).into_owned(),
        ));
    }
    let prompt = unsafe { CStr::from_ptr(prompt_out) }
        .to_string_lossy()
        .into_owned();
    unsafe { ffi::syllabix_chat_string_free(prompt_out) };
    Ok(prompt)
}

fn chat_parse_output_raw(
    session: *mut ffi::ChatSessionHandle,
    text: &str,
    is_partial: bool,
) -> Result<ParsedChatOutput, LlamaError> {
    let c_text = CString::new(text).map_err(|_| LlamaError::Failed("text NUL".into()))?;
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut err = vec![0u8; 512];
    let _ggml = ggml_lock();
    let rc = unsafe {
        ffi::syllabix_chat_parse(
            session,
            c_text.as_ptr(),
            i32::from(is_partial),
            &mut result_out,
            err.as_mut_ptr().cast(),
            err.len(),
        )
    };
    if rc != 0 {
        let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
        return Err(LlamaError::Failed(
            String::from_utf8_lossy(&err[..end]).into_owned(),
        ));
    }
    let json_text = unsafe { CStr::from_ptr(result_out) }
        .to_string_lossy()
        .into_owned();
    unsafe { ffi::syllabix_chat_string_free(result_out) };
    parse_chat_output_json(&json_text)
}

fn parse_chat_output_json(json_text: &str) -> Result<ParsedChatOutput, LlamaError> {
    let value: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|err| LlamaError::Failed(format!("chat parse json: {err}")))?;
    let content = value
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let mut tool_calls = Vec::new();
    if let Some(array) = value.get("tool_calls").and_then(|v| v.as_array()) {
        for entry in array {
            let name = entry
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = entry
                .get("arguments")
                .map(|v| {
                    if v.is_string() {
                        v.as_str().unwrap_or_default().to_string()
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| "{}".to_string());
            let id = entry
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            tool_calls.push(ParsedChatToolCall {
                name,
                arguments,
                id,
            });
        }
    }
    Ok(ParsedChatOutput {
        content,
        tool_calls,
    })
}

impl Drop for ChatSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_chat_session_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        drop(self.chat.take());
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

    pub(crate) fn as_c_int(self) -> c_int {
        match self {
            Self::Cpu => 0,
            Self::Metal => 1,
            Self::Vulkan => 2,
        }
    }

    pub(crate) fn from_c_int(value: c_int) -> Self {
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

/// LFM3-Audio S2S generation error from the native shim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lfm3S2sError {
    Cancelled,
    Failed(String),
}

type Lfm3PcmCallback<'a> = dyn FnMut(i32, &[f32], bool) -> Result<(), Lfm3S2sError> + 'a;
type Lfm3TextCallback<'a> = dyn FnMut(&str, bool) -> Result<(), Lfm3S2sError> + 'a;

/// In-process LFM3-Audio S2S context (backbone + mmproj).
pub struct Lfm3AudioContext {
    raw: *mut ffi::S2sHandle,
    in_rate: i32,
    out_rate: i32,
}

unsafe impl Send for Lfm3AudioContext {}

impl Lfm3AudioContext {
    /// Load backbone GGUF and mmproj. `seed` pins sampling (`u32::MAX` for runtime randomness).
    pub fn load(
        model_path: impl AsRef<Path>,
        mmproj_path: impl AsRef<Path>,
        n_threads: i32,
        seed: u32,
        n_ctx: u32,
        voice_chat_system_prompt: &str,
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
        let c_system = CString::new(voice_chat_system_prompt)
            .map_err(|_| "voice chat system prompt contains NUL".to_string())?;
        let _ggml = ggml_lock();
        let raw = unsafe {
            ffi::syllabix_s2s_load(
                c_model.as_ptr(),
                c_mmproj.as_ptr(),
                n_threads,
                seed,
                n_ctx as c_int,
                c_system.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(format!(
                "failed to load LFM3-Audio S2S at {} (mmproj {})",
                model_path.as_ref().display(),
                mmproj_path.as_ref().display()
            ));
        }
        let in_rate = unsafe { ffi::syllabix_s2s_in_rate(raw) };
        let out_rate = unsafe { ffi::syllabix_s2s_out_rate(raw) };
        Ok(Self {
            raw,
            in_rate,
            out_rate,
        })
    }

    pub fn in_rate_hz(&self) -> i32 {
        self.in_rate
    }

    pub fn out_rate_hz(&self) -> i32 {
        self.out_rate
    }

    pub fn n_gpu_layers(&self) -> i32 {
        unsafe { ffi::syllabix_s2s_n_gpu_layers(self.raw) }
    }

    /// Clear KV and conversation position.
    pub fn reset(&mut self) {
        let _ggml = ggml_lock();
        unsafe { ffi::syllabix_s2s_reset(self.raw) };
    }

    pub fn n_past(&self) -> i32 {
        unsafe { ffi::syllabix_s2s_n_past(self.raw) }
    }

    /// Text sampled during the last `generate_streaming`, including a cancelled partial turn.
    pub fn last_text(&self) -> &str {
        unsafe {
            let ptr = ffi::syllabix_s2s_last_text(self.raw);
            if ptr.is_null() {
                return "";
            }
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }

    pub fn last_error(&self) -> &str {
        unsafe {
            let ptr = ffi::syllabix_s2s_last_error(self.raw);
            if ptr.is_null() {
                return "";
            }
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("")
        }
    }

    /// Stream one utterance as native-rate f32 windows.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn generate_streaming(
        &mut self,
        pcm_in: &[f32],
        max_out_samples: i32,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_pcm: &mut Lfm3PcmCallback<'_>,
        on_text: &mut Lfm3TextCallback<'_>,
    ) -> Result<(), Lfm3S2sError> {
        if pcm_in.is_empty() {
            return Err(Lfm3S2sError::Failed("empty utterance pcm".into()));
        }
        let mut stream = Lfm3S2sStream {
            on_pcm,
            on_text,
            error: None,
        };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_s2s_generate(
                self.raw,
                pcm_in.as_ptr(),
                pcm_in.len() as c_int,
                max_out_samples,
                abort,
                abort_user,
                Some(lfm3_pcm_callback),
                (&mut stream as *mut Lfm3S2sStream<'_>).cast(),
                Some(lfm3_text_callback),
                (&mut stream as *mut Lfm3S2sStream<'_>).cast(),
            )
        };
        if let Some(err) = stream.error {
            return Err(err);
        }
        match rc {
            0 => Ok(()),
            1 => Err(Lfm3S2sError::Cancelled),
            _ => Err(Lfm3S2sError::Failed(
                "LFM3-Audio S2S generation failed".into(),
            )),
        }
    }

    /// Offline ASR on one utterance (resets KV before and after).
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn transcribe_streaming(
        &mut self,
        pcm_in: &[f32],
        language: Option<&str>,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_text: &mut Lfm3TextCallback<'_>,
    ) -> Result<(), Lfm3S2sError> {
        if pcm_in.is_empty() {
            return Err(Lfm3S2sError::Failed("empty utterance pcm".into()));
        }
        let c_lang = language
            .filter(|lang| !lang.is_empty())
            .map(|lang| {
                CString::new(lang).map_err(|_| Lfm3S2sError::Failed("language contains NUL".into()))
            })
            .transpose()?;
        let mut stream = Lfm3TranscribeStream {
            on_text,
            error: None,
        };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_s2s_transcribe(
                self.raw,
                pcm_in.as_ptr(),
                pcm_in.len() as c_int,
                c_lang
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
                abort,
                abort_user,
                Some(lfm3_transcribe_text_callback),
                (&mut stream as *mut Lfm3TranscribeStream<'_>).cast(),
            )
        };
        if let Some(err) = stream.error {
            return Err(err);
        }
        match rc {
            0 => Ok(()),
            1 => Err(Lfm3S2sError::Cancelled),
            _ => {
                let detail = self.last_error();
                Err(Lfm3S2sError::Failed(if detail.is_empty() {
                    "LFM3-Audio ASR transcription failed".into()
                } else {
                    detail.to_string()
                }))
            }
        }
    }

    /// TTS for one text prompt (resets KV before and after).
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn speak_streaming(
        &mut self,
        text: &str,
        language: Option<&str>,
        voice: Option<&str>,
        max_out_samples: i32,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
        on_pcm: &mut Lfm3PcmCallback<'_>,
    ) -> Result<(), Lfm3S2sError> {
        if text.is_empty() {
            return Err(Lfm3S2sError::Failed("empty speak text".into()));
        }
        let c_text =
            CString::new(text).map_err(|_| Lfm3S2sError::Failed("text contains NUL".into()))?;
        let c_lang = language
            .filter(|lang| !lang.is_empty())
            .map(|lang| {
                CString::new(lang).map_err(|_| Lfm3S2sError::Failed("language contains NUL".into()))
            })
            .transpose()?;
        let c_voice = voice
            .filter(|v| !v.is_empty())
            .map(|v| CString::new(v).map_err(|_| Lfm3S2sError::Failed("voice contains NUL".into())))
            .transpose()?;
        let mut stream = Lfm3SpeakStream {
            on_pcm,
            error: None,
        };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_s2s_speak(
                self.raw,
                c_text.as_ptr(),
                c_lang
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
                c_voice
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(std::ptr::null()),
                max_out_samples,
                abort,
                abort_user,
                Some(lfm3_speak_pcm_callback),
                (&mut stream as *mut Lfm3SpeakStream<'_>).cast(),
            )
        };
        if let Some(err) = stream.error {
            return Err(err);
        }
        match rc {
            0 => Ok(()),
            1 => Err(Lfm3S2sError::Cancelled),
            _ => {
                let detail = self.last_error();
                Err(Lfm3S2sError::Failed(if detail.is_empty() {
                    "LFM3-Audio TTS speak failed".into()
                } else {
                    detail.to_string()
                }))
            }
        }
    }
}

impl Drop for Lfm3AudioContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_s2s_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

struct Lfm3S2sStream<'a> {
    on_pcm: &'a mut Lfm3PcmCallback<'a>,
    on_text: &'a mut Lfm3TextCallback<'a>,
    error: Option<Lfm3S2sError>,
}

struct Lfm3TranscribeStream<'a> {
    on_text: &'a mut Lfm3TextCallback<'a>,
    error: Option<Lfm3S2sError>,
}

struct Lfm3SpeakStream<'a> {
    on_pcm: &'a mut Lfm3PcmCallback<'a>,
    error: Option<Lfm3S2sError>,
}

unsafe extern "C" fn lfm3_pcm_callback(
    sample_rate: i32,
    pcm: *const f32,
    n_samples: i64,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() || n_samples < 0 || (n_samples > 0 && pcm.is_null()) {
        return -1;
    }
    let stream = unsafe { &mut *(user as *mut Lfm3S2sStream<'_>) };
    let samples = if n_samples == 0 {
        &[]
    } else {
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

unsafe extern "C" fn lfm3_text_callback(
    text: *const c_char,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() {
        return -1;
    }
    let stream = unsafe { &mut *(user as *mut Lfm3S2sStream<'_>) };
    let piece = if text.is_null() {
        ""
    } else {
        unsafe { std::ffi::CStr::from_ptr(text).to_str().unwrap_or("") }
    };
    match (stream.on_text)(piece, is_last != 0) {
        Ok(()) => 0,
        Err(err) => {
            stream.error = Some(err);
            1
        }
    }
}

unsafe extern "C" fn lfm3_transcribe_text_callback(
    text: *const c_char,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() {
        return -1;
    }
    let stream = unsafe { &mut *(user as *mut Lfm3TranscribeStream<'_>) };
    let piece = if text.is_null() {
        ""
    } else {
        unsafe { std::ffi::CStr::from_ptr(text).to_str().unwrap_or("") }
    };
    match (stream.on_text)(piece, is_last != 0) {
        Ok(()) => 0,
        Err(err) => {
            stream.error = Some(err);
            1
        }
    }
}

unsafe extern "C" fn lfm3_speak_pcm_callback(
    sample_rate: i32,
    pcm: *const f32,
    n_samples: i64,
    is_last: c_int,
    user: *mut c_void,
) -> c_int {
    if user.is_null() || n_samples < 0 || (n_samples > 0 && pcm.is_null()) {
        return -1;
    }
    let stream = unsafe { &mut *(user as *mut Lfm3SpeakStream<'_>) };
    let samples = if n_samples == 0 {
        &[]
    } else {
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

/// In-process Qwen3-ASR context: decoder GGUF + qwen3a mmproj through the
/// shared ggml. One instance transcribes utterances sequentially.
pub struct QwenAsrContext {
    raw: *mut ffi::QwenAsrHandle,
}

unsafe impl Send for QwenAsrContext {}

impl QwenAsrContext {
    /// Load the decoder GGUF and audio encoder mmproj with an explicit
    /// [`QwenTtsBackend`] placement (same ids as Qwen3-TTS).
    pub fn load(
        model_path: impl AsRef<Path>,
        mmproj_path: impl AsRef<Path>,
        n_threads: i32,
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
            ffi::syllabix_qwen_asr_load(
                c_model.as_ptr(),
                c_mmproj.as_ptr(),
                n_threads,
                backend.as_c_int(),
            )
        };
        if raw.is_null() {
            return Err(format!(
                "failed to load Qwen3-ASR at {} (mmproj {})",
                model_path.as_ref().display(),
                mmproj_path.as_ref().display()
            ));
        }
        Ok(Self { raw })
    }

    /// Compute backend confirmed by the native context after load.
    pub fn backend(&self) -> QwenTtsBackend {
        let _ggml = ggml_lock();
        QwenTtsBackend::from_c_int(unsafe { ffi::syllabix_qwen_asr_backend(self.raw) })
    }

    /// Decode one 16 kHz mono f32 utterance.
    ///
    /// `language` is an English display name (`English`) or `auto`.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn decode(
        &mut self,
        pcm: &[f32],
        language: &str,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<String, DecodeError> {
        let c_lang = CString::new(language)
            .map_err(|_| DecodeError::Failed("language contains NUL".into()))?;
        let mut out = vec![0u8; 8192];
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_qwen_asr_decode(
                self.raw,
                pcm.as_ptr(),
                pcm.len() as c_int,
                c_lang.as_ptr(),
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
            _ => Err(DecodeError::Failed("Qwen3-ASR decode failed".into())),
        }
    }
}

impl Drop for QwenAsrContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_qwen_asr_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Fluid Speech Pro / Parakeet GGUF through vendored parakeet.cpp.
#[cfg(feature = "parakeet")]
pub struct ParakeetContext {
    raw: *mut ffi::ParakeetHandle,
}

#[cfg(feature = "parakeet")]
unsafe impl Send for ParakeetContext {}

#[cfg(feature = "parakeet")]
impl ParakeetContext {
    pub fn load(path: impl AsRef<Path>, n_threads: i32) -> Result<Self, String> {
        hush_logs();
        unsafe { ffi::syllabix_llama_backend_init() };
        let c_path = path
            .as_ref()
            .to_str()
            .ok_or_else(|| format!("path is not valid UTF-8: {}", path.as_ref().display()))?;
        let c_path =
            CString::new(c_path).map_err(|_| "path contains an interior NUL".to_string())?;
        let _ggml = ggml_lock();
        let raw = unsafe { ffi::syllabix_parakeet_load(c_path.as_ptr(), n_threads) };
        if raw.is_null() {
            return Err(format!(
                "failed to load Parakeet GGUF at {}",
                path.as_ref().display()
            ));
        }
        if unsafe { ffi::syllabix_parakeet_model_loaded(raw) } == 0 {
            let mut out = vec![0u8; 8192];
            let _ = unsafe {
                ffi::syllabix_parakeet_transcribe(
                    raw,
                    [0.0f32].as_ptr(),
                    1,
                    out.as_mut_ptr().cast::<c_char>(),
                    out.len() as c_int,
                )
            };
            let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
            let detail = String::from_utf8_lossy(&out[..end]).into_owned();
            unsafe { ffi::syllabix_parakeet_free(raw) };
            return Err(if detail.is_empty() {
                format!(
                    "failed to load Parakeet GGUF at {}",
                    path.as_ref().display()
                )
            } else {
                format!(
                    "failed to load Parakeet GGUF at {} ({detail})",
                    path.as_ref().display()
                )
            });
        }
        Ok(Self { raw })
    }

    pub fn set_threads(&self, n_threads: i32) {
        let _ggml = ggml_lock();
        unsafe { ffi::syllabix_parakeet_set_threads(n_threads) };
    }

    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String, String> {
        let mut out = vec![0u8; 8192];
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_parakeet_transcribe(
                self.raw,
                pcm.as_ptr(),
                pcm.len() as c_int,
                out.as_mut_ptr().cast::<c_char>(),
                out.len() as c_int,
            )
        };
        let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
        let message = String::from_utf8_lossy(&out[..end]).into_owned();
        if rc == 0 {
            Ok(message)
        } else {
            Err(if message.is_empty() {
                "Parakeet transcribe failed".into()
            } else {
                message
            })
        }
    }
}

#[cfg(feature = "parakeet")]
impl Drop for ParakeetContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_parakeet_free(self.raw) };
            self.raw = std::ptr::null_mut();
        }
    }
}

#[cfg(feature = "cpp")]
#[derive(Debug, PartialEq, Eq)]
pub enum MioError {
    Cancelled,
    Failed(String),
}

#[cfg(feature = "cpp")]
struct MioCodeSink {
    codes: Vec<i32>,
}

#[cfg(feature = "cpp")]
unsafe extern "C" fn mio_code_callback(code: i32, user: *mut c_void) {
    if user.is_null() {
        return;
    }
    // Safety: `generate` keeps this stack value alive for the C call.
    let sink = unsafe { &mut *(user as *mut MioCodeSink) };
    sink.codes.push(code);
}

/// Indic-Mio Qwen3 GGUF context. Each `generate` call clears the KV cache
/// and samples speech-token codes; the codec lives in `syllabix-tts`.
#[cfg(feature = "cpp")]
pub struct MioLmContext {
    raw: *mut ffi::MioHandle,
    backend: QwenTtsBackend,
}

#[cfg(feature = "cpp")]
unsafe impl Send for MioLmContext {}

#[cfg(feature = "cpp")]
impl MioLmContext {
    /// Load the Indic-Mio GGUF. `backend` follows [`QwenTtsBackend`]: CPU
    /// keeps `n_gpu_layers = 0`; Metal/Vulkan pass `-1`.
    pub fn load(
        model_path: impl AsRef<Path>,
        n_threads: i32,
        backend: QwenTtsBackend,
    ) -> Result<Self, String> {
        hush_logs();
        unsafe { ffi::syllabix_llama_backend_init() };
        let path = model_path.as_ref();
        let s = path
            .to_str()
            .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))?;
        let c_path = CString::new(s).map_err(|_| "path contains an interior NUL".to_string())?;
        let n_gpu_layers = if backend == QwenTtsBackend::Cpu {
            0
        } else {
            -1
        };
        let _ggml = ggml_lock();
        let raw = unsafe { ffi::syllabix_mio_load(c_path.as_ptr(), n_threads, n_gpu_layers) };
        if raw.is_null() {
            return Err(format!("failed to load Indic-Mio at {}", path.display()));
        }
        Ok(Self { raw, backend })
    }

    pub fn backend(&self) -> QwenTtsBackend {
        self.backend
    }

    /// Sample speech codes for one sentence.
    ///
    /// # Safety
    /// `abort_user` must remain valid for the duration of the call when `abort` is `Some`.
    pub unsafe fn generate(
        &mut self,
        text: &str,
        seed: u32,
        max_tokens: i32,
        abort: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
        abort_user: *mut c_void,
    ) -> Result<Vec<i32>, MioError> {
        if text.is_empty() {
            return Err(MioError::Failed("empty speak text".into()));
        }
        let c_text =
            CString::new(text).map_err(|_| MioError::Failed("text contains NUL".into()))?;
        let mut sink = MioCodeSink { codes: Vec::new() };
        let _ggml = ggml_lock();
        let rc = unsafe {
            ffi::syllabix_mio_generate(
                self.raw,
                c_text.as_ptr(),
                seed,
                max_tokens,
                abort,
                abort_user,
                Some(mio_code_callback),
                (&mut sink as *mut MioCodeSink).cast(),
            )
        };
        match rc {
            0 => Ok(sink.codes),
            1 => Err(MioError::Cancelled),
            _ => Err(MioError::Failed("Indic-Mio generation failed".into())),
        }
    }
}

#[cfg(feature = "cpp")]
impl Drop for MioLmContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ggml = ggml_lock();
            unsafe { ffi::syllabix_mio_free(self.raw) };
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
    #[cfg(all(not(coverage), feature = "cpp"))]
    fn missing_mio_weights_do_not_load() {
        let err = match MioLmContext::load("/no/such/indic-mio.gguf", 1, QwenTtsBackend::Cpu) {
            Err(err) => err,
            Ok(_) => panic!("missing Indic-Mio weights should fail"),
        };
        assert!(err.contains("failed to load Indic-Mio"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_lfm3_s2s_weights_do_not_load() {
        const VOICE_CHAT_SYSTEM: &str = "Respond with interleaved text and audio.";
        let err = match Lfm3AudioContext::load(
            "/no/such/lfm3-audio.gguf",
            "/no/such/mmproj.gguf",
            1,
            0,
            0,
            VOICE_CHAT_SYSTEM,
        ) {
            Err(err) => err,
            Ok(_) => panic!("missing LFM3-Audio weights should fail"),
        };
        assert!(err.contains("failed to load LFM3-Audio S2S"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn missing_qwen_asr_weights_do_not_load() {
        let err = match QwenAsrContext::load(
            "/no/such/qwen3-asr.gguf",
            "/no/such/mmproj.gguf",
            1,
            QwenTtsBackend::Cpu,
        ) {
            Err(err) => err,
            Ok(_) => panic!("missing Qwen3-ASR weights should fail"),
        };
        assert!(err.contains("failed to load Qwen3-ASR"));
    }

    #[test]
    #[cfg(all(not(coverage), feature = "parakeet"))]
    fn missing_parakeet_weights_do_not_load() {
        let err = match ParakeetContext::load("/no/such/fluid-speech-pro.gguf", 1) {
            Err(err) => err,
            Ok(_) => panic!("missing Parakeet weights should fail"),
        };
        assert!(err.contains("failed to load Parakeet GGUF"));
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
        assert!(ctx.chat_tools_supported().unwrap());
        let messages_json = r#"[{"role":"user","content":"hello"}]"#;
        let prompt = ctx
            .chat_apply_prompt(messages_json, "[]", false)
            .expect("chat apply");
        assert!(prompt.contains("fake-chat-prompt"));
        assert!(ctx.chat_apply_prompt("bad\0json", "[]", false).is_err());
        let tool_parse = ctx
            .chat_parse_output("tool_call", false)
            .expect("chat parse tools");
        assert_eq!(tool_parse.tool_calls.len(), 1);
        let plain = ctx
            .chat_parse_output("hello", false)
            .expect("chat parse content");
        assert_eq!(plain.content, "fake reply");
        assert!(ctx.chat_parse_output("parse-fail", false).is_err());
        assert!(ctx.chat_last_format_name().unwrap().contains("CONTENT"));
        unsafe {
            ctx.chat_generate(
                messages_json,
                "[]",
                false,
                true,
                opts,
                None,
                std::ptr::null_mut(),
                &mut |_, _| Ok(()),
            )
            .expect("chat generate");
            assert!(ctx
                .chat_generate(
                    r#"chat-error"#,
                    "[]",
                    false,
                    true,
                    opts,
                    None,
                    std::ptr::null_mut(),
                    &mut |_, _| Ok(()),
                )
                .is_err());
        }
        let mut tmpl =
            ChatTemplateSession::from_template("fake-template").expect("template session");
        assert!(tmpl
            .apply_prompt(messages_json, "[]", false)
            .expect("template apply")
            .contains("fake-chat-prompt"));
        let parsed = tmpl
            .parse_output("tool_call", false)
            .expect("template parse");
        assert_eq!(parsed.tool_calls.len(), 1);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_lfm3_s2s_wrapper_without_weights() {
        const VOICE_CHAT_SYSTEM: &str = "Respond with interleaved text and audio.";
        assert!(
            Lfm3AudioContext::load("fail-s2s", "fake-mmproj", 1, 0, 0, VOICE_CHAT_SYSTEM).is_err()
        );
        let mut ctx = Lfm3AudioContext::load("fake-s2s", "fake-mmproj", 1, 0, 0, VOICE_CHAT_SYSTEM)
            .expect("fake context");
        assert_eq!(ctx.in_rate_hz(), 16_000);
        assert_eq!(ctx.out_rate_hz(), 24_000);
        assert!(unsafe {
            ctx.generate_streaming(
                &[],
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());
        let mut windows = Vec::new();
        unsafe {
            let mut text_pieces = Vec::new();
            ctx.generate_streaming(
                &[0.0, 0.1],
                0,
                None,
                std::ptr::null_mut(),
                &mut |rate, pcm, last| {
                    windows.push((rate, pcm.to_vec(), last));
                    Ok(())
                },
                &mut |piece, last| {
                    text_pieces.push((piece.to_string(), last));
                    Ok(())
                },
            )
            .unwrap();
            let joined: String = text_pieces.iter().map(|(s, _)| s.as_str()).collect();
            assert_eq!(joined, "fake s2s reply");
            assert!(text_pieces.last().is_some_and(|(_, last)| *last));
        }
        assert_eq!(windows.len(), 2);
        assert!(windows[1].1.is_empty());
        assert!(windows[1].2);
        assert_eq!(ctx.n_past(), 37);
        ctx.reset();
        assert_eq!(ctx.n_past(), 0);
        unsafe {
            ctx.generate_streaming(
                &[0.0],
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
                &mut |_, _| Ok(()),
            )
            .unwrap();
        }
        assert_eq!(ctx.n_past(), 37);
        let cancel = crate::Cancel::new();
        let cancel_ptr = &cancel as *const crate::Cancel as *mut c_void;
        unsafe extern "C" fn abort_now(user: *mut c_void) -> bool {
            !user.is_null()
        }
        assert_eq!(
            unsafe {
                ctx.generate_streaming(
                    &[0.0],
                    0,
                    Some(abort_now),
                    cancel_ptr,
                    &mut |_, _, _| Ok(()),
                    &mut |_, _| Ok(()),
                )
            },
            Err(Lfm3S2sError::Cancelled)
        );
        assert!(unsafe {
            ctx.generate_streaming(
                &[0.0],
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Err(Lfm3S2sError::Failed("pcm callback".into())),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());

        assert!(unsafe {
            ctx.transcribe_streaming(&[], None, None, std::ptr::null_mut(), &mut |_, _| Ok(()))
        }
        .is_err());
        let mut asr_pieces = Vec::new();
        unsafe {
            ctx.transcribe_streaming(
                &[0.0, 0.1],
                Some("en"),
                None,
                std::ptr::null_mut(),
                &mut |piece, last| {
                    asr_pieces.push((piece.to_string(), last));
                    Ok(())
                },
            )
            .unwrap();
        }
        let asr_joined: String = asr_pieces.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(asr_joined, "fake s2s transcript");
        assert!(asr_pieces.last().is_some_and(|(_, last)| *last));
        assert_eq!(ctx.n_past(), 0);

        assert!(unsafe {
            ctx.speak_streaming(
                "",
                None,
                None,
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
            )
        }
        .is_err());
        assert!(unsafe {
            ctx.speak_streaming(
                "bad\0text",
                None,
                None,
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
            )
        }
        .is_err());
        let mut speak_windows = Vec::new();
        unsafe {
            ctx.speak_streaming(
                "hello",
                Some("en"),
                Some("female"),
                0,
                None,
                std::ptr::null_mut(),
                &mut |rate, pcm, last| {
                    speak_windows.push((rate, pcm.to_vec(), last));
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(speak_windows.len(), 2);
        assert!(speak_windows[1].1.is_empty());
        assert!(speak_windows[1].2);
        assert_eq!(ctx.n_past(), 0);
        assert_eq!(
            unsafe {
                ctx.transcribe_streaming(&[0.0], None, Some(abort_now), cancel_ptr, &mut |_, _| {
                    Ok(())
                })
            },
            Err(Lfm3S2sError::Cancelled)
        );
        assert!(unsafe {
            ctx.speak_streaming(
                "hello",
                None,
                None,
                0,
                Some(abort_now),
                cancel_ptr,
                &mut |_, _, _| Ok(()),
            )
        }
        .is_err());
        assert!(unsafe {
            ctx.speak_streaming(
                "hello",
                None,
                None,
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Err(Lfm3S2sError::Failed("pcm callback".into())),
            )
        }
        .is_err());
        assert!(unsafe {
            ctx.transcribe_streaming(&[0.0], None, None, std::ptr::null_mut(), &mut |_, _| {
                Err(Lfm3S2sError::Failed("text callback".into()))
            })
        }
        .is_err());
        let mut err_ctx = Lfm3AudioContext::load(
            "fake-s2s-nativeerr",
            "fake-mmproj",
            1,
            0,
            0,
            "Respond with interleaved text and audio.",
        )
        .expect("nativeerr");
        assert!(unsafe {
            err_ctx.generate_streaming(
                &[0.0],
                0,
                None,
                std::ptr::null_mut(),
                &mut |_, _, _| Ok(()),
                &mut |_, _| Ok(()),
            )
        }
        .is_err());
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_exercises_qwen_asr_wrapper_without_weights() {
        assert!(
            QwenAsrContext::load("fail-qwen-asr", "fake-mmproj", 1, QwenTtsBackend::Cpu).is_err()
        );
        let mut ctx = QwenAsrContext::load("fake-qwen-asr", "fake-mmproj", 1, QwenTtsBackend::Cpu)
            .expect("fake context");
        assert!(unsafe { ctx.decode(&[], "en", None, std::ptr::null_mut()) }.is_err());
        assert!(unsafe { ctx.decode(&[0.0], "bad\0lang", None, std::ptr::null_mut()) }.is_err());
        assert_eq!(
            unsafe { ctx.decode(&[0.0], "English", None, std::ptr::null_mut()) }.unwrap(),
            "hello from fake qwen asr"
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], "auto", None, std::ptr::null_mut()) }.unwrap(),
            "language Spanish<asr_text>hola"
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], "cancel", None, std::ptr::null_mut()) },
            Err(DecodeError::Cancelled)
        );
        assert_eq!(
            unsafe { ctx.decode(&[0.0], "error", None, std::ptr::null_mut()) },
            Err(DecodeError::Failed("Qwen3-ASR decode failed".into()))
        );
    }

    #[cfg(all(coverage, feature = "parakeet"))]
    #[test]
    fn coverage_exercises_parakeet_wrapper_without_weights() {
        assert!(ParakeetContext::load("fail-parakeet", 1).is_err());
        let mut ctx = ParakeetContext::load("fake-parakeet", 1).expect("fake context");
        ctx.set_threads(4);
        assert!(ctx.transcribe(&[]).is_err());
        assert_eq!(ctx.transcribe(&[0.0]).unwrap(), "hello from fake parakeet");
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

    #[cfg(all(coverage, feature = "cpp"))]
    #[test]
    fn coverage_exercises_mio_wrapper_without_weights() {
        assert!(MioLmContext::load("fail-mio", 1, QwenTtsBackend::Cpu).is_err());
        let mut ctx = MioLmContext::load("fake-mio", 1, QwenTtsBackend::Cpu).expect("fake context");
        assert_eq!(ctx.backend(), QwenTtsBackend::Cpu);
        assert!(unsafe { ctx.generate("", 1, 8, None, std::ptr::null_mut()) }.is_err());
        assert!(unsafe { ctx.generate("bad\0text", 1, 8, None, std::ptr::null_mut()) }.is_err());
        assert_eq!(
            unsafe { ctx.generate("hello", 1, 8, None, std::ptr::null_mut()) }.unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            unsafe { ctx.generate("cancel", 1, 8, None, std::ptr::null_mut()) },
            Err(MioError::Cancelled)
        );
        assert_eq!(
            unsafe { ctx.generate("error", 1, 8, None, std::ptr::null_mut()) },
            Err(MioError::Failed("Indic-Mio generation failed".into()))
        );
    }
}
