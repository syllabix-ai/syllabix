#include "syllabix_native.h"

#include "chat.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

static char *dup_cstr(const std::string & s) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out == NULL) {
        return NULL;
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

static void write_err(char *out, size_t cap, const char *msg) {
    if (out == NULL || cap == 0) {
        return;
    }
    std::snprintf(out, cap, "%s", msg != NULL ? msg : "s2s voice chat render error");
}

static bool split_audio_marker(const std::string & rendered, std::string & prefix, std::string & suffix) {
    static const char marker[] = "<|audio|>";
    const auto                 pos = rendered.find(marker);
    if (pos == std::string::npos) {
        return false;
    }
    prefix = rendered.substr(0, pos);
    suffix = rendered.substr(pos + sizeof(marker) - 1);
    return true;
}

extern "C" int syllabix_s2s_voice_chat_render(struct llama_model *model, const char *system_prompt,
                                              char **prefix_out, char **suffix_out, char *err_out,
                                              size_t err_cap) {
    if (model == NULL || system_prompt == NULL || prefix_out == NULL || suffix_out == NULL) {
        write_err(err_out, err_cap, "invalid s2s voice chat render arguments");
        return -1;
    }
    *prefix_out = NULL;
    *suffix_out = NULL;
    try {
        const char *tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr || tmpl_src[0] == '\0') {
            write_err(err_out, err_cap, "model has no default chat template");
            return -1;
        }
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, tmpl_src, "", "");
        if (!tmpls) {
            write_err(err_out, err_cap, "failed to initialize chat templates");
            return -1;
        }

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = false;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;

        common_chat_msg system_msg;
        system_msg.role    = "system";
        system_msg.content = system_prompt;

        common_chat_msg user_msg;
        user_msg.role    = "user";
        user_msg.content = "<|audio|>";

        inputs.messages = { system_msg, user_msg };

        const common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
        const std::string        rendered = params.prompt.str();
        std::string              prefix;
        std::string              suffix;
        if (!split_audio_marker(rendered, prefix, suffix)) {
            write_err(err_out, err_cap, "voice chat template did not render <|audio|> marker");
            return -1;
        }

        *prefix_out = dup_cstr(prefix);
        if (*prefix_out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        *suffix_out = dup_cstr(suffix);
        if (*suffix_out == NULL) {
            syllabix_chat_string_free(*prefix_out);
            *prefix_out = NULL;
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "s2s voice chat render failed");
        return -1;
    }
}

static const char S2S_RENDER_SENTINEL[] = "\x7fSYLLABIX_S2S_RENDER_SENTINEL\x7f";

extern "C" int syllabix_s2s_asr_render(struct llama_model *model, const char *language, char **prefix_out,
                                       char **suffix_out, char *err_out, size_t err_cap) {
    if (model == NULL || prefix_out == NULL || suffix_out == NULL) {
        write_err(err_out, err_cap, "invalid s2s asr render arguments");
        return -1;
    }
    *prefix_out = NULL;
    *suffix_out = NULL;
    try {
        const char *tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr || tmpl_src[0] == '\0') {
            write_err(err_out, err_cap, "model has no default chat template");
            return -1;
        }
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, tmpl_src, "", "");
        if (!tmpls) {
            write_err(err_out, err_cap, "failed to initialize chat templates");
            return -1;
        }

        const std::string lang = language != NULL ? language : "";
        const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls.get(), lang);

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = false;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;

        common_chat_msg system_msg;
        system_msg.role    = "system";
        system_msg.content = preset.system;

        common_chat_msg user_msg;
        user_msg.role    = "user";
        user_msg.content = std::string("<|audio|>") + preset.user;

        inputs.messages = { system_msg, user_msg };

        const common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
        const std::string        rendered = params.prompt.str();
        std::string              prefix;
        std::string              suffix;
        if (!split_audio_marker(rendered, prefix, suffix)) {
            write_err(err_out, err_cap, "asr template did not render <|audio|> marker");
            return -1;
        }

        *prefix_out = dup_cstr(prefix);
        if (*prefix_out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        *suffix_out = dup_cstr(suffix);
        if (*suffix_out == NULL) {
            syllabix_chat_string_free(*prefix_out);
            *prefix_out = NULL;
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "s2s asr render failed");
        return -1;
    }
}

extern "C" int syllabix_s2s_tts_render(struct llama_model *model, const char *text, const char *language,
                                         const char *voice, char **out, char *err_out, size_t err_cap) {
    if (model == NULL || text == NULL || out == NULL) {
        write_err(err_out, err_cap, "invalid s2s tts render arguments");
        return -1;
    }
    *out = NULL;
    try {
        const char *tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr || tmpl_src[0] == '\0') {
            write_err(err_out, err_cap, "model has no default chat template");
            return -1;
        }
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, tmpl_src, "", "");
        if (!tmpls) {
            write_err(err_out, err_cap, "failed to initialize chat templates");
            return -1;
        }

        const std::string lang = language != NULL ? language : "";
        const std::string voc  = voice != NULL ? voice : "";
        const std::string rendered =
            common_chat_get_tts_prompt(tmpls.get(), text, lang, voc);
        if (rendered.empty()) {
            write_err(err_out, err_cap, "model has no tts chat template");
            return -1;
        }

        *out = dup_cstr(rendered);
        if (*out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "s2s tts render failed");
        return -1;
    }
}

extern "C" int syllabix_s2s_voice_chat_render_next(struct llama_model *model, const char *system_prompt,
                                                   char **prefix_out, char **suffix_out, char *err_out,
                                                   size_t err_cap) {
    if (model == NULL || system_prompt == NULL || prefix_out == NULL || suffix_out == NULL) {
        write_err(err_out, err_cap, "invalid s2s voice chat render arguments");
        return -1;
    }
    *prefix_out = NULL;
    *suffix_out = NULL;
    try {
        const char *tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr || tmpl_src[0] == '\0') {
            write_err(err_out, err_cap, "model has no default chat template");
            return -1;
        }
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(model, tmpl_src, "", "");
        if (!tmpls) {
            write_err(err_out, err_cap, "failed to initialize chat templates");
            return -1;
        }

        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = false;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_NONE;

        common_chat_msg system_msg;
        system_msg.role    = "system";
        system_msg.content = system_prompt;

        common_chat_msg user1;
        user1.role    = "user";
        user1.content = "<|audio|>";

        common_chat_msg assistant_sentinel;
        assistant_sentinel.role    = "assistant";
        assistant_sentinel.content = S2S_RENDER_SENTINEL;

        common_chat_msg user2;
        user2.role    = "user";
        user2.content = "<|audio|>";

        inputs.messages = { system_msg, user1, assistant_sentinel, user2 };

        const common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
        const std::string        rendered = params.prompt.str();
        const auto                 pos = rendered.find(S2S_RENDER_SENTINEL);
        if (pos == std::string::npos) {
            write_err(err_out, err_cap, "voice chat continuation template missing sentinel");
            return -1;
        }
        const std::string tail = rendered.substr(pos + sizeof(S2S_RENDER_SENTINEL) - 1);
        std::string       prefix;
        std::string       suffix;
        if (!split_audio_marker(tail, prefix, suffix)) {
            write_err(err_out, err_cap, "voice chat continuation did not render <|audio|> marker");
            return -1;
        }

        *prefix_out = dup_cstr(prefix);
        if (*prefix_out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        *suffix_out = dup_cstr(suffix);
        if (*suffix_out == NULL) {
            syllabix_chat_string_free(*prefix_out);
            *prefix_out = NULL;
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "s2s voice chat render failed");
        return -1;
    }
}