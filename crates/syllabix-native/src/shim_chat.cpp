#include "syllabix_native.h"

#include "chat.h"
#include "common.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct syllabix_llama;
struct llama_model *syllabix_llama_model(struct syllabix_llama *llm);

struct syllabix_chat_session {
    struct syllabix_llama *              llm;
    common_chat_templates_ptr            tmpls;
    common_chat_parser_params            parser_params;
    std::string                          last_format;
    bool                                 tools_supported;
    /* Single-token ids of the template's preserved tokens; generation
     * renders these as text so the parser sees tool-call markers. */
    std::vector<int32_t>                 render_special;
};

static void write_err(char *out, size_t cap, const char *msg) {
    if (out == NULL || cap == 0) {
        return;
    }
    std::snprintf(out, cap, "%s", msg != NULL ? msg : "chat error");
}

static char *dup_cstr(const std::string & s) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out == NULL) {
        return NULL;
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

void syllabix_chat_string_free(char *s) {
    std::free(s);
}

static bool template_has_tools(const common_chat_templates * tmpls) {
    const auto caps = common_chat_templates_get_caps(tmpls);
    const auto it   = caps.find("supports_tools");
    if (it != caps.end() && !it->second) {
        return false;
    }
    try {
        common_chat_templates_inputs inputs;
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = false;
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = "ping";
        inputs.messages = { msg };
        common_chat_tool tool;
        tool.name        = "shell";
        tool.description = "run";
        tool.parameters  = R"({"type":"object","properties":{"command":{"type":"string"}}})";
        inputs.tools = { tool };
        const auto params = common_chat_templates_apply(tmpls, inputs);
        if (params.format == COMMON_CHAT_FORMAT_CONTENT_ONLY && params.parser.empty()) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

static void apply_parser_from_params(syllabix_chat_session * session, const common_chat_params & params) {
    session->parser_params                    = common_chat_parser_params(params);
    session->parser_params.parse_tool_calls   = true;
    session->parser_params.reasoning_format   = COMMON_REASONING_FORMAT_DEEPSEEK;
    session->last_format                      = common_chat_format_name(params.format);
    if (!params.parser.empty()) {
        session->parser_params.parser.load(params.parser);
    }
}

static std::unique_ptr<syllabix_chat_session> init_chat_session(
    struct syllabix_llama *llm, common_chat_templates_ptr tmpls, char *err_out, size_t err_cap) {
    if (!tmpls) {
        write_err(err_out, err_cap, "failed to initialize chat templates");
        return nullptr;
    }
    const std::string src = common_chat_templates_source(tmpls.get());
    if (src.empty()) {
        write_err(err_out, err_cap,
                  "chat template is empty; tool calling needs a Jinja chat_template");
        return nullptr;
    }
    auto session             = std::make_unique<syllabix_chat_session>();
    session->llm             = llm;
    session->tmpls           = std::move(tmpls);
    session->tools_supported = template_has_tools(session->tmpls.get());
    session->last_format.clear();
    return session;
}

struct syllabix_chat_session *syllabix_chat_session_new(struct syllabix_llama *llm, char *err_out,
                                                        size_t err_cap) {
    if (llm == NULL) {
        write_err(err_out, err_cap, "chat session requires a loaded model");
        return NULL;
    }
    try {
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(syllabix_llama_model(llm), "", "", "");
        auto session = init_chat_session(llm, std::move(tmpls), err_out, err_cap);
        return session ? session.release() : NULL;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return NULL;
    } catch (...) {
        write_err(err_out, err_cap, "failed to initialize chat templates");
        return NULL;
    }
}

struct syllabix_chat_session *syllabix_chat_session_from_template(const char *template_src,
                                                                  char *err_out, size_t err_cap) {
    if (template_src == NULL || template_src[0] == '\0') {
        write_err(err_out, err_cap, "chat template override is required");
        return NULL;
    }
    try {
        common_chat_templates_ptr tmpls =
            common_chat_templates_init(/* model= */ nullptr, template_src, "", "");
        auto session = init_chat_session(/* llm= */ nullptr, std::move(tmpls), err_out, err_cap);
        return session ? session.release() : NULL;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return NULL;
    } catch (...) {
        write_err(err_out, err_cap, "failed to initialize chat templates");
        return NULL;
    }
}

void syllabix_chat_session_free(struct syllabix_chat_session *session) {
    delete session;
}

int syllabix_chat_tools_supported(const struct syllabix_chat_session *session) {
    return session != NULL && session->tools_supported ? 1 : 0;
}

const char *syllabix_chat_last_format_name(const struct syllabix_chat_session *session) {
    if (session == NULL || session->last_format.empty()) {
        return "";
    }
    return session->last_format.c_str();
}

static json parse_json_or_throw(const char *text, const char *what) {
    if (text == NULL) {
        throw std::invalid_argument(std::string("missing ") + what);
    }
    return json::parse(text);
}

int syllabix_chat_apply(struct syllabix_chat_session *session, const char *messages_json,
                        const char *tools_json, int enable_thinking, char **prompt_out,
                        char *err_out, size_t err_cap) {
    if (session == NULL || messages_json == NULL || prompt_out == NULL) {
        write_err(err_out, err_cap, "invalid chat apply arguments");
        return -1;
    }
    *prompt_out = NULL;
    try {
        const json messages = parse_json_or_throw(messages_json, "messages_json");
        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.use_jinja             = true;
        inputs.add_generation_prompt = true;
        inputs.parallel_tool_calls   = false;
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.enable_thinking       = enable_thinking != 0;
        // Match llama-server's default (--reasoning-format auto/deepseek):
        // templates that force-open <think> (LFM2.5) must have the reasoning
        // split off, or it leaks into spoken content.
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (tools_json != NULL && tools_json[0] != '\0') {
            inputs.tools = common_chat_tools_parse_oaicompat(parse_json_or_throw(tools_json, "tools_json"));
        }
        const common_chat_params params =
            common_chat_templates_apply(session->tmpls.get(), inputs);
        apply_parser_from_params(session, params);
        session->render_special.clear();
        if (session->llm != NULL) {
            const llama_vocab * vocab = llama_model_get_vocab(syllabix_llama_model(session->llm));
            for (const auto & t : params.preserved_tokens) {
                const auto ids = common_tokenize(vocab, t, false, true);
                if (ids.size() == 1) {
                    session->render_special.push_back(ids[0]);
                }
            }
        }
        *prompt_out = dup_cstr(params.prompt);
        if (*prompt_out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "chat apply failed");
        return -1;
    }
}

static json tool_calls_to_json(const common_chat_msg & msg) {
    json calls = json::array();
    for (const auto & tc : msg.tool_calls) {
        json entry;
        entry["name"]      = tc.name;
        entry["arguments"] = tc.arguments;
        entry["id"]        = tc.id;
        calls.push_back(std::move(entry));
    }
    return calls;
}

int syllabix_chat_parse(struct syllabix_chat_session *session, const char *text, int is_partial,
                        char **result_out, char *err_out, size_t err_cap) {
    if (session == NULL || text == NULL || result_out == NULL) {
        write_err(err_out, err_cap, "invalid chat parse arguments");
        return -1;
    }
    *result_out = NULL;
    try {
        const common_chat_msg msg =
            common_chat_parse(text, is_partial != 0, session->parser_params);
        json out;
        out["content"]     = msg.content;
        out["tool_calls"]  = tool_calls_to_json(msg);
        const std::string dumped = out.dump();
        *result_out              = dup_cstr(dumped);
        if (*result_out == NULL) {
            write_err(err_out, err_cap, "out of memory");
            return -1;
        }
        return 0;
    } catch (const std::exception & ex) {
        write_err(err_out, err_cap, ex.what());
        return -1;
    } catch (...) {
        write_err(err_out, err_cap, "chat parse failed");
        return -1;
    }
}

int syllabix_chat_generate(
    struct syllabix_chat_session *session, const char *messages_json, const char *tools_json,
    int enable_thinking, int thinking, int n_threads, bool (*abort_cb)(void *user),
    void *abort_user, int (*token_cb)(const char *piece, int is_last, void *user), void *token_user,
    char *err_out, size_t err_cap) {
    if (session == NULL || session->llm == NULL) {
        write_err(err_out, err_cap, "chat generate requires a loaded model");
        return -1;
    }
    char * prompt = NULL;
    if (syllabix_chat_apply(session, messages_json, tools_json, enable_thinking, &prompt, err_out,
                            err_cap)
        != 0) {
        return -1;
    }
    const int32_t prompt_len = static_cast<int32_t>(std::strlen(prompt));
    const int     rc         = syllabix_llama_run_prompt(
        session->llm, prompt, prompt_len, thinking, n_threads, abort_cb, abort_user, token_cb,
        token_user, session->render_special.data(), static_cast<int>(session->render_special.size()));
    syllabix_chat_string_free(prompt);
    return rc;
}
