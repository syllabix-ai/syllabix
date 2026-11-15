#include "ced_tagger.hpp"

#include "bundle.hpp"
#include "bundle_map.hpp"
#include "gguf.h"

#ifdef PARAKEET_WITH_CED
#include "ced_capi.h"
#endif

namespace pk {

bool gguf_is_ced(const std::string& path) {
    gguf_init_params p{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "general.architecture");
    const bool ced = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                     std::string(gguf_get_val_str(g, id)) == "ced";
    gguf_free(g);
    return ced;
}

#ifdef PARAKEET_WITH_CED

bool CedTagger::available() { return true; }

std::unique_ptr<CedTagger> CedTagger::load(const std::string& path, const std::string& component,
                                           std::string* err) {
    ced_ctx* c = nullptr;
    if (component.empty()) {
        c = ced_capi_load(path.c_str());
    } else {
        // The bundle is mapped read-only and ced.cpp copies the tensors of the
        // component out of the map during the call (docs/bundle.md).
        BundleInfo info;
        std::string e;
        if (!read_bundle_info(path, info, &e)) { if (err) *err = e; return nullptr; }
        const BundleComponent* bc = info.find(component);
        if (!bc) { if (err) *err = "the bundle has no component \"" + component + "\""; return nullptr; }
        if (bc->kind != kBundleKindCed) {
            if (err) *err = "component \"" + component + "\" is of kind \"" + bc->kind + "\", not \"ced\"";
            return nullptr;
        }
        std::unique_ptr<MappedFile> map = MappedFile::open(path, &e);
        if (!map) { if (err) *err = e; return nullptr; }
        c = ced_capi_load_from_memory_prefixed(map->data(), map->size(), (component + ".").c_str());
    }
    if (!c) {
        if (err) {
            const char* m = ced_capi_last_error(nullptr);
            *err = "cannot load the sound model " + (component.empty() ? path : component) +
                   ((m && *m) ? std::string(": ") + m : std::string());
        }
        return nullptr;
    }
    std::unique_ptr<CedTagger> t(new CedTagger());
    t->ctx_ = c;
    return t;
}

CedTagger::~CedTagger() { ced_capi_free(static_cast<ced_ctx*>(ctx_)); }

int CedTagger::n_classes() const { return ced_capi_num_classes(static_cast<ced_ctx*>(ctx_)); }

const char* CedTagger::label(int i) const { return ced_capi_label(static_cast<ced_ctx*>(ctx_), i); }

SoundScorer CedTagger::scorer() {
    return [this](const float* pcm, int n, std::vector<float>& probs) {
        auto* c = static_cast<ced_ctx*>(ctx_);
        last_error_.clear();   // describes the latest call only
        probs.assign((size_t)n_classes(), 0.0f);
        const int w = ced_capi_classify_pcm_probs(c, pcm, n, 16000, probs.data(), (int)probs.size());
        if (w != (int)probs.size()) {
            const char* m = ced_capi_last_error(c);
            last_error_ = m ? m : "";
            return false;
        }
        return true;
    };
}

#else  // PARAKEET_WITH_CED

bool CedTagger::available() { return false; }
std::unique_ptr<CedTagger> CedTagger::load(const std::string&, const std::string&, std::string* err) {
    if (err) *err = "this build has no sound-event support (PARAKEET_WITH_CED=OFF)";
    return nullptr;
}
CedTagger::~CedTagger() = default;
int CedTagger::n_classes() const { return 0; }
const char* CedTagger::label(int) const { return nullptr; }
SoundScorer CedTagger::scorer() {
    return [](const float*, int, std::vector<float>&) { return false; };
}

#endif

} // namespace pk
