/* Optional Moonshine C API via dlopen. Default builds stay free of libmoonshine;
 * medium STT resolves the shared library at runtime when selected. */

#include "syllabix_native.h"
#include "moonshine-c-api.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t (*fn_load_from_files)(const char *, uint32_t,
                                      const struct moonshine_option_t *,
                                      uint64_t, int32_t);
typedef int32_t (*fn_transcribe)(int32_t, const float *, size_t, int32_t,
                                 uint32_t, struct transcript_t **);
typedef void (*fn_free_transcriber)(int32_t);
typedef const char *(*fn_error_to_string)(int32_t);

struct syllabix_moonshine {
    void *lib;
    int32_t handle;
    fn_transcribe transcribe;
    fn_free_transcriber free_transcriber;
    fn_error_to_string error_to_string;
};

static void *open_moonshine_lib(void) {
    const char *override_path = getenv("SYLLABIX_LIBMOONSHINE");
    if (override_path && override_path[0] != '\0') {
        void *lib = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            return lib;
        }
    }
    const char *candidates[] = {
        "libmoonshine.so",
        "libmoonshine.dylib",
        "/usr/local/lib/libmoonshine.so",
        "/usr/lib/libmoonshine.so",
        NULL,
    };
    for (int i = 0; candidates[i] != NULL; i++) {
        void *lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            return lib;
        }
    }
    return NULL;
}

int syllabix_moonshine_available(void) {
    void *lib = open_moonshine_lib();
    if (!lib) {
        return 0;
    }
    dlclose(lib);
    return 1;
}

struct syllabix_moonshine *syllabix_moonshine_load(const char *model_dir) {
    if (model_dir == NULL || model_dir[0] == '\0') {
        return NULL;
    }
    void *lib = open_moonshine_lib();
    if (!lib) {
        return NULL;
    }
    fn_load_from_files load =
        (fn_load_from_files)dlsym(lib, "moonshine_load_transcriber_from_files");
    fn_transcribe transcribe =
        (fn_transcribe)dlsym(lib, "moonshine_transcribe_without_streaming");
    fn_free_transcriber free_fn =
        (fn_free_transcriber)dlsym(lib, "moonshine_free_transcriber");
    fn_error_to_string err_fn =
        (fn_error_to_string)dlsym(lib, "moonshine_error_to_string");
    if (!load || !transcribe || !free_fn) {
        dlclose(lib);
        return NULL;
    }

    struct moonshine_option_t opts[] = {
        {"ort_providers", "CPU"},
    };
    int32_t handle =
        load(model_dir, MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING, opts, 1,
             MOONSHINE_HEADER_VERSION);
    if (handle < 0) {
        if (err_fn) {
            fprintf(stderr, "syllabix moonshine load failed: %s (%d)\n",
                    err_fn(handle), handle);
        }
        dlclose(lib);
        return NULL;
    }

    struct syllabix_moonshine *ms = calloc(1, sizeof(*ms));
    if (!ms) {
        free_fn(handle);
        dlclose(lib);
        return NULL;
    }
    ms->lib = lib;
    ms->handle = handle;
    ms->transcribe = transcribe;
    ms->free_transcriber = free_fn;
    ms->error_to_string = err_fn;
    return ms;
}

void syllabix_moonshine_free(struct syllabix_moonshine *ms) {
    if (!ms) {
        return;
    }
    if (ms->free_transcriber) {
        ms->free_transcriber(ms->handle);
    }
    if (ms->lib) {
        dlclose(ms->lib);
    }
    free(ms);
}

int syllabix_moonshine_transcribe(struct syllabix_moonshine *ms,
                                  const float *pcm, int n_samples, char *out,
                                  int out_cap) {
    if (!ms || !pcm || n_samples <= 0 || !out || out_cap <= 0) {
        return -1;
    }
    out[0] = '\0';
    struct transcript_t *tr = NULL;
    int32_t err =
        ms->transcribe(ms->handle, pcm, (size_t)n_samples, 16000, 0, &tr);
    if (err != 0 || tr == NULL) {
        if (ms->error_to_string) {
            fprintf(stderr, "syllabix moonshine transcribe failed: %s (%d)\n",
                    ms->error_to_string(err), err);
        }
        return -1;
    }
    size_t used = 0;
    for (uint64_t i = 0; i < tr->line_count; i++) {
        const char *text = tr->lines[i].text;
        if (!text || text[0] == '\0') {
            continue;
        }
        size_t len = strlen(text);
        if (used > 0 && used + 1 < (size_t)out_cap) {
            out[used++] = ' ';
        }
        if (used + len >= (size_t)out_cap) {
            len = (size_t)out_cap - used - 1;
        }
        memcpy(out + used, text, len);
        used += len;
        out[used] = '\0';
        if (used + 1 >= (size_t)out_cap) {
            break;
        }
    }
    return 0;
}
