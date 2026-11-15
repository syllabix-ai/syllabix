#include "vad_head.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "backend.hpp"
#include "ggml.h"
#include "model_loader.hpp"

namespace pk {

namespace {

std::vector<float> load_f32(const ModelLoader& ml, const std::string& name, size_t expect) {
    const ggml_tensor* t = ml.tensor(name);
    if (!t) throw std::runtime_error("VAD head: missing tensor " + name);
    if (t->type != GGML_TYPE_F32 || (size_t)ggml_nelements(t) != expect)
        throw std::runtime_error("VAD head: unexpected type or size for " + name);
    // The tensor may live in device memory: read it through the backend.
    std::vector<float> out;
    weight_to_host_f32(ml, name.c_str(), out);
    return out;
}

inline float activate(VadAct a, float x) {
    return a == VadAct::kReLU ? std::max(0.0f, x) : x / (1.0f + std::exp(-x));
}

}  // namespace

VadHead::VadHead(const ModelLoader& ml) {
    const VadCfg& c = ml.config().vad;
    if (!c.present) throw std::runtime_error("model has no VAD head");
    w_.d_in = (int)c.d_in;
    w_.hidden = (int)c.hidden;
    w_.kernel = (int)c.kernel;
    frame_sec_ = c.frame_sec;
    const size_t D = c.d_in, H = c.hidden, K = c.kernel;
    w_.proj_w = load_f32(ml, "vad_head.proj.weight", H * D);
    w_.proj_b = load_f32(ml, "vad_head.proj.bias", H);
    w_.ctx_w = load_f32(ml, "vad_head.ctx.weight", H * H * K);
    w_.ctx_b = load_f32(ml, "vad_head.ctx.bias", H);
    w_.out_w = load_f32(ml, "vad_head.out.weight", H);
    w_.out_b = load_f32(ml, "vad_head.out.bias", 1)[0];
}

std::vector<float> VadHead::run(const VadWeights& w, const VadVariant& v, const float* x, int T) {
    const int D = w.d_in, H = w.hidden, K = w.kernel, pad = K / 2;
    std::vector<float> h1((size_t)T * H), p((size_t)T);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < H; ++o) {
            float acc = w.proj_b[o];
            const float* wr = &w.proj_w[(size_t)o * D];
            const float* xr = x + (size_t)t * D;
            for (int i = 0; i < D; ++i) acc += wr[i] * xr[i];
            h1[(size_t)t * H + o] = activate(v.act_proj, acc);
        }
    for (int t = 0; t < T; ++t) {
        std::vector<float> h2(H);
        for (int o = 0; o < H; ++o) {
            float acc = w.ctx_b[o];
            for (int kk = 0; kk < K; ++kk) {
                const int tt = t + kk - pad;
                if (tt < 0 || tt >= T) continue;
                const float* hr = &h1[(size_t)tt * H];
                for (int i = 0; i < H; ++i) acc += w.ctx_w[((size_t)o * H + i) * K + kk] * hr[i];
            }
            if (v.residual) acc += h1[(size_t)t * H + o];
            h2[o] = activate(v.act_ctx, acc);
        }
        float z = w.out_b;
        for (int i = 0; i < H; ++i) z += w.out_w[i] * h2[i];
        p[t] = 1.0f / (1.0f + std::exp(-z));
    }
    return p;
}

std::vector<float> VadHead::probabilities(const float* x, int T, const VadVariant* v) const {
    return run(w_, v ? *v : VadVariant(), x, T);
}

}  // namespace pk
