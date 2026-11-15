#pragma once
// Voice-activity head that moondream/parakeet-ultra and -redux carry on the
// encoder's subsampler output: proj (1x1 conv) -> ctx (conv, k=5) -> out (1x1
// conv) -> sigmoid, one probability per 80 ms frame.
#include <vector>

namespace pk {

class ModelLoader;

struct VadWeights {
    int d_in = 0, hidden = 0, kernel = 0;
    std::vector<float> proj_w, proj_b;   // [hidden][d_in], [hidden]
    std::vector<float> ctx_w, ctx_b;     // [hidden][hidden][kernel], [hidden]
    std::vector<float> out_w;            // [hidden]
    float out_b = 0.0f;
};

enum class VadAct { kSiLU, kReLU };

// Activation after proj, activation after ctx and the optional ctx residual.
// The default is the wiring evidenced by Moondream's public behaviour: SiLU
// after proj, SiLU after ctx, no residual (docs/ternary.md, "VAD head wiring").
// The other choices are debug options (`parakeet-cli vad-probe --variant N`).
struct VadVariant {
    VadAct act_proj = VadAct::kSiLU;
    bool residual = false;
    VadAct act_ctx = VadAct::kSiLU;
    // Index 0 is the default. Bit 0: ReLU after proj. Bit 1: residual. Bit 2:
    // ReLU after ctx.
    static VadVariant from_index(int i) {
        VadVariant v;
        if (i & 1) v.act_proj = VadAct::kReLU;
        v.residual = (i & 2) != 0;
        if (i & 4) v.act_ctx = VadAct::kReLU;
        return v;
    }
};

class VadHead {
public:
    // Throws std::runtime_error if the GGUF has no VAD head or its tensors are
    // missing or mis-shaped.
    explicit VadHead(const ModelLoader& ml);
    // x: row-major [T][d_in]. Returns T probabilities in [0, 1].
    std::vector<float> probabilities(const float* x, int T, const VadVariant* v = nullptr) const;
    float frame_sec() const { return frame_sec_; }
    static std::vector<float> run(const VadWeights& w, const VadVariant& v, const float* x, int T);

private:
    VadWeights w_;
    float frame_sec_ = 0.08f;
};

}  // namespace pk
