#pragma once
#include "prediction.hpp"
#include "joint.hpp"
#include "decode_types.hpp"
#include <vector>
#include <cstdint>
namespace pk {
// Batched greedy decode for N utterances. encs[n]: row-major [T[n], enc_hidden].
// durations empty -> RNNT (advance-by-1); non-empty -> TDT (advance-by-duration).
// Outputs per item: ids[n], and (if toks != nullptr) TokenInfo[n]. Applies the
// same per-item decision rule as rnnt_greedy / tdt_greedy.
//
// On the CPU backend the result is bit-identical to per-item decode (logits,
// token ids, frames, spans and confidences): the prediction step and the joint
// use mul_mat_cols_exact (exact_matvec.hpp), which runs the same vec_dot as the
// single column matmul for every column. test_exact_batch guards this for every
// decoder weight type. On other backends the batched matmul is the ordinary
// ggml one: logits then differ from per-item decode by float noise (up to about
// 1e-4 measured), and a near-tie argmax could in principle flip.
void transducer_greedy_batch(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<std::vector<float>>& encs,
    const std::vector<int>& T,             // [N] per-item frame counts
    int enc_hidden,
    const std::vector<int32_t>& durations, // empty=RNNT
    int blank_id, int max_symbols,
    std::vector<std::vector<int32_t>>& ids,            // OUT [N][.]
    std::vector<std::vector<TokenInfo>>* toks);         // OUT [N][.] or nullptr
} // namespace pk
