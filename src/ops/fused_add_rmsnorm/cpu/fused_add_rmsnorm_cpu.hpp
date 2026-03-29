#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace llaisys::ops::cpu {

inline void fused_add_rmsnorm(
    void* out, void* residual_out,
    const void* a, const void* b,
    const void* weight,
    llaisysDataType_t dtype,
    size_t cols, size_t rows,
    float eps
) {
    if (dtype != LLAISYS_DTYPE_F32)
        throw std::runtime_error("fused_add_rmsnorm CPU: only F32 supported");

    const float* fa = static_cast<const float*>(a);
    const float* fb = static_cast<const float*>(b);
    const float* fw = static_cast<const float*>(weight);
    float* fres = static_cast<float*>(residual_out);
    float* fout = static_cast<float*>(out);

    for (size_t r = 0; r < rows; r++) {
        size_t off = r * cols;
        // Add + accumulate sum_sq
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float s = fa[off + j] + fb[off + j];
            fres[off + j] = s;
            sum_sq += s * s;
        }
        // RMSNorm
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        for (size_t j = 0; j < cols; j++) {
            fout[off + j] = fw[j] * fres[off + j] * inv_rms;
        }
    }
}

} // namespace llaisys::ops::cpu
