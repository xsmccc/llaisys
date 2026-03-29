#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

/**
 * @brief Fused Add + RMSNorm
 *
 * residual_out = a + b
 * out = weight * residual_out * inv_rms(residual_out)
 *
 * 融合版本：1 kernel, 省 1 次 hidden_size 的 global memory 读
 */
void fused_add_rmsnorm(
    void* out,              // [rows, cols] 归一化输出
    void* residual_out,     // [rows, cols] 残差输出 (a + b)
    const void* a,          // [rows, cols] 输入 a
    const void* b,          // [rows, cols] 输入 b
    const void* weight,     // [cols] RMSNorm 权重
    llaisysDataType_t dtype,
    size_t cols,
    size_t rows,
    float eps
);

} // namespace llaisys::ops::nvidia
