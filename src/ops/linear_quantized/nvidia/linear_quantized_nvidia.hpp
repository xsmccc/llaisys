#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

/**
 * @brief W8A32 量化 Linear — NVIDIA CUDA 实现
 *
 * 流程: INT8 weight dequant → F32 临时矩阵 → cublasSgemm → bias add
 *
 * @param out           输出 [M, N], F32
 * @param in            输入 [M, K], F32
 * @param weight_int8   INT8 权重 [N, K]
 * @param scales        per-channel 缩放因子 [N], F32
 * @param bias          偏置 [N], F32 (可选)
 * @param in_features   K
 * @param out_features  N
 * @param rows          M
 */
void linear_quantized(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_int8,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
);

} // namespace llaisys::ops::nvidia
