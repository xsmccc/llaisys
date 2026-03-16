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


/**
 * @brief W4A32 量化 Linear — NVIDIA CUDA 实现 (INT4 group quantization)
 *
 * 流程: INT4 packed weight dequant → FP16 cache → cublasGemmEx TC → bias add
 *
 * @param out             输出 [M, N], F32
 * @param in              输入 [M, K_orig], F32
 * @param weight_packed   INT4 packed 权重 [N, K_orig/2], U8
 * @param scales          group-wise 缩放因子 [N, num_groups], F16
 * @param bias            偏置 [N], F32 (可选)
 * @param in_features     K_orig (原始 input features)
 * @param out_features    N
 * @param rows            M
 * @param num_groups      K_orig / group_size
 * @param group_size      量化组大小 (通常 128)
 */
void linear_quantized_int4(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_packed,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows,
    size_t num_groups,
    size_t group_size
);

/// 释放所有 FP16 权重缓存 (INT8 + INT4)，在模型销毁时调用
void cleanup_quantized_weight_cache();

} // namespace llaisys::ops::nvidia
