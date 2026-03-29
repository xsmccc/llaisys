#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

/**
 * @brief W8A16 量化 Linear — NVIDIA CUDA 实现
 *
 * 流程: INT8 weight dequant → FP16 persistent cache → cublasGemmEx FP16 TC
 * 支持 FP16/FP32 activation 输入, FP16/FP32 输出
 *
 * @param out           输出 [M, N], F16 或 F32 (由 out_dtype 决定)
 * @param in            输入 [M, K], F16 或 F32 (由 in_dtype 决定)
 * @param weight_int8   INT8 权重 [N, K]
 * @param scales        per-channel 缩放因子 [N], F32
 * @param bias          偏置 [N], F16/F32 (可选, 由 bias_dtype 决定)
 * @param in_features   K
 * @param out_features  N
 * @param rows          M
 * @param in_dtype      输入数据类型 (LLAISYS_DTYPE_F16 或 LLAISYS_DTYPE_F32)
 * @param out_dtype     输出数据类型
 * @param bias_dtype    偏置数据类型
 */
void linear_quantized(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_int8,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
);


/**
 * @brief W4A16 量化 Linear — NVIDIA CUDA 实现 (INT4 group quantization)
 *
 * 流程: INT4 packed weight dequant → FP16 cache → cublasGemmEx TC
 * 支持 FP16/FP32 activation 输入, FP16/FP32 输出
 *
 * @param out             输出 [M, N], F16/F32
 * @param in              输入 [M, K_orig], F16/F32
 * @param weight_packed   INT4 packed 权重 [N, K_orig/2], U8
 * @param scales          group-wise 缩放因子 [N, num_groups], F16
 * @param bias            偏置 [N], F16/F32 (可选)
 * @param in_features     K_orig (原始 input features)
 * @param out_features    N
 * @param rows            M
 * @param num_groups      K_orig / group_size
 * @param group_size      量化组大小 (通常 128)
 * @param in_dtype        输入数据类型
 * @param out_dtype       输出数据类型
 * @param bias_dtype      偏置数据类型
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
    size_t group_size,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
);

/// 释放所有 FP16 权重缓存 (INT8 + INT4)，在模型销毁时调用
void cleanup_quantized_weight_cache();

} // namespace llaisys::ops::nvidia
