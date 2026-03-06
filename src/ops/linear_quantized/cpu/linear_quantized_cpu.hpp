#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

/**
 * @brief W8A32 量化 Linear — CPU 实现
 *
 * 流程: INT8 weight dequant → F32 临时矩阵 → cblas_sgemm → bias add
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

} // namespace llaisys::ops::cpu
