/**
 * @file linear_quantized_tianshu.cu
 * @brief W8A32 量化 Linear — Tianshu TOPSRIDER 实现 (待硬件验证)
 *
 * 当前为占位实现，需在天数智芯 BI-150 硬件上验证后完善。
 * 架构与 NVIDIA 版本相同: dequant INT8 → F32 → topsblas SGEMM
 */
#include "linear_quantized_tianshu.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::tianshu {

void linear_quantized(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_int8,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    TO_BE_IMPLEMENTED();
}

} // namespace llaisys::ops::tianshu
