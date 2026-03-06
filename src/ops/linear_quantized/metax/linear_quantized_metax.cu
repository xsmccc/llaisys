/**
 * @file linear_quantized_metax.cu
 * @brief W8A32 量化 Linear — MetaX MACA 实现 (待硬件验证)
 *
 * 当前为占位实现，需在 MetaX C500 硬件上验证后完善。
 * 架构与 NVIDIA 版本相同: dequant INT8 → F32 → macablas SGEMM
 */
#include "linear_quantized_metax.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::metax {

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

} // namespace llaisys::ops::metax
