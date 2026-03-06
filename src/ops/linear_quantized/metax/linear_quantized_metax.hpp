#pragma once
#include "llaisys.h"
#include <cstddef>

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
);
} // namespace llaisys::ops::metax
