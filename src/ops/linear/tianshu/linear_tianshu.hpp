#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::tianshu {
void linear(std::byte* out, llaisysDataType_t dtype,
            const std::byte* in, const std::byte* weight, const std::byte* bias,
            size_t in_features, size_t out_features, size_t rows);
} // namespace llaisys::ops::tianshu
