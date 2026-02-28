#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void rms_norm(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* in_ptr,
    const std::byte* weight_ptr,
    size_t cols,
    size_t rows,
    float eps
);

} // namespace llaisys::ops::nvidia
