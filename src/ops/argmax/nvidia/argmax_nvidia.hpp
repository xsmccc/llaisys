#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void argmax(
    std::byte* max_idx,
    llaisysDataType_t idx_dtype,
    std::byte* max_val,
    const std::byte* vals,
    llaisysDataType_t val_dtype,
    size_t numel
);

} // namespace llaisys::ops::nvidia
