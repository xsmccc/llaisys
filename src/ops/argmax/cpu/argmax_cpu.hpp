#pragma once
#include "llaisys.h" // 为了用 std::byte 和 DType 枚举

#include <cstddef>

namespace llaisys::ops::cpu {

void argmax(std::byte* max_idx, 
            llaisysDataType_t idx_dtype, 
            std::byte* max_val,
            const std::byte* vals, 
            llaisysDataType_t val_dtype,
            size_t numel);

} // namespace llaisys::ops::cpu