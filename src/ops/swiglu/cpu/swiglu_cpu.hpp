#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu{
    void swiglu(
        std::byte* out_ptr,
        llaisysDataType_t out_dtype,
        const std::byte* gate_ptr,
        const std::byte* up_ptr,
        size_t numel
    );
} 