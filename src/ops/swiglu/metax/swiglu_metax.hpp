#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::metax {
void swiglu(std::byte* out_ptr, llaisysDataType_t dtype,
            const std::byte* gate_ptr, const std::byte* up_ptr, size_t numel);
} // namespace llaisys::ops::metax
