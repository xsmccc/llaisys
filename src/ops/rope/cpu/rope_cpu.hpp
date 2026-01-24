#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu{
    void rope(
        std::byte* out_ptr,
        llaisysDataType_t dtype,
        const std::byte* in_ptr,
        const std::byte* pos_ids,
        size_t seq_len,
        size_t n_heads,
        size_t head_dim,
        float theta
    );
}