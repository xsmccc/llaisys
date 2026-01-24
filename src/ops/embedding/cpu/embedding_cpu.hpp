#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu{

void embedding(
        std::byte* out_ptr,
        llaisysDataType_t out_type,
        const std::byte* index_ptr,
        size_t num_indices,
        llaisysDataType_t index_dtype,
        const std::byte* weight_ptr,
        size_t vocab_size,
        size_t embedding_dim
    );
}