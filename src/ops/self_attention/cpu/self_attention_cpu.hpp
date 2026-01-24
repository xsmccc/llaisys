#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu{
    void self_attention(
        std::byte* attn_val_ptr,
        llaisysDataType_t attn_val_type,
        const std::byte* q,
        const std::byte* k,
        const std::byte* v,
        size_t seq_len,
        size_t total_len,
        size_t nhead,
        size_t kv_head,
        size_t head_dim,
        size_t v_head_dim,
        float scale
    );
}