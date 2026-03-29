#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

// Quantize FP32/FP16/BF16 KV rows → INT8 cache (per-token per-head symmetric)
// src:    [seq_len, num_kv_heads, head_dim] in compute_dtype
// dst:    [seq_len, num_kv_heads, head_dim] in INT8
// scales: [max_seq_len, num_kv_heads] in F32, only rows [start_pos, start_pos+seq_len) written
void kv_quantize_to_cache(
    int8_t* dst,
    float* scales,
    const std::byte* src,
    llaisysDataType_t src_dtype,
    size_t start_pos,
    size_t seq_len,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_seq_len
);

// Dequantize INT8 cache → compute_dtype for attention computation
// src:    [valid_len, num_kv_heads, head_dim] in INT8  (already sliced)
// scales: [valid_len, num_kv_heads] in F32
// dst:    [valid_len, num_kv_heads, head_dim] in compute_dtype
void kv_dequantize_from_cache(
    std::byte* dst,
    llaisysDataType_t dst_dtype,
    const int8_t* src,
    const float* scales,
    size_t valid_len,
    size_t num_kv_heads,
    size_t head_dim
);

} // namespace llaisys::ops::nvidia
