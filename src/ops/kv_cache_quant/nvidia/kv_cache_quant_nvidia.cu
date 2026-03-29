// ============================================================
//  KV Cache INT8 Quantization/Dequantization Kernels
// ============================================================
//
//  Per-token per-head symmetric quantization:
//    scale = max(|x|) / 127.0
//    q = round(clamp(x / scale, -127, 127))
//    x_hat = q * scale
//
//  Memory layout:
//    KV data:  [seq_len, num_kv_heads, head_dim]
//    Scales:   [max_seq_len, num_kv_heads]  (one scale per token per head)
//
// ============================================================

#include "kv_cache_quant_nvidia.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
#include <iostream>
#include <stdexcept>

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

namespace {

// ── Type conversion helpers ──
template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

constexpr int WARP_SIZE = 32;

// ── Warp-level max reduction ──
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ============================================================
//  Quantize kernel: FP → INT8 (per-token per-head)
//
//  Grid: (seq_len, num_kv_heads)
//  Block: min(head_dim, 256) threads
//  Each block handles one (token, head) pair
// ============================================================
template <typename T>
__global__ void kv_quantize_kernel(
    int8_t* __restrict__  dst,       // cache: [max_seq_len, num_kv_heads, head_dim]
    float*  __restrict__  scales,    // [max_seq_len, num_kv_heads]
    const T* __restrict__ src,       // [seq_len, num_kv_heads, head_dim]
    size_t start_pos,
    size_t num_kv_heads,
    size_t head_dim,
    size_t max_seq_len
) {
    const size_t seq_idx = blockIdx.x;      // which token in this batch
    const size_t kv_h    = blockIdx.y;      // which KV head
    const size_t global_pos = start_pos + seq_idx;  // absolute position in cache

    // Source row: src[seq_idx, kv_h, :]
    const T* src_row = src + (seq_idx * num_kv_heads + kv_h) * head_dim;
    // Destination row: dst[global_pos, kv_h, :]
    int8_t* dst_row = dst + (global_pos * num_kv_heads + kv_h) * head_dim;

    // Find absmax across head_dim (warp-level reduction)
    float local_max = 0.0f;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(to_float(src_row[d])));
    }
    local_max = warp_reduce_max(local_max);

    // Block-level max (shared memory for multi-warp)
    __shared__ float s_max;
    if (blockDim.x > WARP_SIZE) {
        // Simple 2-step: first warp reduces, then broadcast
        __shared__ float warp_maxes[8];  // up to 256 threads = 8 warps
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) warp_maxes[warp_id] = local_max;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = warp_maxes[0];
            int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
            for (int w = 1; w < num_warps; w++)
                m = fmaxf(m, warp_maxes[w]);
            s_max = m;
        }
        __syncthreads();
        local_max = s_max;
    }

    // Compute scale
    float scale = local_max / 127.0f;
    float inv_scale = (scale > 0.0f) ? (127.0f / local_max) : 0.0f;

    // Write scale
    if (threadIdx.x == 0) {
        scales[global_pos * num_kv_heads + kv_h] = scale;
    }

    // Quantize and store
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = to_float(src_row[d]) * inv_scale;
        val = fmaxf(-127.0f, fminf(127.0f, rintf(val)));
        dst_row[d] = static_cast<int8_t>(val);
    }
}

// ============================================================
//  Dequantize kernel: INT8 → FP (per-token per-head)
//
//  Grid: (valid_len, num_kv_heads)
//  Block: min(head_dim, 256) threads
// ============================================================
template <typename T>
__global__ void kv_dequantize_kernel(
    T* __restrict__           dst,     // [valid_len, num_kv_heads, head_dim]
    const int8_t* __restrict__ src,    // [valid_len, num_kv_heads, head_dim]
    const float* __restrict__  scales, // [valid_len, num_kv_heads]
    size_t num_kv_heads,
    size_t head_dim
) {
    const size_t t    = blockIdx.x;   // token position
    const size_t kv_h = blockIdx.y;   // KV head

    float scale = scales[t * num_kv_heads + kv_h];

    const int8_t* src_row = src + (t * num_kv_heads + kv_h) * head_dim;
    T* dst_row = dst + (t * num_kv_heads + kv_h) * head_dim;

    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(src_row[d]) * scale;
        dst_row[d] = from_float<T>(val);
    }
}

} // anonymous namespace

// ============================================================
//  Public API
// ============================================================
namespace llaisys::ops::nvidia {

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
) {
    dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(num_kv_heads));
    int threads = std::min(static_cast<int>(head_dim), 256);
    // Round up to warp size
    threads = ((threads + 31) / 32) * 32;

    switch (src_dtype) {
    case LLAISYS_DTYPE_F32:
        kv_quantize_kernel<float><<<grid, threads>>>(
            dst, scales, reinterpret_cast<const float*>(src),
            start_pos, num_kv_heads, head_dim, max_seq_len);
        break;
    case LLAISYS_DTYPE_F16:
        kv_quantize_kernel<__half><<<grid, threads>>>(
            dst, scales, reinterpret_cast<const __half*>(src),
            start_pos, num_kv_heads, head_dim, max_seq_len);
        break;
    case LLAISYS_DTYPE_BF16:
        kv_quantize_kernel<__nv_bfloat16><<<grid, threads>>>(
            dst, scales, reinterpret_cast<const __nv_bfloat16*>(src),
            start_pos, num_kv_heads, head_dim, max_seq_len);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for kv_quantize_to_cache");
    }
    checkCuda(cudaGetLastError(), "kv_quantize_kernel launch failed");
}

void kv_dequantize_from_cache(
    std::byte* dst,
    llaisysDataType_t dst_dtype,
    const int8_t* src,
    const float* scales,
    size_t valid_len,
    size_t num_kv_heads,
    size_t head_dim
) {
    dim3 grid(static_cast<unsigned>(valid_len), static_cast<unsigned>(num_kv_heads));
    int threads = std::min(static_cast<int>(head_dim), 256);
    threads = ((threads + 31) / 32) * 32;

    switch (dst_dtype) {
    case LLAISYS_DTYPE_F32:
        kv_dequantize_kernel<float><<<grid, threads>>>(
            reinterpret_cast<float*>(dst), src, scales,
            num_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        kv_dequantize_kernel<__half><<<grid, threads>>>(
            reinterpret_cast<__half*>(dst), src, scales,
            num_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        kv_dequantize_kernel<__nv_bfloat16><<<grid, threads>>>(
            reinterpret_cast<__nv_bfloat16*>(dst), src, scales,
            num_kv_heads, head_dim);
        break;
    default:
        throw std::runtime_error("Unsupported dtype for kv_dequantize_from_cache");
    }
    checkCuda(cudaGetLastError(), "kv_dequantize_kernel launch failed");
}

} // namespace llaisys::ops::nvidia
