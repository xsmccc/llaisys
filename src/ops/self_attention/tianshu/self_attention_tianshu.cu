/**
 * @file self_attention_tianshu.cu
 * @brief Causal Self-Attention 算子的 Tianshu TOPSRIDER 实现（融合 kernel）
 *
 * 基于 NVIDIA CUDA 版本适配，TOPSRIDER SDK 提供 CUDA 兼容 API。
 * 融合 4 个阶段：QK^T → causal mask → softmax → scores@V
 */

#include "self_attention_tianshu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS   = THREADS / 32;

inline void checkTops(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[TOPS ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================
//  类型转换辅助
// ============================================================
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ============================================================
//  Block Reduce — warp shuffle + shared memory
// ============================================================
__device__ __forceinline__
float block_reduce_max(float val, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int delta = 16; delta >= 1; delta >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));

    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();

    val = (threadIdx.x < WARPS) ? warp_buf[threadIdx.x] : -INFINITY;
    if (wid == 0) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));
    }
    return val;
}

__device__ __forceinline__
float block_reduce_sum(float val, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();

    val = (threadIdx.x < WARPS) ? warp_buf[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

// ============================================================
//  融合 Self-Attention Kernel
// ============================================================
template <typename T>
__global__ void self_attention_fused(
    T* __restrict__       attn_val,
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
) {
    size_t i    = blockIdx.x;
    size_t h    = blockIdx.y;
    size_t kv_h = h / (nhead / kv_head);

    extern __shared__ char smem_bytes[];
    float* scores   = reinterpret_cast<float*>(smem_bytes);
    float* warp_buf = scores + total_len;
    float* s_q      = warp_buf + WARPS;

    __shared__ float s_max;
    __shared__ float s_sum;

    // 预加载 Q 到共享内存
    const T* q_row = q + i * nhead * head_dim + h * head_dim;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_q[d] = to_float(q_row[d]);
    }
    __syncthreads();

    // Phase 1: Q @ K^T (float4 向量化)
    constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T);
    const size_t head_dim_vec  = head_dim / ELT_PER_VEC;
    const size_t head_dim_tail = head_dim_vec * ELT_PER_VEC;
    const size_t kv_stride = kv_head * head_dim;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        const T* k_row = k + t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;

        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        for (size_t vi = 0; vi < head_dim_vec; vi++) {
            float4 kv = k4[vi];
            const T* ke = reinterpret_cast<const T*>(&kv);
            #pragma unroll
            for (size_t e = 0; e < ELT_PER_VEC; e++) {
                dot += s_q[vi * ELT_PER_VEC + e] * to_float(ke[e]);
            }
        }

        for (size_t d = head_dim_tail; d < head_dim; d++) {
            dot += s_q[d] * to_float(k_row[d]);
        }

        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Causal Mask
    size_t current_pos = total_len - seq_len + i;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        if (t > current_pos) {
            scores[t] = -INFINITY;
        }
    }
    __syncthreads();

    // Phase 3: Safe Softmax
    float local_max = -INFINITY;
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    local_max = block_reduce_max(local_max, warp_buf);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_val = s_max;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_val);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        local_sum += scores[t];
    }
    local_sum = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Phase 4: scores @ V
    T* out_row = attn_val + i * nhead * v_head_dim + h * v_head_dim;
    const size_t kv_v_stride = kv_head * v_head_dim;

    for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x) {
        float val = 0.0f;
        for (size_t t = 0; t < total_len; t++) {
            const T* v_row = v + t * kv_v_stride + kv_h * v_head_dim;
            val += scores[t] * to_float(v_row[dv]);
        }
        out_row[dv] = from_float<T>(val);
    }
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::tianshu {

void self_attention(
    std::byte* attn_val_ptr,
    llaisysDataType_t dtype,
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
) {
    dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(nhead));
    constexpr int threads = THREADS;
    size_t smem_size = (total_len + WARPS + head_dim) * sizeof(float);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_attention_fused<float><<<grid, threads, smem_size>>>(
            reinterpret_cast<float*>(attn_val_ptr),
            reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v),
            seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
        );
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_fused<__half><<<grid, threads, smem_size>>>(
            reinterpret_cast<__half*>(attn_val_ptr),
            reinterpret_cast<const __half*>(q),
            reinterpret_cast<const __half*>(k),
            reinterpret_cast<const __half*>(v),
            seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
        );
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_fused<__nv_bfloat16><<<grid, threads, smem_size>>>(
            reinterpret_cast<__nv_bfloat16*>(attn_val_ptr),
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const __nv_bfloat16*>(k),
            reinterpret_cast<const __nv_bfloat16*>(v),
            seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
        );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    checkTops(cudaGetLastError(), "self_attention_fused kernel launch failed");
}

} // namespace llaisys::ops::tianshu
