/**
 * @file rms_norm_tianshu.cu
 * @brief RMSNorm 算子的 Tianshu TOPSRIDER 实现
 *
 * 基于 NVIDIA CUDA 版本适配，TOPSRIDER SDK 提供 CUDA 兼容 API。
 *   Y_i = W_i * X_i / sqrt( mean(X^2) + eps )
 */

#include "rms_norm_tianshu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>
#include <cmath>

namespace {

inline void checkTops(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[TOPS ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============ 类型转换 ============
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(llaisys::fp16_t v) {
    return __half2float(*reinterpret_cast<const __half*>(&v._v));
}
__device__ __forceinline__ float to_float(llaisys::bf16_t v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v._v));
}

__device__ __forceinline__ float from_float_impl(float v, float*) { return v; }
__device__ __forceinline__ llaisys::fp16_t from_float_impl(float v, llaisys::fp16_t*) {
    __half h = __float2half(v);
    return *reinterpret_cast<const llaisys::fp16_t*>(&h);
}
__device__ __forceinline__ llaisys::bf16_t from_float_impl(float v, llaisys::bf16_t*) {
    __nv_bfloat16 b = __float2bfloat16(v);
    return *reinterpret_cast<const llaisys::bf16_t*>(&b);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return from_float_impl(v, static_cast<T*>(nullptr));
}

// ============ Warp 级求和归约 ============
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

// ============ Block 级求和归约 ============
__device__ float block_reduce_sum(float val) {
    __shared__ float s_partial[32];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        s_partial[warp_id] = val;
    }
    __syncthreads();

    val = (lane < num_warps) ? s_partial[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================
//  RMSNorm Kernel — float4 向量化版
// ============================================================
template <typename T>
__global__ void rms_norm_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const T* __restrict__ weight,
    size_t rows,
    size_t cols,
    float eps
) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T);
    size_t vec_cols = cols / ELT_PER_VEC;

    const float4* in4 = reinterpret_cast<const float4*>(in + row * cols);
    float4* out4 = reinterpret_cast<float4*>(out + row * cols);
    const float4* w4 = reinterpret_cast<const float4*>(weight);

    // Phase 1: 向量化求平方和
    float sum_sq = 0.0f;
    for (size_t j = threadIdx.x; j < vec_cols; j += blockDim.x) {
        float4 raw = in4[j];
        const T* elts = reinterpret_cast<const T*>(&raw);
        #pragma unroll
        for (size_t k = 0; k < ELT_PER_VEC; k++) {
            float v = to_float(elts[k]);
            sum_sq += v * v;
        }
    }
    const T* row_in = in + row * cols;
    size_t tail_start = vec_cols * ELT_PER_VEC;
    for (size_t j = tail_start + threadIdx.x; j < cols; j += blockDim.x) {
        float v = to_float(row_in[j]);
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    // Phase 2: 向量化归一化 + 缩放
    for (size_t j = threadIdx.x; j < vec_cols; j += blockDim.x) {
        float4 raw_v = in4[j];
        float4 raw_w = w4[j];
        const T* v_elts = reinterpret_cast<const T*>(&raw_v);
        const T* w_elts = reinterpret_cast<const T*>(&raw_w);

        T result[ELT_PER_VEC];
        #pragma unroll
        for (size_t k = 0; k < ELT_PER_VEC; k++) {
            float v = to_float(v_elts[k]);
            float w = to_float(w_elts[k]);
            result[k] = from_float<T>(v * w * inv_rms);
        }

        out4[j] = *reinterpret_cast<const float4*>(result);
    }
    T* row_out = out + row * cols;
    for (size_t j = tail_start + threadIdx.x; j < cols; j += blockDim.x) {
        float v = to_float(row_in[j]);
        float w = to_float(weight[j]);
        row_out[j] = from_float<T>(v * w * inv_rms);
    }
}

// ============ 启动函数 ============
void launch_rms_norm(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* in_ptr,
    const std::byte* weight_ptr,
    size_t cols,
    size_t rows,
    float eps
) {
    constexpr int THREADS = 256;
    int blocks = static_cast<int>(rows);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<const float*>(in_ptr),
            reinterpret_cast<const float*>(weight_ptr),
            rows, cols, eps
        );
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(in_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(weight_ptr),
            rows, cols, eps
        );
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(in_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(weight_ptr),
            rows, cols, eps
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for TOPSRIDER rms_norm");
    }

    checkTops(cudaGetLastError(), "Failed to launch rms_norm kernel");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::tianshu {

void rms_norm(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* in_ptr,
    const std::byte* weight_ptr,
    size_t cols,
    size_t rows,
    float eps
) {
    launch_rms_norm(out_ptr, dtype, in_ptr, weight_ptr, cols, rows, eps);
}

} // namespace llaisys::ops::tianshu
