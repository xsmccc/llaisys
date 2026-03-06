/**
 * @file rope_tianshu.cu
 * @brief RoPE（旋转位置编码）算子的 Tianshu TOPSRIDER 实现
 *
 * 基于 NVIDIA CUDA 版本适配，TOPSRIDER SDK 提供 CUDA 兼容 API。
 * Elementwise 旋转：a' = a*cos - b*sin, b' = b*cos + a*sin
 */

#include "rope_tianshu.hpp"
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

// ============================================================
//  RoPE Kernel — sincosf 优化版
// ============================================================
template <typename T>
__global__ void rope_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const int64_t* __restrict__ pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * n_heads * half_dim;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += blockDim.x * gridDim.x)
    {
        size_t j = tid % half_dim;
        size_t h = (tid / half_dim) % n_heads;
        size_t i = tid / (half_dim * n_heads);

        size_t offset = i * n_heads * head_dim + h * head_dim;

        float a = to_float(in[offset + j]);
        float b = to_float(in[offset + j + half_dim]);

        float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
        float theta_pow = powf(theta, exponent);
        float angle = static_cast<float>(pos_ids[i]) / theta_pow;

        float sin_val, cos_val;
        sincosf(angle, &sin_val, &cos_val);

        out[offset + j]            = from_float<T>(a * cos_val - b * sin_val);
        out[offset + j + half_dim] = from_float<T>(b * cos_val + a * sin_val);
    }
}

// ============ 启动函数 ============
void launch_rope(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* in_ptr,
    const std::byte* pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
) {
    constexpr int THREADS = 256;
    size_t total = seq_len * n_heads * (head_dim / 2);
    int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    blocks = std::min(blocks, 65535);

    const int64_t* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<const float*>(in_ptr),
            pos_ids_ptr, seq_len, n_heads, head_dim, theta
        );
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(in_ptr),
            pos_ids_ptr, seq_len, n_heads, head_dim, theta
        );
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(in_ptr),
            pos_ids_ptr, seq_len, n_heads, head_dim, theta
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for TOPSRIDER rope");
    }

    checkTops(cudaGetLastError(), "Failed to launch rope kernel");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::tianshu {

void rope(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* in_ptr,
    const std::byte* pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
) {
    launch_rope(out_ptr, dtype, in_ptr, pos_ids, seq_len, n_heads, head_dim, theta);
}

} // namespace llaisys::ops::tianshu
