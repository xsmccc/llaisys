/**
 * @file add_tianshu.cu
 * @brief Add 算子 Tianshu TOPSRIDER 实现 — 向量化版本
 *
 * 天数智芯 TOPSRIDER SDK 提供 CUDA 兼容 API，topscc 编译器编译标准 .cu 语法。
 * 算子逻辑与 NVIDIA 版本一致：float4 向量化 + Grid-Stride Loop。
 *
 * 【Tianshu BI-150 适配注意事项】
 *   - Warp Size: BI-150 可能使用 128 线程/warp，需验证 __shfl_down_sync 行为
 *   - 本算子不涉及 warp shuffle（纯 elementwise），无需 warp 适配
 *   - float4 向量化在 TOPSRIDER 上同样有效（LD.128 指令）
 */

#include "add_tianshu.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

inline void checkTops(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[TOPS ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============ 类型转换辅助函数 ============
__device__ __forceinline__ __half to_cuda_half(llaisys::fp16_t v) {
    return *reinterpret_cast<const __half *>(&v._v);
}
__device__ __forceinline__ llaisys::fp16_t from_cuda_half(__half h) {
    return *reinterpret_cast<const llaisys::fp16_t *>(&h);
}
__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(llaisys::bf16_t v) {
    return *reinterpret_cast<const __nv_bfloat16 *>(&v._v);
}
__device__ __forceinline__ llaisys::bf16_t from_cuda_bfloat16(__nv_bfloat16 b) {
    return *reinterpret_cast<const llaisys::bf16_t *>(&b);
}

// ============ F32 向量化 Kernel ============
__global__ void add_kernel_f32_vec(
    float * __restrict__ c,
    const float * __restrict__ a,
    const float * __restrict__ b,
    size_t numel
) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; (i + 1) * 4 <= numel; i += stride) {
        size_t base = i * 4;
        float4 a4 = *reinterpret_cast<const float4 *>(a + base);
        float4 b4 = *reinterpret_cast<const float4 *>(b + base);
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        *reinterpret_cast<float4 *>(c + base) = c4;
    }

    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;
        for (size_t i = tail_start; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

// ============ F16 向量化 Kernel ============
__global__ void add_kernel_f16_vec(
    llaisys::fp16_t * __restrict__ c,
    const llaisys::fp16_t * __restrict__ a,
    const llaisys::fp16_t * __restrict__ b,
    size_t numel
) {
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; (i + 1) * 8 <= numel; i += stride) {
        size_t base = i * 8;
        float4 a_chunk = *reinterpret_cast<const float4 *>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4 *>(b + base);
        __half2 *a_h2 = reinterpret_cast<__half2 *>(&a_chunk);
        __half2 *b_h2 = reinterpret_cast<__half2 *>(&b_chunk);
        __half2 c_h2[4];
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]);
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]);
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]);
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]);
        *reinterpret_cast<float4 *>(c + base) = *reinterpret_cast<float4 *>(c_h2);
    }

    if (tid == 0) {
        size_t tail_start = (numel / 8) * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            __half ha = to_cuda_half(a[i]);
            __half hb = to_cuda_half(b[i]);
            c[i] = from_cuda_half(__hadd(ha, hb));
        }
    }
}

// ============ BF16 向量化 Kernel ============
__global__ void add_kernel_bf16_vec(
    llaisys::bf16_t * __restrict__ c,
    const llaisys::bf16_t * __restrict__ a,
    const llaisys::bf16_t * __restrict__ b,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; (i + 1) * 8 <= numel; i += stride) {
        size_t base = i * 8;
        float4 a_chunk = *reinterpret_cast<const float4 *>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4 *>(b + base);
        __nv_bfloat162 *a_h2 = reinterpret_cast<__nv_bfloat162 *>(&a_chunk);
        __nv_bfloat162 *b_h2 = reinterpret_cast<__nv_bfloat162 *>(&b_chunk);
        __nv_bfloat162 c_h2[4];
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]);
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]);
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]);
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]);
        *reinterpret_cast<float4 *>(c + base) = *reinterpret_cast<float4 *>(c_h2);
    }

    if (tid == 0) {
        size_t tail_start = (numel / 8) * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            __nv_bfloat16 ba = to_cuda_bfloat16(a[i]);
            __nv_bfloat16 bb = to_cuda_bfloat16(b[i]);
            c[i] = from_cuda_bfloat16(__hadd(ba, bb));
        }
    }
}

// ============ Kernel 启动器 ============
int get_num_sm() {
    static int num_sm = 0;
    if (num_sm == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    }
    return num_sm;
}

void launch_add_kernel(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    constexpr int threads = 256;
    const int blocks = get_num_sm() * 8;
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel_f32_vec<<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel_f16_vec<<<blocks, threads>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel_bf16_vec<<<blocks, threads>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b), numel);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for TOPSRIDER add");
    }
    checkTops(cudaGetLastError(), "Failed to launch add kernel");
}

} // namespace

namespace llaisys::ops::tianshu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    launch_add_kernel(c, a, b, type, numel);
}
} // namespace llaisys::ops::tianshu
