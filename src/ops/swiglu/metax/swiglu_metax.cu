/**
 * @file swiglu_metax.cu
 * @brief SwiGLU 算子的 MetaX MACA 实现
 *
 * 基于 NVIDIA CUDA 版本适配，MACA SDK 提供 CUDA 兼容 API。
 *   out = up ⊙ SiLU(gate)
 *   SiLU(x) = x / (1 + exp(-x))
 *
 * 仅包含 Grid-Stride + float4 向量化优化版本。
 */

#include "swiglu_metax.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

inline void checkMaca(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[MACA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============ 类型转换辅助函数 ============
__device__ __forceinline__ __half to_cuda_half(llaisys::fp16_t v) {
    return *reinterpret_cast<const __half*>(&v._v);
}

__device__ __forceinline__ llaisys::fp16_t from_cuda_half(__half h) {
    return *reinterpret_cast<const llaisys::fp16_t*>(&h);
}

__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(llaisys::bf16_t v) {
    return *reinterpret_cast<const __nv_bfloat16*>(&v._v);
}

__device__ __forceinline__ llaisys::bf16_t from_cuda_bfloat16(__nv_bfloat16 b) {
    return *reinterpret_cast<const llaisys::bf16_t*>(&b);
}

// ============ SiLU 激活函数 ============
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================
//  F32 Grid-Stride + float4 向量化 Kernel
// ============================================================
__global__ void swiglu_kernel_f32_vec4(
    float* out,
    const float* gate,
    const float* up,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; (i + 1) * 4 <= numel; i += stride) {
        size_t base = i * 4;

        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        float4 o4;
        o4.x = u4.x * silu_f32(g4.x);
        o4.y = u4.y * silu_f32(g4.y);
        o4.z = u4.z * silu_f32(g4.z);
        o4.w = u4.w * silu_f32(g4.w);

        *reinterpret_cast<float4*>(out + base) = o4;
    }

    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;
        for (size_t i = tail_start; i < numel; ++i) {
            out[i] = up[i] * silu_f32(gate[i]);
        }
    }
}

// ============================================================
//  F16 Grid-Stride + float4 向量化 Kernel
// ============================================================
__global__ void swiglu_kernel_f16_vec4(
    llaisys::fp16_t* __restrict__ out,
    const llaisys::fp16_t* __restrict__ gate,
    const llaisys::fp16_t* __restrict__ up,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t numel_vec = numel / 8;

    for (size_t i = tid; i < numel_vec; i += stride) {
        size_t base = i * 8;

        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        __half2 g_01 = *reinterpret_cast<const __half2*>(&g4.x);
        __half2 g_23 = *reinterpret_cast<const __half2*>(&g4.y);
        __half2 g_45 = *reinterpret_cast<const __half2*>(&g4.z);
        __half2 g_67 = *reinterpret_cast<const __half2*>(&g4.w);

        __half2 u_01 = *reinterpret_cast<const __half2*>(&u4.x);
        __half2 u_23 = *reinterpret_cast<const __half2*>(&u4.y);
        __half2 u_45 = *reinterpret_cast<const __half2*>(&u4.z);
        __half2 u_67 = *reinterpret_cast<const __half2*>(&u4.w);

        __half2 o_01 = __halves2half2(
            __float2half(__half2float(__low2half(u_01)) * silu_f32(__half2float(__low2half(g_01)))),
            __float2half(__half2float(__high2half(u_01)) * silu_f32(__half2float(__high2half(g_01)))));
        __half2 o_23 = __halves2half2(
            __float2half(__half2float(__low2half(u_23)) * silu_f32(__half2float(__low2half(g_23)))),
            __float2half(__half2float(__high2half(u_23)) * silu_f32(__half2float(__high2half(g_23)))));
        __half2 o_45 = __halves2half2(
            __float2half(__half2float(__low2half(u_45)) * silu_f32(__half2float(__low2half(g_45)))),
            __float2half(__half2float(__high2half(u_45)) * silu_f32(__half2float(__high2half(g_45)))));
        __half2 o_67 = __halves2half2(
            __float2half(__half2float(__low2half(u_67)) * silu_f32(__half2float(__low2half(g_67)))),
            __float2half(__half2float(__high2half(u_67)) * silu_f32(__half2float(__high2half(g_67)))));

        float4 o4;
        *reinterpret_cast<__half2*>(&o4.x) = o_01;
        *reinterpret_cast<__half2*>(&o4.y) = o_23;
        *reinterpret_cast<__half2*>(&o4.z) = o_45;
        *reinterpret_cast<__half2*>(&o4.w) = o_67;

        *reinterpret_cast<float4*>(out + base) = o4;
    }

    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __half2float(to_cuda_half(gate[i]));
            float u = __half2float(to_cuda_half(up[i]));
            out[i] = from_cuda_half(__float2half(u * silu_f32(g)));
        }
    }
}

// ============================================================
//  BF16 Grid-Stride + float4 向量化 Kernel
// ============================================================
__global__ void swiglu_kernel_bf16_vec4(
    llaisys::bf16_t* __restrict__ out,
    const llaisys::bf16_t* __restrict__ gate,
    const llaisys::bf16_t* __restrict__ up,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t numel_vec = numel / 8;

    for (size_t i = tid; i < numel_vec; i += stride) {
        size_t base = i * 8;

        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        __nv_bfloat162 g_01 = *reinterpret_cast<const __nv_bfloat162*>(&g4.x);
        __nv_bfloat162 g_23 = *reinterpret_cast<const __nv_bfloat162*>(&g4.y);
        __nv_bfloat162 g_45 = *reinterpret_cast<const __nv_bfloat162*>(&g4.z);
        __nv_bfloat162 g_67 = *reinterpret_cast<const __nv_bfloat162*>(&g4.w);

        __nv_bfloat162 u_01 = *reinterpret_cast<const __nv_bfloat162*>(&u4.x);
        __nv_bfloat162 u_23 = *reinterpret_cast<const __nv_bfloat162*>(&u4.y);
        __nv_bfloat162 u_45 = *reinterpret_cast<const __nv_bfloat162*>(&u4.z);
        __nv_bfloat162 u_67 = *reinterpret_cast<const __nv_bfloat162*>(&u4.w);

        __nv_bfloat162 o_01 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_01)) * silu_f32(__bfloat162float(__low2bfloat16(g_01)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_01)) * silu_f32(__bfloat162float(__high2bfloat16(g_01)))));
        __nv_bfloat162 o_23 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_23)) * silu_f32(__bfloat162float(__low2bfloat16(g_23)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_23)) * silu_f32(__bfloat162float(__high2bfloat16(g_23)))));
        __nv_bfloat162 o_45 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_45)) * silu_f32(__bfloat162float(__low2bfloat16(g_45)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_45)) * silu_f32(__bfloat162float(__high2bfloat16(g_45)))));
        __nv_bfloat162 o_67 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_67)) * silu_f32(__bfloat162float(__low2bfloat16(g_67)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_67)) * silu_f32(__bfloat162float(__high2bfloat16(g_67)))));

        float4 o4;
        *reinterpret_cast<__nv_bfloat162*>(&o4.x) = o_01;
        *reinterpret_cast<__nv_bfloat162*>(&o4.y) = o_23;
        *reinterpret_cast<__nv_bfloat162*>(&o4.z) = o_45;
        *reinterpret_cast<__nv_bfloat162*>(&o4.w) = o_67;

        *reinterpret_cast<float4*>(out + base) = o4;
    }

    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __bfloat162float(to_cuda_bfloat16(gate[i]));
            float u = __bfloat162float(to_cuda_bfloat16(up[i]));
            out[i] = from_cuda_bfloat16(__float2bfloat16(u * silu_f32(g)));
        }
    }
}

// ============================================================
//  Kernel 启动函数
// ============================================================
void launch_swiglu_kernel(
    std::byte* out,
    const std::byte* gate,
    const std::byte* up,
    llaisysDataType_t dtype,
    size_t numel
) {
    constexpr int THREADS = 256;

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int BLOCKS = num_sm * 8;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel_f32_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<const float*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel_f16_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(gate),
            reinterpret_cast<const llaisys::fp16_t*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel_bf16_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(gate),
            reinterpret_cast<const llaisys::bf16_t*>(up),
            numel
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for MACA swiglu");
    }

    checkMaca(cudaGetLastError(), "Failed to launch swiglu kernel");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::metax {

void swiglu(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* gate_ptr,
    const std::byte* up_ptr,
    size_t numel
) {
    launch_swiglu_kernel(out_ptr, gate_ptr, up_ptr, dtype, numel);
}

} // namespace llaisys::ops::metax
