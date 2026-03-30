/**
 * SwiGLU CUDA kernel: out = up * SiLU(gate), SiLU(x) = x * sigmoid(x)
 *
 * F32: float4 vectorized, grid-stride loop (keeps enough warps for latency hiding)
 * F16/BF16: half2/bfloat162 vectorized
 *
 * Arithmetic intensity ~0.33 FLOP/B → memory-bound.
 * Measured: DRAM throughput 95%+ (ncu, RTX 4060 Ti).
 */

#include "swiglu_nvidia.hpp"
#include "../../../utils.hpp"
#include "../../../core/context/context.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

// ============ 错误检查 ============
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// 类型转换: llaisys fp16/bf16 ↔ CUDA __half/__nv_bfloat16 (binary-compatible)

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

// SiLU(x) = x * sigmoid(x)
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

// ── Naive kernels: 1 thread per element, baseline ──
__global__ void swiglu_kernel_f32_naive(
    float* out,
    const float* gate,
    const float* up,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        float g = gate[idx];
        float u = up[idx];
        out[idx] = u * silu_f32(g);
    }
}

// F16 naive: fp16 → float → SiLU → float → fp16
__global__ void swiglu_kernel_f16_naive(
    llaisys::fp16_t* out,
    const llaisys::fp16_t* gate,
    const llaisys::fp16_t* up,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        __half g_half = to_cuda_half(gate[idx]);
        __half u_half = to_cuda_half(up[idx]);
        
        float g = __half2float(g_half);
        float u = __half2float(u_half);
        
        float silu_val = silu_f32(g);
        float result = u * silu_val;
        
        out[idx] = from_cuda_half(__float2half(result));
    }
}

// BF16 naive: bf16 → float → SiLU → float → bf16
__global__ void swiglu_kernel_bf16_naive(
    llaisys::bf16_t* out,
    const llaisys::bf16_t* gate,
    const llaisys::bf16_t* up,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        __nv_bfloat16 g_bf16 = to_cuda_bfloat16(gate[idx]);
        __nv_bfloat16 u_bf16 = to_cuda_bfloat16(up[idx]);
        
        float g = __bfloat162float(g_bf16);
        float u = __bfloat162float(u_bf16);
        
        float silu_val = silu_f32(g);
        float result = u * silu_val;
        
        out[idx] = from_cuda_bfloat16(__float2bfloat16(result));
    }
}

// ── 向量化版本：Grid-Stride + vec4/vec2 ──
// Grid = num_sm * 8，固定不变；每线程循环处理多个 vec4，避免 Grid 过小导致 SM 空闲

// F32 vec4 kernel: 每线程处理 4 个 float (LD.128/ST.128)
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

    // 尾部处理
    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;
        for (size_t i = tail_start; i < numel; ++i) {
            out[i] = up[i] * silu_f32(gate[i]);
        }
    }
}

// F16 vec4 kernel: float4 搬运 8 个 half, reinterpret 为 half2 计算
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
            __float2half(__half2float(__low2half(u_01)) * silu_f32(__half2float(__low2half(g_01)))),//low
            __float2half(__half2float(__high2half(u_01)) * silu_f32(__half2float(__high2half(g_01)))));//high
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

    // 尾部处理
    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __half2float(to_cuda_half(gate[i]));
            float u = __half2float(to_cuda_half(up[i]));
            out[i] = from_cuda_half(__float2half(u * silu_f32(g)));
        }
    }
}

// BF16 vec4 kernel: 同 F16 结构, bfloat162 类型
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

    // 尾部处理
    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __bfloat162float(to_cuda_bfloat16(gate[i]));
            float u = __bfloat162float(to_cuda_bfloat16(up[i]));
            out[i] = from_cuda_bfloat16(__float2bfloat16(u * silu_f32(g)));
        }
    }
}

// 根据 dtype 分派 kernel，Grid = num_sm * 8 保证 SM 满载
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
        swiglu_kernel_f32_vec4<<<BLOCKS, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<const float*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel_f16_vec4<<<BLOCKS, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(gate),
            reinterpret_cast<const llaisys::fp16_t*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel_bf16_vec4<<<BLOCKS, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(gate),
            reinterpret_cast<const llaisys::bf16_t*>(up),
            numel
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for CUDA swiglu");
    }

    checkCuda(cudaGetLastError(), "Failed to launch swiglu kernel");
}

} // anonymous namespace

namespace llaisys::ops::nvidia {

void swiglu(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* gate_ptr,
    const std::byte* up_ptr,
    size_t numel
) {
    launch_swiglu_kernel(out_ptr, gate_ptr, up_ptr, dtype, numel);
}

} // namespace llaisys::ops::nvidia
