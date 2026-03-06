/**
 * @file linear_tianshu.cu
 * @brief Linear (全连接层) 算子的 Tianshu TOPSRIDER 实现 — 使用 cuBLAS
 *
 * 基于 NVIDIA CUDA 版本适配，TOPSRIDER SDK 提供 CUDA/cuBLAS 兼容 API。
 *   out = in @ weight.T + bias
 */

#include "linear_tianshu.hpp"
#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>
#include <mutex>

namespace {

// ============================================================
//  全局 cuBLAS Handle（懒初始化，线程安全）
// ============================================================
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("[TOPS BLAS] Failed to create handle");
        }
    });
    return handle;
}

inline void checkTops(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[TOPS ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

inline void checkCublas(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[TOPS BLAS ERROR] " << msg << ": status=" << static_cast<int>(st) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================
//  Bias Add Kernel — 广播加偏置
// ============================================================

// ---- 标量 fallback ----
__global__ void add_bias_f32_scalar(
    float* __restrict__ out,
    const float* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    size_t total = rows * out_features;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
        out[idx] += bias[idx % out_features];
    }
}

__global__ void add_bias_f16_scalar(
    __half* __restrict__ out,
    const __half* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    size_t total = rows * out_features;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
        out[idx] = __hadd(out[idx], bias[idx % out_features]);
    }
}

__global__ void add_bias_bf16_scalar(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    size_t total = rows * out_features;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
#if __CUDA_ARCH__ >= 800
        out[idx] = __hadd(out[idx], bias[idx % out_features]);
#else
        out[idx] = __float2bfloat16(__bfloat162float(out[idx]) + __bfloat162float(bias[idx % out_features]));
#endif
    }
}

// ---- F32 向量化：float4 = 4 floats ----
__global__ void add_bias_f32(
    float* __restrict__ out,
    const float* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    constexpr size_t VEC = 4;
    const size_t vec_cols  = out_features / VEC;
    const size_t total_vec = rows * vec_cols;

    float4*       out4  = reinterpret_cast<float4*>(out);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec;
         i += gridDim.x * blockDim.x) {
        size_t vc = i % vec_cols;
        float4 o = out4[i];
        float4 b = bias4[vc];
        o.x += b.x;  o.y += b.y;  o.z += b.z;  o.w += b.w;
        out4[i] = o;
    }
}

// ---- F16 向量化：float4 = 8 halfs ----
__global__ void add_bias_f16(
    __half* __restrict__ out,
    const __half* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    constexpr size_t VEC = 8;
    const size_t vec_cols  = out_features / VEC;
    const size_t total_vec = rows * vec_cols;

    float4*       out4  = reinterpret_cast<float4*>(out);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec;
         i += gridDim.x * blockDim.x) {
        size_t vc = i % vec_cols;
        float4 o = out4[i];
        float4 b = bias4[vc];
        __half2* oh = reinterpret_cast<__half2*>(&o);
        const __half2* bh = reinterpret_cast<const __half2*>(&b);
        #pragma unroll
        for (int k = 0; k < 4; ++k) oh[k] = __hadd2(oh[k], bh[k]);
        out4[i] = o;
    }
}

// ---- BF16 向量化：float4 = 8 bf16s ----
__global__ void add_bias_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    constexpr size_t VEC = 8;
    const size_t vec_cols  = out_features / VEC;
    const size_t total_vec = rows * vec_cols;

    float4*       out4  = reinterpret_cast<float4*>(out);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec;
         i += gridDim.x * blockDim.x) {
        size_t vc = i % vec_cols;
        float4 o = out4[i];
        float4 b = bias4[vc];
#if __CUDA_ARCH__ >= 800
        __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
        const __nv_bfloat162* bh = reinterpret_cast<const __nv_bfloat162*>(&b);
        #pragma unroll
        for (int k = 0; k < 4; ++k) oh[k] = __hadd2(oh[k], bh[k]);
#else
        __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
        const __nv_bfloat16* be = reinterpret_cast<const __nv_bfloat16*>(&b);
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            oe[k] = __float2bfloat16(__bfloat162float(oe[k]) + __bfloat162float(be[k]));
#endif
        out4[i] = o;
    }
}

// ---- Bias Add dispatch helpers ----
inline void launch_bias_f32(float* out, const float* bias, size_t rows, size_t N) {
    constexpr int THREADS = 256;
    if (N % 4 == 0) {
        size_t total_vec = rows * (N / 4);
        int blocks = static_cast<int>((total_vec + THREADS - 1) / THREADS);
        add_bias_f32<<<blocks, THREADS>>>(out, bias, rows, N);
    } else {
        size_t total = rows * N;
        int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        add_bias_f32_scalar<<<blocks, THREADS>>>(out, bias, rows, N);
    }
    checkTops(cudaGetLastError(), "add_bias_f32 launch failed");
}

inline void launch_bias_f16(__half* out, const __half* bias, size_t rows, size_t N) {
    constexpr int THREADS = 256;
    if (N % 8 == 0) {
        size_t total_vec = rows * (N / 8);
        int blocks = static_cast<int>((total_vec + THREADS - 1) / THREADS);
        add_bias_f16<<<blocks, THREADS>>>(out, bias, rows, N);
    } else {
        size_t total = rows * N;
        int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        add_bias_f16_scalar<<<blocks, THREADS>>>(out, bias, rows, N);
    }
    checkTops(cudaGetLastError(), "add_bias_f16 launch failed");
}

inline void launch_bias_bf16(__nv_bfloat16* out, const __nv_bfloat16* bias, size_t rows, size_t N) {
    constexpr int THREADS = 256;
    if (N % 8 == 0) {
        size_t total_vec = rows * (N / 8);
        int blocks = static_cast<int>((total_vec + THREADS - 1) / THREADS);
        add_bias_bf16<<<blocks, THREADS>>>(out, bias, rows, N);
    } else {
        size_t total = rows * N;
        int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        add_bias_bf16_scalar<<<blocks, THREADS>>>(out, bias, rows, N);
    }
    checkTops(cudaGetLastError(), "add_bias_bf16 launch failed");
}

// ============================================================
//  F32 Linear — cublasSgemm
// ============================================================
void linear_f32(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight,
            static_cast<int>(in_features),
            in,
            static_cast<int>(in_features),
            &beta,
            out,
            static_cast<int>(out_features)
        ),
        "cublasSgemm failed"
    );

    if (bias != nullptr) {
        launch_bias_f32(out, bias, rows, out_features);
    }
}

// ============================================================
//  F16 Linear — cublasGemmEx (FP16 compute)
// ============================================================
void linear_f16(
    __half* out,
    const __half* in,
    const __half* weight,
    const __half* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight, CUDA_R_16F,
            static_cast<int>(in_features),
            in, CUDA_R_16F,
            static_cast<int>(in_features),
            &beta,
            out, CUDA_R_16F,
            static_cast<int>(out_features),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx F16 failed"
    );

    if (bias != nullptr) {
        launch_bias_f16(out, bias, rows, out_features);
    }
}

// ============================================================
//  BF16 Linear — cublasGemmEx (F32 compute, BF16 I/O)
// ============================================================
void linear_bf16(
    __nv_bfloat16* out,
    const __nv_bfloat16* in,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight, CUDA_R_16BF,
            static_cast<int>(in_features),
            in, CUDA_R_16BF,
            static_cast<int>(in_features),
            &beta,
            out, CUDA_R_16BF,
            static_cast<int>(out_features),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx BF16 failed"
    );

    if (bias != nullptr) {
        launch_bias_bf16(out, bias, rows, out_features);
    }
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::tianshu {

void linear(
    std::byte* out,
    llaisysDataType_t dtype,
    const std::byte* in,
    const std::byte* weight,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_f32(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            bias ? reinterpret_cast<const float*>(bias) : nullptr,
            in_features, out_features, rows
        );
    case LLAISYS_DTYPE_F16:
        return linear_f16(
            reinterpret_cast<__half*>(out),
            reinterpret_cast<const __half*>(in),
            reinterpret_cast<const __half*>(weight),
            bias ? reinterpret_cast<const __half*>(bias) : nullptr,
            in_features, out_features, rows
        );
    case LLAISYS_DTYPE_BF16:
        return linear_bf16(
            reinterpret_cast<__nv_bfloat16*>(out),
            reinterpret_cast<const __nv_bfloat16*>(in),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            bias ? reinterpret_cast<const __nv_bfloat16*>(bias) : nullptr,
            in_features, out_features, rows
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::tianshu
