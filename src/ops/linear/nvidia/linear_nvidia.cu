/**
 * @file linear_nvidia.cu
 * @brief Linear (全连接层) 算子的 CUDA 实现 — 使用 cuBLAS
 *
 * ── 算子公式 ---
 *   out = in @ weight.T + bias
 *   in:     [M, K]  (rows × in_features)
 *   weight: [N, K]  (out_features × in_features)
 *   bias:   [N]     (out_features)    — 可选
 *   out:    [M, N]  (rows × out_features)
 *
 * ── cuBLAS 行主序适配 ---
 *   cuBLAS 默认列主序。对于行主序数据，利用转置恒等式：
 *     out^T = weight @ in^T
 *   即 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
 *                  alpha, weight, K, in, K, beta, out, N)
 *
 *   推导过程：
 *     行主序 [M,K] 矩阵 → cuBLAS 视为 列主序 [K,M]（即 in^T）
 *     行主序 [N,K] 矩阵 → cuBLAS 视为 列主序 [K,N]（即 weight^T）
 *     我们要 out^T[N,M] = weight[N,K] × in^T[K,M]
 *     cuBLAS 拿到的是 weight^T[K,N]，需转置 → transa = CUBLAS_OP_T
 *     cuBLAS 拿到的是 in^T[K,M]，不需转置 → transb = CUBLAS_OP_N
 *
 * ── Bias 处理 ---
 *   GEMM 后用简单的 elementwise kernel 广播加 bias
 */

#include "linear_nvidia.hpp"
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
// cuBLAS 的所有 API 调用都需要一个 handle 来管理内部状态（选择 GPU、流绑定等）。
// 使用全局 static handle 而非从 Resource 传入，是因为算子调用层不直接暴露 Resource 对象。
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() { // C++11 线程安全的单例初始化
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("[cuBLAS] Failed to create handle");
        }
    });
    return handle;
}

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

inline void checkCublas(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cuBLAS ERROR] " << msg << ": status=" << static_cast<int>(st) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================
//  SM 版本检测
// ============================================================
int get_sm_version() {
    static int ver = 0;
    static std::once_flag flag;
    std::call_once(flag, []() {
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ver = major * 10 + minor;
    });
    return ver;
}

// ============================================================
//  Bias Add Kernel — 广播加偏置
// ============================================================
//  out[i, j] += bias[j]
//
//  主路径：float4 向量化（128-bit 访存）
//  - F32:  float4 = 4 个 float，每线程处理 4 列
//  - F16:  float4 = 8 个 __half，用 __half2 SIMD 算术
//  - BF16: float4 = 8 个 __nv_bfloat16，用 __nv_bfloat162 SIMD 算术
//
//  回退路径：当 out_features 不能被 VEC 整除时（小尺寸），用标量 kernel

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
    constexpr size_t VEC = 4;                         // float4 = 4 floats
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

// ---- F16 向量化：float4 = 8 halfs，用 __half2 SIMD 加法 ----
__global__ void add_bias_f16(
    __half* __restrict__ out,
    const __half* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    constexpr size_t VEC = 8;                         // float4 = 8 halfs
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

// ---- BF16 向量化：float4 = 8 bf16s，用 __nv_bfloat162 SIMD 加法 ----
__global__ void add_bias_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    size_t rows,
    size_t out_features
) {
    constexpr size_t VEC = 8;                         // float4 = 8 bf16s
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
    checkCuda(cudaGetLastError(), "add_bias_f32 launch failed");
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
    checkCuda(cudaGetLastError(), "add_bias_f16 launch failed");
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
    checkCuda(cudaGetLastError(), "add_bias_bf16 launch failed");
}

// ============================================================
//  F32 Linear — cublasGemmEx Tensor Core (sm>=70) / cublasSgemm fallback
// ============================================================
void linear_f32(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    size_t in_features,   // K
    size_t out_features,  // N
    size_t rows           // M
) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    if (get_sm_version() >= 70) {
        // ── Tensor Core 加速路径 (Volta+) ──
        // cublasGemmEx + CUBLAS_COMPUTE_32F + TENSOR_OP:
        //   cuBLAS 选择最优 GEMM 内核 (TF32 Tensor Core on sm>=80)
        //   零额外显存开销，保持 FP32 累加精度
        checkCublas(
            cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)out_features, (int)rows, (int)in_features,
                &alpha,
                weight, CUDA_R_32F, (int)in_features,
                in, CUDA_R_32F, (int)in_features,
                &beta,
                out, CUDA_R_32F, (int)out_features,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx F32 TC-accelerated");
    } else {
        // ── FP32 FMA 回退路径 (sm < 70) ──
        checkCublas(
            cublasSgemm(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(out_features),
                static_cast<int>(rows),
                static_cast<int>(in_features),
                &alpha,
                weight, static_cast<int>(in_features),
                in, static_cast<int>(in_features),
                &beta,
                out, static_cast<int>(out_features)
            ),
            "cublasSgemm failed"
        );
    }

    // 加偏置（float4 向量化 / 标量回退）
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

    // 用 F32 累加以匹配 PyTorch 默认行为，提高精度
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,            // transa
            CUBLAS_OP_N,            // transb
            static_cast<int>(out_features),   // m = N
            static_cast<int>(rows),           // n = M
            static_cast<int>(in_features),    // k = K
            &alpha,
            weight, CUDA_R_16F,               // A
            static_cast<int>(in_features),    // lda
            in, CUDA_R_16F,                   // B
            static_cast<int>(in_features),    // ldb
            &beta,
            out, CUDA_R_16F,                  // C
            static_cast<int>(out_features),   // ldc
            CUBLAS_COMPUTE_32F,               // F32 累加精度更高
            CUBLAS_GEMM_DEFAULT               // algo
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

    // BF16 GEMM 用 F32 累加精度更高
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
            CUBLAS_COMPUTE_32F,    // BF16 输入输出，F32 累加
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
namespace llaisys::ops::nvidia {

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

} // namespace llaisys::ops::nvidia
