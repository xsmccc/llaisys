/**
 * @file linear_quantized_nvidia.cu
 * @brief W8A32 量化 Linear 算子的 CUDA 实现
 *
 * ── 量化方案 ────────────────────────────────────────────
 *   Per-channel 对称量化 (absmax):
 *     scale[n] = max(|W_fp32[n, :]|) / 127.0
 *     W_int8[n, k] = round(W_fp32[n, k] / scale[n])
 *     还原: W_fp32[n, k] = W_int8[n, k] * scale[n]
 *
 * ── 执行流程 ────────────────────────────────────────────
 *   1. 分配 F32 临时缓冲区 [N, K]
 *   2. CUDA Kernel: dequant INT8 → F32 (向量化 int4 读取 + float4 写入)
 *   3. cublasSgemm: out = in @ dequant_weight^T
 *   4. Bias add kernel (复用 linear_nvidia.cu 的逻辑)
 *   5. 释放临时 buffer
 *
 * ── 性能说明 ────────────────────────────────────────────
 *   虽然 dequant + sgemm 不如原生 INT8 GEMM (cublasLt) 快,
 *   但此方案兼容所有 sm_60+ GPU, 且权重存储减少 ~4x。
 *   后续可升级为 cublasLtMatmul INT8 路径 (sm_75+)。
 */

#include "linear_quantized_nvidia.hpp"
#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <mutex>

namespace {

// ============================================================
//  全局 cuBLAS Handle（与 linear_nvidia.cu 共享模式）
// ============================================================
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
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
//  Dequantize Kernel: INT8 → F32
// ============================================================
//  weight_int8: [N, K]  (行主序)
//  scales:      [N]
//  weight_f32:  [N, K]  (行主序)
//
//  weight_f32[n, k] = weight_int8[n, k] * scales[n]
//
//  使用 int4 向量化读取 INT8 (每次 4 字节)，float4 写入 F32
//  每个线程处理 4 个 INT8 元素

// ---- 向量化版本: K 能被 4 整除 ----
__global__ void dequant_int8_to_f32_vec4(
    float* __restrict__ weight_f32,
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scales,
    size_t N,  // out_features
    size_t K   // in_features
) {
    // 每线程处理 4 个连续的 INT8 元素
    const size_t K4 = K / 4;
    const size_t total_vec = N * K4;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vec;
         idx += gridDim.x * blockDim.x) {
        size_t n = idx / K4;          // 行号 (out_features)
        float s = scales[n];          // per-channel scale

        // 向量化读取 4 个 INT8
        const char4* src = reinterpret_cast<const char4*>(weight_int8);
        char4 v = src[idx];

        // 反量化并写入 F32
        float4 out;
        out.x = static_cast<float>(v.x) * s;
        out.y = static_cast<float>(v.y) * s;
        out.z = static_cast<float>(v.z) * s;
        out.w = static_cast<float>(v.w) * s;

        reinterpret_cast<float4*>(weight_f32)[idx] = out;
    }
}

// ---- 标量回退版本: K 不能被 4 整除 ----
__global__ void dequant_int8_to_f32_scalar(
    float* __restrict__ weight_f32,
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scales,
    size_t N,
    size_t K
) {
    size_t total = N * K;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
        size_t n = idx / K;
        float s = scales[n];
        weight_f32[idx] = static_cast<float>(weight_int8[idx]) * s;
    }
}

// ============================================================
//  Bias Add Kernel (F32) — 与 linear_nvidia.cu 相同逻辑
// ============================================================
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

__global__ void add_bias_f32_vec4(
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

inline void launch_bias_f32(float* out, const float* bias, size_t rows, size_t N) {
    constexpr int THREADS = 256;
    if (N % 4 == 0) {
        size_t total_vec = rows * (N / 4);
        int blocks = static_cast<int>((total_vec + THREADS - 1) / THREADS);
        add_bias_f32_vec4<<<blocks, THREADS>>>(out, bias, rows, N);
    } else {
        size_t total = rows * N;
        int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        add_bias_f32_scalar<<<blocks, THREADS>>>(out, bias, rows, N);
    }
    checkCuda(cudaGetLastError(), "add_bias_f32 (quantized) launch failed");
}

// ============================================================
//  Dequant + cuBLAS SGEMM + Bias
// ============================================================
void linear_quantized_impl(
    float* out,                 // [M, N]
    const float* in,            // [M, K]
    const int8_t* weight_int8,  // [N, K]
    const float* scales,        // [N]
    const float* bias,          // [N] or nullptr
    size_t K,
    size_t N,
    size_t M
) {
    cublasHandle_t handle = get_cublas_handle();

    // ── 1. 分配 F32 临时缓冲区 ──
    float* weight_f32 = nullptr;
    checkCuda(cudaMalloc(&weight_f32, N * K * sizeof(float)),
              "Failed to allocate dequantized weight buffer");

    // ── 2. Dequantize: INT8 → F32 ──
    constexpr int THREADS = 256;
    if (K % 4 == 0) {
        size_t total_vec = N * (K / 4);
        int blocks = static_cast<int>((total_vec + THREADS - 1) / THREADS);
        dequant_int8_to_f32_vec4<<<blocks, THREADS>>>(
            weight_f32, weight_int8, scales, N, K);
    } else {
        size_t total = N * K;
        int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        dequant_int8_to_f32_scalar<<<blocks, THREADS>>>(
            weight_f32, weight_int8, scales, N, K);
    }
    checkCuda(cudaGetLastError(), "dequant_int8_to_f32 launch failed");

    // ── 3. cuBLAS SGEMM: out = in @ weight_f32^T ──
    //   行主序适配 (与 linear_nvidia.cu 一致):
    //   out^T[N,M] = weight[N,K] × in^T[K,M]
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    checkCublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,                          // transa: 转置 weight
            CUBLAS_OP_N,                          // transb: in 不转置
            static_cast<int>(N),                  // m = N
            static_cast<int>(M),                  // n = M
            static_cast<int>(K),                  // k = K
            &alpha,
            weight_f32,                           // A = weight_f32, lda = K
            static_cast<int>(K),                  // lda
            in,                                   // B = in, ldb = K
            static_cast<int>(K),                  // ldb
            &beta,
            out,                                  // C = out, ldc = N
            static_cast<int>(N)                   // ldc
        ),
        "cublasSgemm (quantized linear) failed"
    );

    // ── 4. Bias Add ──
    if (bias != nullptr) {
        launch_bias_f32(out, bias, M, N);
    }

    // ── 5. 释放临时 buffer ──
    checkCuda(cudaFree(weight_f32), "Failed to free dequantized weight buffer");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::nvidia {

void linear_quantized(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_int8,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
) {
    linear_quantized_impl(
        reinterpret_cast<float*>(out),
        reinterpret_cast<const float*>(in),
        reinterpret_cast<const int8_t*>(weight_int8),
        reinterpret_cast<const float*>(scales),
        bias ? reinterpret_cast<const float*>(bias) : nullptr,
        in_features,   // K
        out_features,  // N
        rows           // M
    );
}

} // namespace llaisys::ops::nvidia
