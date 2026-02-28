// 矩阵乘法 - 使用 OpenBLAS 加速
/*
Linear: out = in @ weight.T + bias
X 形状为 [rows, in_features] ——[M, K]
W 形状为 [out_features, in_features] ——[N, K]
out 形状为 [rows, out_features] ——[M, N]

使用 BLAS GEMM: C = alpha * A @ B.T + beta * C
*/
#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cblas.h>
#include <vector>
#include <cstring>

// 通用 Linear 内核（用于非 F32 类型）
template <typename T>
void linear_kernel_generic(
    T* out,
    const T* in,
    const T* weight,
    const T* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
){
    // 对于 F16/BF16，先转换为 F32 计算，再转回
    std::vector<float> in_f32(rows * in_features);
    std::vector<float> weight_f32(out_features * in_features);
    std::vector<float> out_f32(rows * out_features);
    
    // 转换输入
    for (size_t i = 0; i < rows * in_features; ++i) {
        in_f32[i] = llaisys::utils::cast<float>(in[i]);
    }
    
    // 转换权重
    for (size_t i = 0; i < out_features * in_features; ++i) {
        weight_f32[i] = llaisys::utils::cast<float>(weight[i]);
    }
    
    // 使用 cblas_sgemm 计算 out = in @ weight.T
    // C = alpha * op(A) @ op(B) + beta * C
    cblas_sgemm(
        CblasRowMajor,          // 行主序
        CblasNoTrans,           // A 不转置 (in)
        CblasTrans,             // B 转置 (weight)
        rows,                   // M: A 的行数
        out_features,           // N: B 转置后的列数 (out_features)
        in_features,            // K: A 的列数 = B 转置前的列数
        1.0f,                   // alpha
        in_f32.data(),          // A
        in_features,            // lda: A 的列数
        weight_f32.data(),      // B
        in_features,            // ldb: B 转置前的列数
        0.0f,                   // beta (不使用 C 的初始值)
        out_f32.data(),         // C
        out_features            // ldc: C 的列数
    );
    
    // 加偏置 采用广播
    if (bias != nullptr) {
        for (size_t m = 0; m < rows; ++m) {
            for (size_t n = 0; n < out_features; ++n) {
                float bias_val = llaisys::utils::cast<float>(bias[n]);
                out_f32[m * out_features + n] += bias_val;
            }
        }
    }
    
    // 转换输出
    for (size_t i = 0; i < rows * out_features; ++i) {
        out[i] = llaisys::utils::cast<T>(out_f32[i]);
    }
}

// F32 特化版本 - 直接使用 OpenBLAS
void linear_kernel_f32(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
){
    // 使用 cblas_sgemm 计算 out = in @ weight.T
    cblas_sgemm(
        CblasRowMajor,          // 行主序
        CblasNoTrans,           // A 不转置 (in)
        CblasTrans,             // B 转置 (weight)
        rows,                   // M: A 的行数
        out_features,           // N: B 转置后的列数 (out_features)
        in_features,            // K: A 的列数 = B 转置前的列数
        1.0f,                   // alpha
        in,                     // A
        in_features,            // lda: A 的列数
        weight,                 // B
        in_features,            // ldb: B 转置前的列数
        0.0f,                   // beta (不使用 C 的初始值)
        out,                    // C
        out_features            // ldc: C 的列数
    );
    
    // 加偏置
    if (bias != nullptr) {
        for (size_t m = 0; m < rows; ++m) {
            for (size_t n = 0; n < out_features; ++n) {
                out[m * out_features + n] += bias[n];
            }
        }
    }
}

namespace llaisys::ops::cpu{
    void linear(
        std::byte* out,
        llaisysDataType_t dtype,
        const std::byte* in,
        const std::byte* weight,
        const std::byte* bias,
        size_t in_features,
        size_t out_features,
        size_t rows){
        switch (dtype){
            case LLAISYS_DTYPE_F32:
            return linear_kernel_f32(
                reinterpret_cast<float*>(out),
                reinterpret_cast<const float*>(in),
                reinterpret_cast<const float*>(weight),
                reinterpret_cast<const float*>(bias),
                in_features, out_features, rows
            );
            case LLAISYS_DTYPE_BF16:
            return linear_kernel_generic(
                reinterpret_cast<llaisys::bf16_t*>(out),
                reinterpret_cast<const llaisys::bf16_t*>(in),
                reinterpret_cast<const llaisys::bf16_t*>(weight),
                reinterpret_cast<const llaisys::bf16_t*>(bias),
                in_features, out_features, rows
            );
            case LLAISYS_DTYPE_F16:
            return linear_kernel_generic(
                reinterpret_cast<llaisys::fp16_t*>(out),
                reinterpret_cast<const llaisys::fp16_t*>(in),
                reinterpret_cast<const llaisys::fp16_t*>(weight),
                reinterpret_cast<const llaisys::fp16_t*>(bias),
                in_features, out_features, rows
            );
            default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
}