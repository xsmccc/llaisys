/**
 * @file linear_quantized_cpu.cpp
 * @brief W8A32 量化 Linear 算子 — CPU 实现 (OpenBLAS)
 *
 * ── 执行流程 ────────────────────────────────────────────
 *   1. 分配 F32 临时缓冲区 [N, K]
 *   2. 反量化: weight_f32[n, k] = weight_int8[n, k] * scales[n]
 *   3. cblas_sgemm: out = in @ weight_f32^T
 *   4. 加偏置 (broadcast)
 */
#include "linear_quantized_cpu.hpp"
#include "../../../utils.hpp"
#include <cblas.h>
#include <vector>
#include <cstring>

namespace llaisys::ops::cpu {

void linear_quantized(
    std::byte* out_raw,
    const std::byte* in_raw,
    const std::byte* weight_raw,
    const std::byte* scales_raw,
    const std::byte* bias_raw,
    size_t in_features,   // K
    size_t out_features,  // N
    size_t rows           // M
) {
    float*       out         = reinterpret_cast<float*>(out_raw);
    const float* in          = reinterpret_cast<const float*>(in_raw);
    const int8_t* weight_int8 = reinterpret_cast<const int8_t*>(weight_raw);
    const float* scales      = reinterpret_cast<const float*>(scales_raw);
    const float* bias        = bias_raw ? reinterpret_cast<const float*>(bias_raw) : nullptr;

    const size_t K = in_features;
    const size_t N = out_features;
    const size_t M = rows;

    // ── 1. 反量化 INT8 → F32 ──
    //  weight_f32[n, k] = weight_int8[n, k] * scales[n]
    std::vector<float> weight_f32(N * K);

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        float s = scales[n];
        const int8_t* row = weight_int8 + n * K;
        float* dst = weight_f32.data() + n * K;
        for (size_t k = 0; k < K; ++k) {
            dst[k] = static_cast<float>(row[k]) * s;
        }
    }

    // ── 2. cblas_sgemm: out = in @ weight_f32^T ──
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,           // A = in (不转置)
        CblasTrans,             // B = weight_f32 (转置)
        static_cast<int>(M),    // M: 行数
        static_cast<int>(N),    // N: 输出特征维度
        static_cast<int>(K),    // K: 输入特征维度
        1.0f,                   // alpha
        in,                     // A
        static_cast<int>(K),    // lda
        weight_f32.data(),      // B
        static_cast<int>(K),    // ldb
        0.0f,                   // beta
        out,                    // C
        static_cast<int>(N)     // ldc
    );

    // ── 3. 加偏置 (broadcast) ──
    if (bias != nullptr) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                out[m * N + n] += bias[n];
            }
        }
    }
}

} // namespace llaisys::ops::cpu
