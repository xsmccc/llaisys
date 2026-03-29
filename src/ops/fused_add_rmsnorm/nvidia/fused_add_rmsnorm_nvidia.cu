/**
 * @file fused_add_rmsnorm_nvidia.cu
 * @brief Fused Add + RMSNorm — 单 kernel 完成残差连接与归一化
 *
 * ── 融合原理 ──────────────────────────────────────────────
 *   分离版本（2 次 kernel launch, 3 次 global memory 读写）:
 *     kernel1: residual_out[j] = a[j] + b[j]        // 2R + 1W
 *     kernel2: out[j] = w[j] * residual_out[j] * inv_rms  // 2R + 1W
 *   → 总计: 4R + 2W global memory transactions
 *
 *   融合版本（1 次 kernel launch, 1 次 pass）:
 *     sum = a[j] + b[j]                              // 2R (a, b)
 *     residual_out[j] = sum                           // 1W (residual)
 *     sum_sq += sum * sum                             // register
 *     inv_rms = rsqrt(sum_sq/d + eps)                 // shared mem
 *     out[j] = w[j] * sum * inv_rms                   // 1R (weight) + 1W (out)
 *   → 总计: 3R + 2W, 省 1 次 hidden_size 的全局读
 *   → 同时省 1 次 kernel launch 开销 (~5μs)
 *
 * ── 3-pass 设计 ─────────────────────────────────────────
 *   Pass 1: 向量化读 a+b → 写 residual_out + 累加 sum_sq
 *   Reduce: block_reduce_sum → inv_rms (shared memory 广播)
 *   Pass 2: 向量化读 residual_out(L1 cache hot) + weight → 写 out
 *
 * ── 线程映射 ─────────────────────────────────────────────
 *   grid:  (rows, 1, 1)  — 每个 block 处理一行（decode: rows=1）
 *   block: (256, 1, 1)   — 256 threads = 8 warps
 */

#include "fused_add_rmsnorm_nvidia.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <iostream>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
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

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    __half h = __float2half(v);
    return *reinterpret_cast<const llaisys::fp16_t*>(&h);
}

template <>
__device__ __forceinline__ llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    __nv_bfloat16 b = __float2bfloat16(v);
    return *reinterpret_cast<const llaisys::bf16_t*>(&b);
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
    if (lane == 0) s_partial[warp_id] = val;
    __syncthreads();

    val = (lane < num_warps) ? s_partial[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

// ============================================================
//  Fused Add + RMSNorm Kernel — float4 向量化
// ============================================================
template <typename T>
__global__ void fused_add_rmsnorm_kernel(
    T* __restrict__ out,          // RMSNorm 输出
    T* __restrict__ residual_out, // 残差输出 (a + b)
    const T* __restrict__ a,      // 输入 a (例如 attn_out)
    const T* __restrict__ b,      // 输入 b (例如 residual)
    const T* __restrict__ weight, // RMSNorm 权重
    size_t rows,
    size_t cols,
    float eps
) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T);
    size_t vec_cols = cols / ELT_PER_VEC;
    size_t tail_start = vec_cols * ELT_PER_VEC;

    // 行指针
    const float4* a4 = reinterpret_cast<const float4*>(a + row * cols);
    const float4* b4 = reinterpret_cast<const float4*>(b + row * cols);
    float4* res4 = reinterpret_cast<float4*>(residual_out + row * cols);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out + row * cols);

    const T* row_a = a + row * cols;
    const T* row_b = b + row * cols;
    T* row_res = residual_out + row * cols;
    T* row_out = out + row * cols;

    // ═══ Pass 1: Add + accumulate sum_sq ═══
    float sum_sq = 0.0f;
    for (size_t j = threadIdx.x; j < vec_cols; j += blockDim.x) {
        float4 va = a4[j];  // LD.128
        float4 vb = b4[j];  // LD.128
        const T* ea = reinterpret_cast<const T*>(&va);
        const T* eb = reinterpret_cast<const T*>(&vb);

        T sum_elts[ELT_PER_VEC];
        #pragma unroll
        for (size_t k = 0; k < ELT_PER_VEC; k++) {
            float fa = to_float(ea[k]);
            float fb = to_float(eb[k]);
            float s = fa + fb;
            sum_elts[k] = from_float<T>(s);
            sum_sq += s * s;
        }
        res4[j] = *reinterpret_cast<const float4*>(sum_elts);  // ST.128
    }
    // 标量尾部
    for (size_t j = tail_start + threadIdx.x; j < cols; j += blockDim.x) {
        float fa = to_float(row_a[j]);
        float fb = to_float(row_b[j]);
        float s = fa + fb;
        row_res[j] = from_float<T>(s);
        sum_sq += s * s;
    }

    // ═══ Reduce: block_reduce_sum → inv_rms ═══
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // ═══ Pass 2: RMSNorm (读 residual_out + weight → 写 out) ═══
    // residual_out 刚写完，大概率还在 L1/L2 cache 中
    for (size_t j = threadIdx.x; j < vec_cols; j += blockDim.x) {
        float4 raw_r = res4[j];  // LD.128 (L1 cache hit)
        float4 raw_w = w4[j];    // LD.128
        const T* er = reinterpret_cast<const T*>(&raw_r);
        const T* ew = reinterpret_cast<const T*>(&raw_w);

        T result[ELT_PER_VEC];
        #pragma unroll
        for (size_t k = 0; k < ELT_PER_VEC; k++) {
            float r = to_float(er[k]);
            float w = to_float(ew[k]);
            result[k] = from_float<T>(r * w * inv_rms);
        }
        out4[j] = *reinterpret_cast<const float4*>(result);  // ST.128
    }
    // 标量尾部
    for (size_t j = tail_start + threadIdx.x; j < cols; j += blockDim.x) {
        float r = to_float(row_res[j]);
        float w = to_float(weight[j]);
        row_out[j] = from_float<T>(r * w * inv_rms);
    }
}

// ============ Launch 函数 ============
void launch_fused_add_rmsnorm(
    std::byte* out_ptr,
    std::byte* residual_out_ptr,
    const std::byte* a_ptr,
    const std::byte* b_ptr,
    const std::byte* weight_ptr,
    llaisysDataType_t dtype,
    size_t rows,
    size_t cols,
    float eps
) {
    constexpr int BLOCK = 256;
    dim3 grid(rows);
    dim3 block(BLOCK);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        fused_add_rmsnorm_kernel<float><<<grid, block>>>(
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<float*>(residual_out_ptr),
            reinterpret_cast<const float*>(a_ptr),
            reinterpret_cast<const float*>(b_ptr),
            reinterpret_cast<const float*>(weight_ptr),
            rows, cols, eps);
        break;
    case LLAISYS_DTYPE_F16:
        fused_add_rmsnorm_kernel<llaisys::fp16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<llaisys::fp16_t*>(residual_out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(a_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(b_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(weight_ptr),
            rows, cols, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        fused_add_rmsnorm_kernel<llaisys::bf16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            reinterpret_cast<llaisys::bf16_t*>(residual_out_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(a_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(b_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(weight_ptr),
            rows, cols, eps);
        break;
    default:
        throw std::runtime_error("fused_add_rmsnorm: unsupported dtype");
    }

    checkCuda(cudaGetLastError(), "fused_add_rmsnorm kernel launch");
}

} // anonymous namespace

namespace llaisys::ops::nvidia {

void fused_add_rmsnorm(
    void* out, void* residual_out,
    const void* a, const void* b,
    const void* weight,
    llaisysDataType_t dtype,
    size_t cols, size_t rows,
    float eps
) {
    launch_fused_add_rmsnorm(
        reinterpret_cast<std::byte*>(out),
        reinterpret_cast<std::byte*>(residual_out),
        reinterpret_cast<const std::byte*>(a),
        reinterpret_cast<const std::byte*>(b),
        reinterpret_cast<const std::byte*>(weight),
        dtype, rows, cols, eps);
}

} // namespace llaisys::ops::nvidia
