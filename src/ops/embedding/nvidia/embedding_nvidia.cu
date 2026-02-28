/**
 * @file embedding_nvidia.cu
 * @brief Embedding 算子的 CUDA 实现
 *
 * ── 算子特性分析 ──────────────────────────────────────────
 *   类型：纯访存密集型（Memory Bound）
 *   操作：按 index 从 weight[vocab_size, embedding_dim] 中复制整行到 out
 *   计算量：零（只有内存拷贝）
 *   访存模式：
 *     - weight 读取：不规则（index 决定行号，可能跳跃访问）
 *     - out 写入：连续（按 index 顺序依次写入）
 *     - index 读取：连续（顺序遍历）
 *
 * ── 优化策略 ────────────────────────────────────────────
 *   1. 每个 block 处理一个 index（一行），block 内线程协作复制该行
 *   2. 使用 float4/half2 向量化访存，减少内存事务数
 *   3. embedding_dim 通常是 128 的倍数（如 1536, 4096），天然对齐
 *
 * ── 线程映射 ────────────────────────────────────────────
 *   grid:  (num_indices, 1, 1)  — 每个 block 负责一个 index
 *   block: (256, 1, 1)          — block 内线程协作复制一行
 *   每线程处理 4 个 float（float4）或 8 个 half（通过 float4）
 */

#include "embedding_nvidia.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================
//  F32 Embedding Kernel — float4 向量化
// ============================================================
// 每个 block 处理 1 行（1 个 index）
// 每线程用 float4 一次复制 16 字节（4 个 float）
template <typename IndexT>
__global__ void embedding_kernel_f32(
    float* __restrict__ out,
    const IndexT* __restrict__ indices,
    const float* __restrict__ weight,
    size_t num_indices,
    size_t embedding_dim
) {
    // blockIdx.x = 第几个 index
    size_t row = blockIdx.x;
    if (row >= num_indices) return;

    IndexT idx = indices[row];
    const float* src = weight + idx * embedding_dim;
    float* dst = out + row * embedding_dim;

    // float4 向量化：每线程每步复制 4 个 float
    size_t vec_dim = embedding_dim / 4;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    // 尾部处理（embedding_dim 不是 4 的倍数时）
    size_t tail_start = vec_dim * 4;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================================================
//  F16 Embedding Kernel — float4 向量化
// ============================================================
// half 是 2 字节，float4 是 16 字节 → 一次复制 8 个 half
template <typename IndexT>
__global__ void embedding_kernel_f16(
    llaisys::fp16_t* __restrict__ out,
    const IndexT* __restrict__ indices,
    const llaisys::fp16_t* __restrict__ weight,
    size_t num_indices,
    size_t embedding_dim
) {
    size_t row = blockIdx.x;
    if (row >= num_indices) return;

    IndexT idx = indices[row];
    const llaisys::fp16_t* src = weight + idx * embedding_dim;
    llaisys::fp16_t* dst = out + row * embedding_dim;

    // 用 float4 搬运：16B = 8 个 fp16
    size_t vec_dim = embedding_dim / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    // 尾部标量处理
    size_t tail_start = vec_dim * 8;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================================================
//  BF16 Embedding Kernel — float4 向量化
// ============================================================
template <typename IndexT>
__global__ void embedding_kernel_bf16(
    llaisys::bf16_t* __restrict__ out,
    const IndexT* __restrict__ indices,
    const llaisys::bf16_t* __restrict__ weight,
    size_t num_indices,
    size_t embedding_dim
) {
    size_t row = blockIdx.x;
    if (row >= num_indices) return;

    IndexT idx = indices[row];
    const llaisys::bf16_t* src = weight + idx * embedding_dim;
    llaisys::bf16_t* dst = out + row * embedding_dim;

    // 用 float4 搬运：16B = 8 个 bf16
    size_t vec_dim = embedding_dim / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    size_t tail_start = vec_dim * 8;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================================================
//  Kernel 启动函数
// ============================================================
template <typename IndexT>
void launch_embedding_typed(
    std::byte* out_ptr,
    llaisysDataType_t out_type,
    const IndexT* index_ptr,
    size_t num_indices,
    const std::byte* weight_ptr,
    size_t embedding_dim
) {
    constexpr int THREADS = 256;
    int blocks = static_cast<int>(num_indices);

    switch (out_type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel_f32<<<blocks, THREADS>>>(
            reinterpret_cast<float*>(out_ptr),
            index_ptr,
            reinterpret_cast<const float*>(weight_ptr),
            num_indices, embedding_dim
        );
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel_f16<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            index_ptr,
            reinterpret_cast<const llaisys::fp16_t*>(weight_ptr),
            num_indices, embedding_dim
        );
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel_bf16<<<blocks, THREADS>>>(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            index_ptr,
            reinterpret_cast<const llaisys::bf16_t*>(weight_ptr),
            num_indices, embedding_dim
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for CUDA embedding");
    }

    checkCuda(cudaGetLastError(), "Failed to launch embedding kernel");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::nvidia {

void embedding(
    std::byte* out_ptr,
    llaisysDataType_t out_type,
    const std::byte* index_ptr,
    size_t num_indices,
    llaisysDataType_t index_dtype,
    const std::byte* weight_ptr,
    size_t vocab_size,
    size_t embedding_dim
) {
    (void)vocab_size; // 暂不做越界检查（GPU 端检查开销大）

    if (index_dtype == LLAISYS_DTYPE_I64) {
        launch_embedding_typed<int64_t>(
            out_ptr, out_type,
            reinterpret_cast<const int64_t*>(index_ptr),
            num_indices, weight_ptr, embedding_dim
        );
    } else if (index_dtype == LLAISYS_DTYPE_I32) {
        launch_embedding_typed<int32_t>(
            out_ptr, out_type,
            reinterpret_cast<const int32_t*>(index_ptr),
            num_indices, weight_ptr, embedding_dim
        );
    } else {
        throw std::invalid_argument("Unsupported index dtype for CUDA embedding");
    }
}

} // namespace llaisys::ops::nvidia
