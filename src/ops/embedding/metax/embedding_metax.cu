/**
 * @file embedding_metax.cu
 * @brief Embedding 算子的 MetaX MACA 实现
 *
 * 基于 NVIDIA CUDA 版本适配，MACA SDK 提供 CUDA 兼容 API。
 * 每个 block 处理一个 index（一行），block 内线程协作复制，使用 float4 向量化。
 */

#include "embedding_metax.hpp"
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

// ============================================================
//  F32 Embedding Kernel — float4 向量化
// ============================================================
template <typename IndexT>
__global__ void embedding_kernel_f32(
    float* __restrict__ out,
    const IndexT* __restrict__ indices,
    const float* __restrict__ weight,
    size_t num_indices,
    size_t embedding_dim
) {
    size_t row = blockIdx.x;
    if (row >= num_indices) return;

    IndexT idx = indices[row];
    const float* src = weight + idx * embedding_dim;
    float* dst = out + row * embedding_dim;

    size_t vec_dim = embedding_dim / 4;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        dst4[i] = src4[i];
    }

    size_t tail_start = vec_dim * 4;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ============================================================
//  F16 Embedding Kernel — float4 向量化
// ============================================================
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
        throw std::invalid_argument("Unsupported dtype for MACA embedding");
    }

    checkMaca(cudaGetLastError(), "Failed to launch embedding kernel");
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::metax {

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
    (void)vocab_size;

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
        throw std::invalid_argument("Unsupported index dtype for MACA embedding");
    }
}

} // namespace llaisys::ops::metax
