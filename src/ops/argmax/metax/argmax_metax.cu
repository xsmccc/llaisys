/**
 * @file argmax_metax.cu
 * @brief Argmax 算子的 MetaX MACA 实现
 *
 * 基于 NVIDIA CUDA 版本适配，MACA SDK 提供 CUDA 兼容 API。
 * 实现策略：
 *   - 小数据（numel <= 1024）：单 block warp shuffle 归约
 *   - 大数据：多 block 两阶段归约（Phase1 并行 + Phase2 汇总）
 */

#include "argmax_metax.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <mutex>

namespace {

inline void checkMaca(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[MACA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============ 类型转换辅助 ============
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(llaisys::fp16_t v) {
    return __half2float(*reinterpret_cast<const __half*>(&v._v));
}
__device__ __forceinline__ float to_float(llaisys::bf16_t v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v._v));
}

__device__ __forceinline__ float from_float_to_f32(float v) { return v; }

// ============================================================
//  Warp 级归约：找 warp 内最大值及其索引
// ============================================================
__device__ __forceinline__ void warp_reduce_max(float& val, size_t& idx) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta);
        size_t other_idx = __shfl_down_sync(0xffffffff, idx, delta);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ============================================================
//  Argmax Kernel — 单 block 归约（小数据路径）
// ============================================================
template <typename T, typename IndexT>
__global__ void argmax_kernel(
    IndexT* __restrict__ max_idx_out,
    T* __restrict__ max_val_out,
    const T* __restrict__ vals,
    size_t numel
) {
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_reduce_max(local_max, local_idx);

    __shared__ float s_val[32];
    __shared__ size_t s_idx[32];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_val[lane] : -FLT_MAX;
        size_t idx = (lane < num_warps) ? s_idx[lane] : 0;

        warp_reduce_max(val, idx);

        if (lane == 0) {
            *max_idx_out = static_cast<IndexT>(idx);
            *max_val_out = vals[idx];
        }
    }
}

// ============================================================
//  多 Block Argmax — Phase 1: 每个 block 归约一部分
// ============================================================
template <typename T>
__global__ void argmax_phase1(
    float* __restrict__ block_vals,
    size_t* __restrict__ block_idxs,
    const T* __restrict__ vals,
    size_t numel
) {
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += blockDim.x * gridDim.x)
    {
        float v = to_float(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    warp_reduce_max(local_max, local_idx);

    __shared__ float s_val[32];
    __shared__ size_t s_idx[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_val[lane] : -FLT_MAX;
        size_t idx = (lane < num_warps) ? s_idx[lane] : 0;
        warp_reduce_max(val, idx);

        if (lane == 0) {
            block_vals[blockIdx.x] = val;
            block_idxs[blockIdx.x] = idx;
        }
    }
}

// ============================================================
//  多 Block Argmax — Phase 2: 归约所有 block 的结果
// ============================================================
template <typename T, typename IndexT>
__global__ void argmax_phase2(
    IndexT* __restrict__ max_idx_out,
    T* __restrict__ max_val_out,
    const T* __restrict__ vals,
    const float* __restrict__ block_vals,
    const size_t* __restrict__ block_idxs,
    int num_blocks
) {
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        if (v > local_max) {
            local_max = v;
            local_idx = block_idxs[i];
        }
    }

    warp_reduce_max(local_max, local_idx);

    __shared__ float s_val[32];
    __shared__ size_t s_idx[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_val[lane] : -FLT_MAX;
        size_t idx = (lane < num_warps) ? s_idx[lane] : 0;
        warp_reduce_max(val, idx);

        if (lane == 0) {
            *max_idx_out = static_cast<IndexT>(idx);
            *max_val_out = vals[idx];
        }
    }
}

// ============================================================
//  持久化临时缓冲区
// ============================================================
struct TempReduceBuffer {
    float* vals = nullptr;
    size_t* idxs = nullptr;
    int capacity = 0;
    std::mutex mtx;

    void ensure(int num_blocks) {
        std::lock_guard<std::mutex> lock(mtx);
        if (num_blocks <= capacity) return;
        if (vals) cudaFree(vals);
        if (idxs) cudaFree(idxs);
        checkCuda(cudaMalloc(&vals, num_blocks * sizeof(float)),
                  "TempReduceBuffer: cudaMalloc vals failed");
        checkCuda(cudaMalloc(&idxs, num_blocks * sizeof(size_t)),
                  "TempReduceBuffer: cudaMalloc idxs failed");
        capacity = num_blocks;
    }
};

static TempReduceBuffer g_temp_buf;

// ============================================================
//  启动函数
// ============================================================
template <typename T, typename IndexT>
void launch_argmax_typed(
    IndexT* max_idx,
    T* max_val,
    const T* vals,
    size_t numel
) {
    constexpr int THREADS = 256;

    if (numel <= 1024) {
        argmax_kernel<T, IndexT><<<1, THREADS>>>(max_idx, max_val, vals, numel);
        checkMaca(cudaGetLastError(), "Failed to launch argmax kernel (single-block)");
        return;
    }

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int num_blocks = num_sm * 4;
    num_blocks = std::min(num_blocks, static_cast<int>((numel + THREADS - 1) / THREADS));

    g_temp_buf.ensure(num_blocks);

    argmax_phase1<T><<<num_blocks, THREADS>>>(
        g_temp_buf.vals, g_temp_buf.idxs, vals, numel);

    argmax_phase2<T, IndexT><<<1, THREADS>>>(
        max_idx, max_val, vals,
        g_temp_buf.vals, g_temp_buf.idxs, num_blocks);

    checkMaca(cudaGetLastError(), "Failed to launch argmax kernel (multi-block)");
}

template <typename T>
void dispatch_idx_type(
    std::byte* max_idx, llaisysDataType_t idx_dtype,
    std::byte* max_val,
    const std::byte* vals,
    size_t numel
) {
    if (idx_dtype == LLAISYS_DTYPE_I64) {
        launch_argmax_typed<T, int64_t>(
            reinterpret_cast<int64_t*>(max_idx),
            reinterpret_cast<T*>(max_val),
            reinterpret_cast<const T*>(vals),
            numel
        );
    } else if (idx_dtype == LLAISYS_DTYPE_I32) {
        launch_argmax_typed<T, int32_t>(
            reinterpret_cast<int32_t*>(max_idx),
            reinterpret_cast<T*>(max_val),
            reinterpret_cast<const T*>(vals),
            numel
        );
    } else {
        throw std::invalid_argument("Unsupported index dtype for MACA argmax");
    }
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::metax {

void argmax(
    std::byte* max_idx,
    llaisysDataType_t idx_dtype,
    std::byte* max_val,
    const std::byte* vals,
    llaisysDataType_t val_dtype,
    size_t numel
) {
    switch (val_dtype) {
    case LLAISYS_DTYPE_F32:
        dispatch_idx_type<float>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        dispatch_idx_type<llaisys::fp16_t>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        dispatch_idx_type<llaisys::bf16_t>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    default:
        throw std::invalid_argument("Unsupported val dtype for MACA argmax");
    }
}

} // namespace llaisys::ops::metax
