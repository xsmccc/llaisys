/**
 * @file argmax_nvidia.cu
 * @brief Argmax 算子的 CUDA 实现
 *
 * ── 算子特性分析 ---
 *   类型：归约操作（Reduction）
 *   输入：1D 张量 vals[numel]
 *   输出：max_val（最大值），max_idx（最大值索引），都是标量
 *
 * ── 归约的核心挑战 ---
 *   N 个元素需要通过 log2(N) 轮比较合并为 1 个结果。
 *   GPU 上的实现分三层：
 *     1. Warp 内归约：使用 __shfl_down_sync（warp shuffle），无需共享内存
 *     2. Block 内归约：多个 warp 结果写入 shared memory，再由 warp 0 归约
 *     3. Grid 级归约：多个 block 结果通过 atomicCAS 或二次 kernel 合并
 *
 * ── Warp Shuffle 原理 ---
 *   __shfl_down_sync(mask, val, delta) :
 *     lane i 读取 lane (i + delta) 的 val，不经过内存，直接寄存器交换
 *     mask = 0xffffffff 表示 warp 内全部 32 个线程参与
 *
 *   示例（4 个线程找最大值）:
 *     step 0: [3, 7, 1, 9]
 *     step 1 (delta=2): lane0 比较 lane2 → max(3,1)=3, lane1 比较 lane3 → max(7,9)=9
 *             最终 [3, 9, -, -]
 *     step 2 (delta=1): lane0 比较 lane1 → max(3,9)=9
 *             最终 [9, -, -, -]  → lane0 持有全局最大值
 *
 * ── 本实现策略 ---
 *   目前 argmax 测试 shape 最大为 (4096,)。使用单 block 实现：
 *   - 1 个 block，256 线程
 *   - 每个线程用 grid-stride 遍历所有元素，找到自己的局部最大值
 *   - warp shuffle 归约 → shared memory → warp 0 最终归约
 *   - 线程 0 写回结果
 */

#include "argmax_nvidia.hpp"
#include "../../../utils.hpp"
#include "../../../core/context/context.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>
#include <cfloat>
#include <mutex>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
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
    // 5 轮 shuffle：32 → 16 → 8 → 4 → 2 → 1
    for (int delta = 16; delta >= 1; delta >>= 1) {
        // other_val/other_idx 是距离 delta 的线程的值和索引
        // val/idx 是当前线程的值和索引
        float other_val = __shfl_down_sync(0xffffffff, val, delta);
        size_t other_idx = __shfl_down_sync(0xffffffff, idx, delta);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // 归约完成后，lane 0 持有本 warp 的最大值和索引
}

// ============================================================
//  Argmax Kernel — 单 block 归约（保留作为小数据路径）
// ============================================================
// 模板参数 T：输入值类型（float / fp16_t / bf16_t）
// 模板参数 IndexT：输出索引类型（int32_t / int64_t）
template <typename T, typename IndexT>
__global__ void argmax_kernel(
    IndexT* __restrict__ max_idx_out,
    T* __restrict__ max_val_out,
    const T* __restrict__ vals,
    size_t numel
) {
    // 每线程找局部最大值
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    // Grid-stride loop（虽然只有 1 个 block，stride = blockDim.x）
    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp 内归约
    warp_reduce_max(local_max, local_idx);

    // Warp 间归约（通过 shared memory）
    // 每个 warp 的 lane 0 把结果写入 shared memory
    __shared__ float s_val[32];   // 最多 32 个 warp（1024 线程 / 32）
    __shared__ size_t s_idx[32];

    int lane = threadIdx.x % 32;        // warp 内的 lane id
    int warp_id = threadIdx.x / 32;     // 本 block 内的 warp id
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_val[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    // Warp 0 做最终归约
    if (warp_id == 0) {
        // 只让前 num_warps 个 lane 参与
        float val = (lane < num_warps) ? s_val[lane] : -FLT_MAX;
        size_t idx = (lane < num_warps) ? s_idx[lane] : 0;

        warp_reduce_max(val, idx);

        // Lane 0 写回全局结果
        if (lane == 0) {
            *max_idx_out = static_cast<IndexT>(idx);
            // 从 vals 中直接取原值更精确（避免 float 转换丢精度）
            *max_val_out = vals[idx];
        }
    }
}

// ============================================================
//  多 Block Argmax — Phase 1: 每个 block 归约一部分
// ============================================================
// 当 numel 很大（如 vocab_size=151936）时，单 block 只用 1 个 SM，
// 剩余 35 个 SM 完全空闲 → 严重的并行度不足。
//
// 多 Block 策略：
//   Phase 1: N 个 block 各自归约一段数据，输出 N 个局部 (val, idx) 对
//   Phase 2: 1 个 block 归约这 N 个结果 → 最终答案
//
// 数据流:
//   vals[151936] ──Phase1──→ block_vals[144], block_idxs[144] ──Phase2──→ max_val, max_idx
//
template <typename T>
__global__ void argmax_phase1(
    float* __restrict__ block_vals,
    size_t* __restrict__ block_idxs,
    const T* __restrict__ vals,
    size_t numel
) {
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    // Grid-stride loop：每个 block 处理 numel / gridDim.x 个元素
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

    // Block 内归约（warp shuffle + shared memory）
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

        // 每个 block 的 thread 0 输出局部最大值
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
    const T* __restrict__ vals,           // 原始数组，用于取精确值
    const float* __restrict__ block_vals,
    const size_t* __restrict__ block_idxs,
    int num_blocks
) {
    float local_max = -FLT_MAX;
    size_t local_idx = 0;

    // 单 block 内遍历所有 block 的局部结果
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float v = block_vals[i];
        if (v > local_max) {
            local_max = v;
            local_idx = block_idxs[i];
        }
    }

    // Block 内归约
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
            *max_val_out = vals[idx];  // 直接从原始数组取值
        }
    }
}

// ============================================================
//  持久化临时缓冲区（避免每次 cudaMalloc/cudaFree 的开销）
// ============================================================
// cudaMalloc 是同步操作，耗时 ~1ms，不能每次调用都分配。
// 使用 static 变量：首次调用分配，后续复用，只增不缩。
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

    // ── 小数据：单 block 直接归约 ──
    // numel <= 1024 时，256 线程每人只处理 4 个元素，开多 block 反而浪费
    if (numel <= 1024) {
        argmax_kernel<T, IndexT><<<1, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(max_idx, max_val, vals, numel);
        checkCuda(cudaGetLastError(), "Failed to launch argmax kernel (single-block)");
        return;
    }

    // ── 大数据：多 Block 两阶段归约 ──
    // 目标：让所有 SM 都参与工作
    //
    // Block 数量选择：SM 数量 × 4
    //   - 每 SM 4 个 block → 充分隐藏访存延迟
    //   - 不能太多，否则 Phase 2 的归约开销增加
    //
    // 例如 RTX 4070 Laptop (36 SMs):
    //   num_blocks = 36 * 4 = 144
    //   Phase 1: 144 blocks → 每 block 处理 151936/144 ≈ 1055 个元素
    //   Phase 2: 1 block → 归约 144 个结果（trivial）
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int num_blocks = num_sm * 4;
    num_blocks = std::min(num_blocks, static_cast<int>((numel + THREADS - 1) / THREADS));

    // 确保临时缓冲区足够大
    g_temp_buf.ensure(num_blocks);

    // Phase 1: 多 block 并行归约
    argmax_phase1<T><<<num_blocks, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
        g_temp_buf.vals, g_temp_buf.idxs, vals, numel);

    // Phase 2: 单 block 汇总
    argmax_phase2<T, IndexT><<<1, THREADS, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
        max_idx, max_val, vals,
        g_temp_buf.vals, g_temp_buf.idxs, num_blocks);

    checkCuda(cudaGetLastError(), "Failed to launch argmax kernel (multi-block)");
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
        throw std::invalid_argument("Unsupported index dtype for CUDA argmax");
    }
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::nvidia {

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
        throw std::invalid_argument("Unsupported val dtype for CUDA argmax");
    }
}

} // namespace llaisys::ops::nvidia
