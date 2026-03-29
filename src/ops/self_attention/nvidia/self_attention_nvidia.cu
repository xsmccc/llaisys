/**
 * @file self_attention_nvidia.cu
 * @brief Causal Self-Attention 算子的 CUDA 实现（融合 kernel）
 *
 * ── 算子公式 ---
 *   Attention(Q, K, V) = softmax(Q @ K^T * scale + causal_mask) @ V
 *
 *   Q:        [seq_len,   nhead,    head_dim]
 *   K:        [total_len, kv_head,  head_dim]
 *   V:        [total_len, kv_head,  v_head_dim]
 *   输出:     [seq_len,   nhead,    v_head_dim]
 *
 *   支持 GQA (Grouped Query Attention):
 *     nhead % kv_head == 0, 每 group_size = nhead/kv_head 个 Q 头共享一组 KV
 *
 * ── 融合策略 ---
 *   将整个 self-attention 融合为一个 kernel，避免中间结果来回全局内存：
 *     Phase 1: Q @ K^T — 每线程负责若干 t，逐元素点积后写入 shared memory
 *     Phase 2: Causal mask — 将 t > current_pos 的分数设为 -inf
 *     Phase 3: Safe softmax — block reduce 求 max → exp → reduce 求 sum → 归一化
 *     Phase 4: scores @ V — 线程映射到 v_head_dim，读共享内存中的 scores 加权求和
 *
 * ── 线程映射 ---
 *   grid:  (seq_len, nhead)    — 每个 block 处理一个 (query_pos, head) 组合
 *   block: (256, 1, 1)         — block 内 256 线程协作
 *
 * ── Shared Memory 布局 ---
 *   dynamic: scores[total_len] + warp_buf[WARPS]
 *   static:  s_max, s_sum (用于广播归约结果)
 *
 * ── 所有计算统一使用 F32 ---
 *   F16/BF16 输入先转 float 再计算，最终转回写出
 *   这与 CPU 版本保持一致，确保数值正确性
 */

#include "self_attention_nvidia.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS   = THREADS / 32;  // 8

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================
//  类型转换辅助 — 设备端
// ============================================================
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ============================================================
//  Block Reduce — warp shuffle + shared memory
// ============================================================

/**
 * block_reduce_max:
 *   1. 每个 warp 内用 __shfl_down_sync 求 max → lane 0 持有 warp max
 *   2. lane 0 写入 warp_buf[warp_id]
 *   3. __syncthreads()
 *   4. 第一个 warp 从 warp_buf 读取所有 warp 的 max，再 shuffle reduce
 *   5. thread 0 持有全局 max
 */
__device__ __forceinline__
float block_reduce_max(float val, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    // Warp 内归约
    for (int delta = 16; delta >= 1; delta >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));

    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();

    // 第一个 warp 汇总所有 warp 的结果
    val = (threadIdx.x < WARPS) ? warp_buf[threadIdx.x] : -INFINITY;
    if (wid == 0) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));
    }
    return val; // 仅 thread 0 持有正确结果
}

/**
 * block_reduce_sum: 与 max 同构，但用加法和 0 初值
 */
__device__ __forceinline__
float block_reduce_sum(float val, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();

    val = (threadIdx.x < WARPS) ? warp_buf[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

// ============================================================
//  融合 Self-Attention Kernel（优化版）
// ============================================================
//  优化要点：
//   1. Q 加载到共享内存（预转 float），避免重复全局内存读取
//   2. Phase 1 (QK^T) 使用 float4 向量化读取 K，减少指令量和内存事务数
//   3. Phase 4 (scores×V) V 读取已天然 coalesced，保持原始路径
//   4. 所有路径均有 scalar tail 处理非对齐 head_dim
template <typename T>
__global__ void self_attention_fused(
    T* __restrict__       attn_val,   // [seq_len, nhead, v_head_dim]
    const T* __restrict__ q,          // [seq_len, nhead, head_dim]
    const T* __restrict__ k,          // [total_len, kv_head, head_dim]
    const T* __restrict__ v,          // [total_len, kv_head, v_head_dim]
    size_t seq_len, // query 序列长度
    size_t total_len,   // KV 序列长度（包含 padding）
    size_t nhead,   // 注意力头总数
    size_t kv_head, // KV 头数（GQA 场景下 nhead % kv_head == 0）
    size_t head_dim, // 每个头的维度
    size_t v_head_dim, // 每个 V 头的维度
    float scale // 缩放因子（通常是 1/sqrt(head_dim)）
) {
    // ── 索引映射 ──
    size_t i    = blockIdx.x;                     // query 位置
    size_t h    = blockIdx.y;                     // 注意力头索引
    size_t kv_h = h / (nhead / kv_head);          // GQA 映射到 KV head

    // ── Shared Memory 布局 ──
    // 动态: scores[total_len] + warp_buf[WARPS] + s_q[head_dim]
    extern __shared__ char smem_bytes[];
    float* scores   = reinterpret_cast<float*>(smem_bytes);
    float* warp_buf = scores + total_len; // 是 block reduce 的临时空间，WARPS 个 float
    float* s_q      = warp_buf + WARPS;           // Q 行（预转 float）

    // 静态: 用于广播归约结果
    __shared__ float s_max;
    __shared__ float s_sum;

    // ── 预加载 Q 到共享内存（一次加载，多次复用）──
    const T* q_row = q + i * nhead * head_dim + h * head_dim;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_q[d] = to_float(q_row[d]);
    }
    __syncthreads();

    // ---
    //  Phase 1: Q @ K^T  →  scores[t] = dot(Q[i,h,:], K[t,kv_h,:]) * scale
    //           float4 向量化读取 K（每次 128-bit = 4 floats / 8 halfs）
    // ---
    constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T); // F32:4, F16:8
    const size_t head_dim_vec  = head_dim / ELT_PER_VEC;       // 向量化迭代次数
    const size_t head_dim_tail = head_dim_vec * ELT_PER_VEC;   // 标量尾起始位置
    const size_t kv_stride = kv_head * head_dim;               // K 行步长

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        const T* k_row = k + t * kv_stride + kv_h * head_dim;
        float dot = 0.0f;

        // 向量化主循环：float4 批量读取 K
        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        for (size_t vi = 0; vi < head_dim_vec; vi++) {
            float4 kv = k4[vi];
            const T* ke = reinterpret_cast<const T*>(&kv);
            #pragma unroll
            for (size_t e = 0; e < ELT_PER_VEC; e++) {
                dot += s_q[vi * ELT_PER_VEC + e] * to_float(ke[e]);
            }
        }

        // 标量尾：处理 head_dim 不能被 ELT_PER_VEC 整除的剩余部分
        for (size_t d = head_dim_tail; d < head_dim; d++) {
            dot += s_q[d] * to_float(k_row[d]);
        }

        scores[t] = dot * scale;
    }
    __syncthreads();

    // ---
    //  Phase 2: Causal Mask  →  future tokens 设为 -inf
    // ---
    size_t current_pos = total_len - seq_len + i;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        if (t > current_pos) {
            scores[t] = -INFINITY;
        }
    }
    __syncthreads();

    // ---
    //  Phase 3: Safe Softmax
    // ---

    // ── 3a: 求全局最大值 (数值稳定性) ──
    float local_max = -INFINITY;
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
         local_max = fmaxf(local_max, scores[t]);
    }
    local_max = block_reduce_max(local_max, warp_buf);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_val = s_max;

    // ── 3b: exp(score - max) ──
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_val);
    }
    __syncthreads();

    // ── 3c: 求 exp 之和 ──
    float local_sum = 0.0f;
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        local_sum += scores[t];
    }
    local_sum = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // ── 3d: 归一化 ──
    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---
    //  Phase 4: scores @ V  →  输出每个 v_head_dim 元素
    //           V 读取沿 v_head_dim 方向天然 coalesced（相邻线程读相邻 dv）
    // ---
    T* out_row = attn_val + i * nhead * v_head_dim + h * v_head_dim;
    const size_t kv_v_stride = kv_head * v_head_dim;

    for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x) {
        float val = 0.0f;
        for (size_t t = 0; t < total_len; t++) {
            const T* v_row = v + t * kv_v_stride + kv_h * v_head_dim;
            val += scores[t] * to_float(v_row[dv]);
        }
        out_row[dv] = from_float<T>(val);
    }
}

// ============================================================
//  FlashAttention v2 Kernel — Tiled Online Softmax (优化版)
// ============================================================
//
//  将 KV 序列分成 Bc 大小的 tile，逐 tile 处理。
//  维护在线统计量 m (running max) 和 l (running sum)，
//  O_acc 保存未归一化的加权输出，最终除以 l 得到结果。
//
//  更新公式（处理第 j 个 tile）:
//    m_new     = max(m_old, tile_max)
//    correction = exp(m_old - m_new)
//    O_acc     = O_acc * correction + sum_t(exp(s_t - m_new) * V[t])
//    l_new     = l_old * correction + sum_t(exp(s_t - m_new))
//
//  ── 优化要点 ──
//    1. Q 预加载到共享内存（float, 复用多个 tile）
//    2. K/V tile 加载到共享内存 → 避免重复全局内存访问
//    3. tile_scores 存共享内存，供 block-wide softmax reduce
//    4. scores@V 直接从共享内存的 s_v 读取
//
//  ── Shared Memory 布局 ──
//    s_q[head_dim]:            Q 行（float）
//    s_k[Bc * head_dim]:       K tile（T 类型）
//    s_v[Bc * v_head_dim]:     V tile（T 类型）
//    warp_buf[WARPS]:          block reduce 空间
//    tile_scores[Bc]:          当前 tile 的 softmax scores（float）
//    Bc编译时常量
template <typename T, int Bc>
__global__ void flash_attention_kernel(
    T* __restrict__       attn_val,   // [seq_len, nhead, v_head_dim]
    const T* __restrict__ q,          // [seq_len, nhead, head_dim]
    const T* __restrict__ k,          // [total_len, kv_head, head_dim]
    const T* __restrict__ v,          // [total_len, kv_head, v_head_dim]
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
) {
    const size_t i    = blockIdx.x; //query的位置
    const size_t h    = blockIdx.y; // 注意力头的位置
    const size_t kv_h = h / (nhead / kv_head);

    // ── Shared Memory 布局 ──
    extern __shared__ char smem_bytes[];
    float* s_q      = reinterpret_cast<float*>(smem_bytes);
    T*     s_k      = reinterpret_cast<T*>(s_q + head_dim);
    T*     s_v      = reinterpret_cast<T*>(s_k + Bc * head_dim);
    float* warp_buf = reinterpret_cast<float*>(s_v + Bc * v_head_dim);
    float* tile_scores = warp_buf + WARPS;  // [Bc]

    __shared__ float s_tile_max;
    __shared__ float s_tile_sum;

    // ── Causal 位置 ──
    const size_t current_pos = total_len - seq_len + i;
    const size_t kv_len = current_pos + 1;
    const size_t num_tiles = (kv_len + Bc - 1) / Bc;

    const size_t kv_stride   = kv_head * head_dim;
    const size_t kv_v_stride = kv_head * v_head_dim;

    // ── 预加载 Q ──
    const T* q_row = q + i * nhead * head_dim + h * head_dim;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_q[d] = to_float(q_row[d]);
    }
    __syncthreads();

    // ── 每线程的 online softmax 状态 ──
    // MAX_DV = ceil(v_head_dim / blockDim.x) 上界。当前 THREADS=256, head_dim<=256
    // 若修改 THREADS 或支持更大 head_dim，需同步调整
    constexpr int MAX_DV = 4;
    static_assert(MAX_DV >= 1, "MAX_DV must be at least 1");
    float o_acc[MAX_DV];
    for (int r = 0; r < MAX_DV; r++) o_acc[r] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // --- 主循环: 逐 tile 处理 KV ---
    for (size_t tile = 0; tile < num_tiles; tile++) {
        const size_t tile_start = tile * Bc;
        const size_t tile_end   = min(tile_start + (size_t)Bc, kv_len);
        const int    tile_len   = (int)(tile_end - tile_start);

        // ── 加载 K tile 到 shared memory ──
        for (size_t idx = threadIdx.x; idx < (size_t)tile_len * head_dim; idx += blockDim.x) {
            size_t t_local = idx / head_dim;
            size_t d       = idx % head_dim;
            s_k[t_local * head_dim + d] = k[(tile_start + t_local) * kv_stride + kv_h * head_dim + d];
        }

        // ── 加载 V tile 到 shared memory ──
        for (size_t idx = threadIdx.x; idx < (size_t)tile_len * v_head_dim; idx += blockDim.x) {
            size_t t_local = idx / v_head_dim;
            size_t d       = idx % v_head_dim;
            s_v[t_local * v_head_dim + d] = v[(tile_start + t_local) * kv_v_stride + kv_h * v_head_dim + d];
        }
        __syncthreads();

        // ── QK^T: 每线程计算若干个 score ──
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float dot = 0.0f;
            const T* k_local = s_k + t * head_dim;
            for (size_t d = 0; d < head_dim; d++) {
                dot += s_q[d] * to_float(k_local[d]);
            }
            tile_scores[t] = dot * scale;
        }
        __syncthreads();

        // ── Tile max (block reduce) ──
        float local_max = -INFINITY;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            local_max = fmaxf(local_max, tile_scores[t]);
        }
        local_max = block_reduce_max(local_max, warp_buf);
        if (threadIdx.x == 0) s_tile_max = local_max;
        __syncthreads();
        float tile_max = s_tile_max;

        // ── exp(score - tile_max) 写回 tile_scores ──
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            tile_scores[t] = expf(tile_scores[t] - tile_max);
        }
        __syncthreads();

        // ── Tile sum (block reduce) ──
        float local_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            local_sum += tile_scores[t];
        }
        local_sum = block_reduce_sum(local_sum, warp_buf);
        if (threadIdx.x == 0) s_tile_sum = local_sum;
        __syncthreads();
        float tile_sum = s_tile_sum;

        // ── Online softmax 更新 ──
        float new_max = fmaxf(running_max, tile_max);
        float old_correction = expf(running_max - new_max);
        float new_correction = expf(tile_max - new_max);
        float new_sum = running_sum * old_correction + tile_sum * new_correction;

        // ── 更新 O_acc ──
        int dv_idx = 0;
        for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x, dv_idx++) {
            // Rescale 旧累积
            o_acc[dv_idx] *= old_correction;
            // 累加新 tile: sum_t(score[t] * V[t,dv]) * new_correction
            float val = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                val += tile_scores[t] * to_float(s_v[t * v_head_dim + dv]);
            }
            o_acc[dv_idx] += val * new_correction;
        }

        running_max = new_max;
        running_sum = new_sum;
        __syncthreads();
    }

    // ── 最终归一化 ──
    T* out_row = attn_val + i * nhead * v_head_dim + h * v_head_dim;
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

    int dv_idx = 0;
    for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x, dv_idx++) {
        out_row[dv] = from_float<T>(o_acc[dv_idx] * inv_sum);
    }
}

// ============================================================
//  FlashDecoding — Split KV for higher decode parallelism
// ============================================================
//
//  优化目标：Decode 阶段 (seq_len=1) 只有 nhead 个 block，
//           GPU 利用率低。FlashDecoding 沿 KV 长度维度切分，
//           增加 block 数量以充分利用所有 SM。
//
//  两个 kernel:
//    1. flash_decoding_partial: 每个 block 处理一个 KV chunk
//       输出 partial_O (未归一化加权 V), partial_m (局部 max), partial_l (局部 sum)
//    2. flash_decoding_reduce: 跨 splits 合并，用 online softmax 修正
//
//  grid:
//    partial: (num_splits, nhead)
//    reduce:  (nhead,)

constexpr int FD_CHUNK_SIZE = 256;  // KV positions per split block

// ── Static workspace management ──
static float* s_fd_workspace = nullptr;
static size_t s_fd_workspace_bytes = 0;

static void ensure_fd_workspace(size_t num_splits, size_t nhead, size_t v_head_dim) {
    size_t needed = (num_splits * nhead * v_head_dim + 2 * num_splits * nhead) * sizeof(float);
    if (needed > s_fd_workspace_bytes) {
        if (s_fd_workspace) cudaFree(s_fd_workspace);
        cudaMalloc(&s_fd_workspace, needed);
        s_fd_workspace_bytes = needed;
    }
}

void cleanup_fd_workspace() {
    if (s_fd_workspace) {
        cudaFree(s_fd_workspace);
        s_fd_workspace = nullptr;
        s_fd_workspace_bytes = 0;
    }
}

template <typename T>
__global__ void flash_decoding_partial_kernel(
    float* __restrict__     partial_O,    // [num_splits, nhead, v_head_dim]
    float* __restrict__     partial_m,    // [num_splits, nhead]
    float* __restrict__     partial_l,    // [num_splits, nhead]
    const T* __restrict__   q,            // [1, nhead, head_dim]
    const T* __restrict__   k,            // [total_len, kv_head, head_dim]
    const T* __restrict__   v,            // [total_len, kv_head, v_head_dim]
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
) {
    const size_t split_id = blockIdx.x;
    const size_t h        = blockIdx.y;
    const size_t kv_h     = h / (nhead / kv_head);

    const size_t kv_start = split_id * FD_CHUNK_SIZE;
    const size_t kv_end   = min(kv_start + (size_t)FD_CHUNK_SIZE, total_len);
    const int    kv_len   = (int)(kv_end - kv_start);

    if (kv_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[split_id * nhead + h] = -INFINITY;
            partial_l[split_id * nhead + h] = 0.0f;
        }
        for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x)
            partial_O[(split_id * nhead + h) * v_head_dim + dv] = 0.0f;
        return;
    }

    // Shared: scores[FD_CHUNK_SIZE] + warp_buf[WARPS] + s_q[head_dim]
    extern __shared__ char smem_bytes[];
    float* scores   = reinterpret_cast<float*>(smem_bytes);
    float* warp_buf = scores + FD_CHUNK_SIZE;
    float* s_q      = warp_buf + WARPS;

    __shared__ float s_max, s_sum;

    // Load Q
    const T* q_row = q + h * head_dim;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x)
        s_q[d] = to_float(q_row[d]);
    __syncthreads();

    // Phase 1: QK^T (float4 vectorized)
    constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T);
    const size_t head_dim_vec  = head_dim / ELT_PER_VEC;
    const size_t head_dim_tail = head_dim_vec * ELT_PER_VEC;
    const size_t kv_stride     = kv_head * head_dim;

    for (int t = threadIdx.x; t < kv_len; t += blockDim.x) {
        const T* k_row = k + (kv_start + t) * kv_stride + kv_h * head_dim;
        float dot = 0.0f;

        const float4* k4 = reinterpret_cast<const float4*>(k_row);
        for (size_t vi = 0; vi < head_dim_vec; vi++) {
            float4 kv4 = k4[vi];
            const T* ke = reinterpret_cast<const T*>(&kv4);
            #pragma unroll
            for (size_t e = 0; e < ELT_PER_VEC; e++)
                dot += s_q[vi * ELT_PER_VEC + e] * to_float(ke[e]);
        }
        for (size_t d = head_dim_tail; d < head_dim; d++)
            dot += s_q[d] * to_float(k_row[d]);

        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: No causal mask for decode (query at end, all KV valid)

    // Phase 3a: Max
    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    local_max = block_reduce_max(local_max, warp_buf);
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_val = s_max;

    // Phase 3b: exp(score - max)
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x)
        scores[t] = expf(scores[t] - max_val);
    __syncthreads();

    // Phase 3c: Sum
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x)
        local_sum += scores[t];
    local_sum = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();

    // Phase 4: Unnormalized scores @ V
    const size_t kv_v_stride = kv_head * v_head_dim;
    float* out_partial = partial_O + (split_id * nhead + h) * v_head_dim;

    for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            const T* v_row = v + (kv_start + t) * kv_v_stride + kv_h * v_head_dim;
            val += scores[t] * to_float(v_row[dv]);
        }
        out_partial[dv] = val;
    }

    // Store local max and sum
    if (threadIdx.x == 0) {
        partial_m[split_id * nhead + h] = max_val;
        partial_l[split_id * nhead + h] = s_sum;
    }
}

template <typename T>
__global__ void flash_decoding_reduce_kernel(
    T* __restrict__           attn_val,    // [1, nhead, v_head_dim]
    const float* __restrict__ partial_O,   // [num_splits, nhead, v_head_dim]
    const float* __restrict__ partial_m,   // [num_splits, nhead]
    const float* __restrict__ partial_l,   // [num_splits, nhead]
    size_t num_splits,
    size_t nhead,
    size_t v_head_dim
) {
    const size_t h = blockIdx.x;

    extern __shared__ char smem_bytes[];
    float* warp_buf = reinterpret_cast<float*>(smem_bytes);

    __shared__ float s_global_max;
    __shared__ float s_global_sum;

    // Global max across splits
    float local_max = -INFINITY;
    for (size_t s = threadIdx.x; s < num_splits; s += blockDim.x)
        local_max = fmaxf(local_max, partial_m[s * nhead + h]);
    local_max = block_reduce_max(local_max, warp_buf);
    if (threadIdx.x == 0) s_global_max = local_max;
    __syncthreads();
    float global_max = s_global_max;

    // Rescaled sum
    float local_sum = 0.0f;
    for (size_t s = threadIdx.x; s < num_splits; s += blockDim.x)
        local_sum += expf(partial_m[s * nhead + h] - global_max) * partial_l[s * nhead + h];
    local_sum = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) s_global_sum = local_sum;
    __syncthreads();
    float inv_sum = (s_global_sum > 0.0f) ? (1.0f / s_global_sum) : 0.0f;

    // Merge partial_O with rescaling
    T* out = attn_val + h * v_head_dim;
    for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x) {
        float val = 0.0f;
        for (size_t s = 0; s < num_splits; s++) {
            float rescale = expf(partial_m[s * nhead + h] - global_max);
            val += rescale * partial_O[(s * nhead + h) * v_head_dim + dv];
        }
        out[dv] = from_float<T>(val * inv_sum);
    }
}

} // anonymous namespace

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::nvidia {

void self_attention(
    std::byte* attn_val_ptr,
    llaisysDataType_t dtype,
    const std::byte* q,
    const std::byte* k,
    const std::byte* v,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
) {
    constexpr int threads = THREADS;
    constexpr int MAX_DV = 4;

    // 安全检查: v_head_dim / threads 必须 <= MAX_DV, 否则 o_acc 越界
    if ((v_head_dim + threads - 1) / threads > MAX_DV) {
        throw std::runtime_error(
            "self_attention: v_head_dim / THREADS > MAX_DV (" +
            std::to_string(v_head_dim) + "/" + std::to_string(threads) +
            " > " + std::to_string(MAX_DV) + "). Increase MAX_DV.");
    }

    // ---
    //  FlashDecoding: decode (seq_len=1) + long enough KV
    // ---
    if (seq_len == 1 && total_len > (size_t)FD_CHUNK_SIZE) {
        size_t num_splits = (total_len + FD_CHUNK_SIZE - 1) / FD_CHUNK_SIZE;
        ensure_fd_workspace(num_splits, nhead, v_head_dim);

        float* partial_O = s_fd_workspace;
        float* partial_m = s_fd_workspace + num_splits * nhead * v_head_dim;
        float* partial_l = partial_m + num_splits * nhead;

        dim3 partial_grid(static_cast<unsigned>(num_splits), static_cast<unsigned>(nhead));
        size_t partial_smem = (FD_CHUNK_SIZE + WARPS + head_dim) * sizeof(float);

        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            flash_decoding_partial_kernel<float><<<partial_grid, threads, partial_smem>>>(
                partial_O, partial_m, partial_l,
                reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
                reinterpret_cast<const float*>(v),
                total_len, nhead, kv_head, head_dim, v_head_dim, scale);
            break;
        case LLAISYS_DTYPE_F16:
            flash_decoding_partial_kernel<__half><<<partial_grid, threads, partial_smem>>>(
                partial_O, partial_m, partial_l,
                reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
                reinterpret_cast<const __half*>(v),
                total_len, nhead, kv_head, head_dim, v_head_dim, scale);
            break;
        case LLAISYS_DTYPE_BF16:
            flash_decoding_partial_kernel<__nv_bfloat16><<<partial_grid, threads, partial_smem>>>(
                partial_O, partial_m, partial_l,
                reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const __nv_bfloat16*>(v),
                total_len, nhead, kv_head, head_dim, v_head_dim, scale);
            break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        checkCuda(cudaGetLastError(), "flash_decoding_partial_kernel launch failed");

        dim3 reduce_grid(static_cast<unsigned>(nhead));
        size_t reduce_smem = WARPS * sizeof(float);

        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            flash_decoding_reduce_kernel<float><<<reduce_grid, threads, reduce_smem>>>(
                reinterpret_cast<float*>(attn_val_ptr), partial_O, partial_m, partial_l,
                num_splits, nhead, v_head_dim);
            break;
        case LLAISYS_DTYPE_F16:
            flash_decoding_reduce_kernel<__half><<<reduce_grid, threads, reduce_smem>>>(
                reinterpret_cast<__half*>(attn_val_ptr), partial_O, partial_m, partial_l,
                num_splits, nhead, v_head_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            flash_decoding_reduce_kernel<__nv_bfloat16><<<reduce_grid, threads, reduce_smem>>>(
                reinterpret_cast<__nv_bfloat16*>(attn_val_ptr), partial_O, partial_m, partial_l,
                num_splits, nhead, v_head_dim);
            break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        checkCuda(cudaGetLastError(), "flash_decoding_reduce_kernel launch failed");
        return;
    }

    // ---
    //  Original paths: Naive Fused / FlashAttention v2
    // ---
    dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(nhead));

    // 策略选择: scores[total_len] 能装进 48KB smem → naive, 否则 flash
    size_t naive_smem = (total_len + WARPS + head_dim) * sizeof(float);
    constexpr size_t SMEM_LIMIT = 48 * 1024;  // 48KB

    bool use_flash = (naive_smem > SMEM_LIMIT);

    if (use_flash) {
        // ── FlashAttention 路径 ──
        // Bc 选择: F32 用 Bc=32（更大的元素），F16/BF16 用 Bc=64
        // 共享内存: s_q[head_dim]*4 + s_k[Bc*head_dim]*sizeof(T) + s_v[Bc*v_head_dim]*sizeof(T) + warp_buf[WARPS]*4 +  tile_scores[Bc]*4
        auto launch_flash = [&](auto* dummy_t, auto dtype_tag) {
            using DT = std::remove_pointer_t<decltype(dummy_t)>;
            constexpr int Bc_val = (sizeof(DT) <= 2) ? 64 : 32;
            size_t flash_smem = head_dim * sizeof(float)                // s_q
                              + Bc_val * head_dim * sizeof(DT)          // s_k
                              + Bc_val * v_head_dim * sizeof(DT)        // s_v
                              + (WARPS + Bc_val) * sizeof(float);       // warp_buf + tile_scores
            flash_attention_kernel<DT, Bc_val><<<grid, threads, flash_smem>>>(
                reinterpret_cast<DT*>(attn_val_ptr),
                reinterpret_cast<const DT*>(q),
                reinterpret_cast<const DT*>(k),
                reinterpret_cast<const DT*>(v),
                seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
            );
        };

        switch (dtype) {
        case LLAISYS_DTYPE_F32:  { float* p = nullptr; launch_flash(p, 0); break; }
        case LLAISYS_DTYPE_F16:  { __half* p = nullptr; launch_flash(p, 0); break; }
        case LLAISYS_DTYPE_BF16: { __nv_bfloat16* p = nullptr; launch_flash(p, 0); break; }
        default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        checkCuda(cudaGetLastError(), "flash_attention_kernel launch failed");
    } else {
        // ── Naive Fused 路径（短序列快速路径）──
        size_t smem_size = naive_smem;

        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            self_attention_fused<float><<<grid, threads, smem_size>>>(
                reinterpret_cast<float*>(attn_val_ptr),
                reinterpret_cast<const float*>(q),
                reinterpret_cast<const float*>(k),
                reinterpret_cast<const float*>(v),
                seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
            );
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_fused<__half><<<grid, threads, smem_size>>>(
                reinterpret_cast<__half*>(attn_val_ptr),
                reinterpret_cast<const __half*>(q),
                reinterpret_cast<const __half*>(k),
                reinterpret_cast<const __half*>(v),
                seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
            );
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_fused<__nv_bfloat16><<<grid, threads, smem_size>>>(
                reinterpret_cast<__nv_bfloat16*>(attn_val_ptr),
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const __nv_bfloat16*>(v),
                seq_len, total_len, nhead, kv_head, head_dim, v_head_dim, scale
            );
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
        checkCuda(cudaGetLastError(), "self_attention_fused kernel launch failed");
    }
}

void cleanup_self_attention_workspace() {
    cleanup_fd_workspace();
}

} // namespace llaisys::ops::nvidia
