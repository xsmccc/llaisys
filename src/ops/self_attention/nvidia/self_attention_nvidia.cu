/**
 * @file self_attention_nvidia.cu
 * @brief Causal Self-Attention 算子的 CUDA 实现（融合 kernel）
 *
 * ── 算子公式 ────────────────────────────────────────────
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
 * ── 融合策略 ────────────────────────────────────────────
 *   将整个 self-attention 融合为一个 kernel，避免中间结果来回全局内存：
 *     Phase 1: Q @ K^T — 每线程负责若干 t，逐元素点积后写入 shared memory
 *     Phase 2: Causal mask — 将 t > current_pos 的分数设为 -inf
 *     Phase 3: Safe softmax — block reduce 求 max → exp → reduce 求 sum → 归一化
 *     Phase 4: scores @ V — 线程映射到 v_head_dim，读共享内存中的 scores 加权求和
 *
 * ── 线程映射 ────────────────────────────────────────────
 *   grid:  (seq_len, nhead)    — 每个 block 处理一个 (query_pos, head) 组合
 *   block: (256, 1, 1)         — block 内 256 线程协作
 *
 * ── Shared Memory 布局 ──────────────────────────────────
 *   dynamic: scores[total_len] + warp_buf[WARPS]
 *   static:  s_max, s_sum (用于广播归约结果)
 *
 * ── 所有计算统一使用 F32 ──────────────────────────────────
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
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
) {
    // ── 索引映射 ──
    size_t i    = blockIdx.x;                     // query 位置
    size_t h    = blockIdx.y;                     // 注意力头索引
    size_t kv_h = h / (nhead / kv_head);          // GQA 映射到 KV head

    // ── Shared Memory 布局 ──
    // 动态: scores[total_len] + warp_buf[WARPS] + s_q[head_dim]
    extern __shared__ char smem_bytes[];
    float* scores   = reinterpret_cast<float*>(smem_bytes);
    float* warp_buf = scores + total_len;
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

    // ════════════════════════════════════════════════════════
    //  Phase 1: Q @ K^T  →  scores[t] = dot(Q[i,h,:], K[t,kv_h,:]) * scale
    //           float4 向量化读取 K（每次 128-bit = 4 floats / 8 halfs）
    // ════════════════════════════════════════════════════════
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

    // ════════════════════════════════════════════════════════
    //  Phase 2: Causal Mask  →  future tokens 设为 -inf
    // ════════════════════════════════════════════════════════
    size_t current_pos = total_len - seq_len + i;

    for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
        if (t > current_pos) {
            scores[t] = -INFINITY;
        }
    }
    __syncthreads();

    // ════════════════════════════════════════════════════════
    //  Phase 3: Safe Softmax
    // ════════════════════════════════════════════════════════

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

    // ════════════════════════════════════════════════════════
    //  Phase 4: scores @ V  →  输出每个 v_head_dim 元素
    //           V 读取沿 v_head_dim 方向天然 coalesced（相邻线程读相邻 dv）
    // ════════════════════════════════════════════════════════
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
    dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(nhead));
    constexpr int threads = THREADS;
    // 动态共享内存: scores[total_len] + warp_buf[WARPS] + s_q[head_dim]
    size_t smem_size = (total_len + WARPS + head_dim) * sizeof(float);

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

} // namespace llaisys::ops::nvidia
