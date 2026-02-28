# Self-Attention 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{causal\_mask}\right) \cdot V$

**这是 Transformer 中最复杂的算子**，包含四个子操作：
1. **QK^T**: 矩阵乘法（计算注意力分数）
2. **Causal Mask**: 遮盖未来 token
3. **Softmax**: 归一化为概率分布
4. **Scores × V**: 加权求和（注意力聚合）

**张量形状**:
- Q: `[seq_len, nhead, head_dim]`
- K: `[total_len, kv_head, head_dim]`
- V: `[total_len, kv_head, v_head_dim]`
- 输出: `[seq_len, nhead, v_head_dim]`

**GQA 支持**: `nhead % kv_head == 0`，多个 Q 头共享一组 KV 头。

---

## 2. 为什么用融合 Kernel 而非分步调用？

### 分步方式（朴素）

```
// Step 1: scores = Q @ K^T                               → 需要 [seq_len, nhead, total_len] 中间张量
// Step 2: scores = causal_mask(scores)                    → 读写中间张量
// Step 3: scores = softmax(scores)                        → 再次读写
// Step 4: out = scores @ V                                → 读中间张量 + V
```

**问题**: `scores[seq_len, nhead, total_len]` 可能非常大！
- 比如 seq_len=2048, nhead=32, total_len=2048 → 128M floats = 512MB
- 每步都要全局内存读写 → 写 512MB × 4 步 = 2GB 数据搬运

### 融合方式（我们的实现）

**一个 Kernel 完成全部四步**，中间结果只存在 **Shared Memory** 中：
- `scores[total_len]` 放在 shared memory（每 block 独立）
- 每个 block 处理一个 `(query_pos, head)` 组合
- 总共只需要 `total_len × 4B` 的 shared memory

**省了多少？**
```
分步: 4 × 全局内存 @ 512MB = 2GB 内存搬运
融合: 0 次额外全局内存 （scores 在 shared memory 内完成生命周期）
```

---

## 3. 线程映射

```cuda
dim3 grid(seq_len, nhead);    // 每个 block = 一个 (query_pos, head)
constexpr int THREADS = 256;  // block 内 256 线程协作
```

**为什么这样映射？**
1. 每个 `(i, h)` 的计算完全独立 → 天然并行
2. 每个 block 内需要做归约（softmax），所以归约维度（total_len）分配给同一 block 的线程
3. block 大小 256 = 8 个 warp，归约效率高

---

## 4. 四阶段 Kernel 详解

### Phase 1: Q @ K^T → scores

```cuda
// 每线程负责若干个 t（Grid-Stride over total_len）
for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
    const T* k_row = k + t * kv_head * head_dim + kv_h * head_dim;
    float dot = 0.0f;
    for (size_t d = 0; d < head_dim; d++) {
        dot += to_float(q_row[d]) * to_float(k_row[d]);  // 逐元素点积
    }
    scores[t] = dot * scale;  // 写入 shared memory
}
__syncthreads();
```

**访存分析**:
- Q: 每线程读同一行（广播，L1 cache 友好）
- K: 不同线程读不同的 t → 跨行访问（但每行自身连续）
- 计算密集度: `head_dim` 次乘加 / 行

### Phase 2: Causal Mask

```cuda
size_t current_pos = total_len - seq_len + i;
for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
    if (t > current_pos) scores[t] = -INFINITY;
}
__syncthreads();
```

**因果掩码**: 位置 i 只能看到 ≤ current_pos 的 token（不能看未来）。
- `current_pos = total_len - seq_len + i`：考虑了 KV cache 的场景

### Phase 3: Safe Softmax

这是最复杂的部分，需要 **3 次 block-wide 同步**：

```
Step 3a: max = block_reduce_max(scores)        → 数值稳定性
Step 3b: scores[t] = exp(scores[t] - max)      → 指数化
Step 3c: sum = block_reduce_sum(scores)         → 求和
Step 3d: scores[t] /= sum                      → 归一化
```

**Safe Softmax 的数值稳定性**:
```
普通: softmax(x_i) = exp(x_i) / Σexp(x_j)     → 如果 x_i 很大，exp 溢出！
Safe: softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)  → 最大值减去后 ≤ 0，不会溢出
```

**Block Reduce 实现（复用 argmax/rms_norm 的模式）**:

```cuda
__device__ float block_reduce_max(float val, float* warp_buf) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    // Layer 1: warp 内 shuffle reduce
    for (int delta = 16; delta >= 1; delta >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));

    // Layer 2: warp 间通过 shared memory
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();

    // Layer 3: warp 0 最终归约
    val = (threadIdx.x < WARPS) ? warp_buf[threadIdx.x] : -INFINITY;
    if (wid == 0) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta));
    }
    return val;
}
```

### Phase 4: Scores × V → 输出

```cuda
// 线程映射切换到 v_head_dim 维度
for (size_t dv = threadIdx.x; dv < v_head_dim; dv += blockDim.x) {
    float val = 0.0f;
    for (size_t t = 0; t < total_len; t++) {
        val += scores[t] * to_float(v_row[dv]);  // scores 从 shared memory 读
    }
    out_row[dv] = from_float<T>(val);
}
```

**注意线程映射变化**:
- Phase 1-3：线程映射到 `total_len`（处理不同的 K token）
- Phase 4：线程映射到 `v_head_dim`（处理输出的不同维度）

---

## 5. Shared Memory 布局

```cuda
extern __shared__ char smem_bytes[];
float* scores   = (float*)(smem_bytes);              // [total_len]
float* warp_buf = scores + total_len;                 // [WARPS=8]

__shared__ float s_max;   // 广播 softmax max
__shared__ float s_sum;   // 广播 softmax sum
```

**动态 Shared Memory 大小**:
```cuda
size_t smem_size = (total_len + WARPS) * sizeof(float);
kernel<<<grid, threads, smem_size>>>(/* ... */);
```

**限制**: RTX 4070 Laptop 每 SM 有 48KB/100KB shared memory（取决于配置），能支持 `total_len ≈ 12K`。更长序列需要分块（FlashAttention 的做法）。

---

## 6. GQA（Grouped Query Attention）

```cuda
size_t kv_h = h / (nhead / kv_head);  // 多个 Q 头映射到同一 KV 头
```

例如 nhead=32, kv_head=4：
- Q head 0-7 → KV head 0
- Q head 8-15 → KV head 1
- ...

**好处**: KV cache 大小减少 `nhead/kv_head` 倍，推理时内存节省显著。

---

## 7. 与 FlashAttention 的对比

| 方面 | 我们的实现 | FlashAttention |
|------|-----------|---------------|
| 融合度 | 完全融合 | 完全融合 |
| softmax 策略 | 全量存 shared memory | **分块 Online Softmax** |
| 序列长度限制 | 受 shared memory 限制 | 无限制（分块处理） |
| IO 复杂度 | O(N²) | O(N² / SRAM_SIZE) |
| 实现复杂度 | ~100 行 | ~1000 行 |
| 适用场景 | 短序列（推理 decode 阶段） | 长序列训练/prefill |

**我们的实现适合推理**：decode 阶段 seq_len=1，total_len 通常几百到几千，完全放得下 shared memory。

---

## 8. 性能分析与优化

### ncu 基线数据（Decode: qlen=1, kvlen=512, nhead=12, kv_head=2, head_dim=128, F32）

| 指标 | 数值 |
|------|------|
| Grid | 12 (1×12) |
| Block | 256 |
| Duration | 64.32µs |
| DRAM Throughput | 6.14% |
| Memory BW | 16.67 GB/s |
| Compute (SM) | 3.50% |
| Achieved Occupancy | 13.10% |
| Excess Sectors | **76%** (688128 / 909504) |
| L1TEX Stall | **86.1%** |

**三大瓶颈**：

1. **Grid 太小**: qlen=1 × nhead=12 = 12 blocks，GPU 有 36 SMs → 只用 1/3，Waves=0.07
2. **K 访问严重 uncoalesced**: Phase 1 中相邻线程访问不同 t 的 K 行，步长 = `kv_head × head_dim × sizeof(T)` = 1024B → 每 sector 只有 7/32 bytes 有效
3. **低 Occupancy**: 只有 13.10%，无法隐藏内存延迟

### 优化实施

#### 优化 1: Q 预加载到 Shared Memory  
Q 行只有 128 floats = 512B，但在 Phase 1 中被每个线程的每次 d 循环重复读取。预加载到 smem 后：
- 避免 L1 cache 压力（L1 被 K 读取占满时 Q 可能被 evict）
- 读 smem 延迟 ~20 cycles vs L1 ~100+ cycles

```cuda
// 预加载 Q 到共享内存
float* s_q = warp_buf + WARPS;
for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x)
    s_q[d] = to_float(q_row[d]);
__syncthreads();
```

共享内存新增 head_dim × 4 = 512B，总 smem = (512 + 8 + 128) × 4 = 2.6KB，远低于限制。

#### 优化 2: float4 向量化 K 读取（Phase 1）

```cuda
constexpr size_t ELT_PER_VEC = sizeof(float4) / sizeof(T); // F32:4, F16:8
const size_t head_dim_vec = head_dim / ELT_PER_VEC;

for (size_t t = threadIdx.x; t < total_len; t += blockDim.x) {
    const float4* k4 = reinterpret_cast<const float4*>(k_row);
    float dot = 0.0f;
    for (size_t vi = 0; vi < head_dim_vec; vi++) {
        float4 kv = k4[vi];  // 128-bit 批量读取
        const T* ke = reinterpret_cast<const T*>(&kv);
        #pragma unroll
        for (size_t e = 0; e < ELT_PER_VEC; e++)
            dot += s_q[vi * ELT_PER_VEC + e] * to_float(ke[e]);
    }
    // scalar tail for non-aligned head_dim...
}
```

**效果**: 
- F32: 128 scalar loads → 32 float4 loads（4x 减少）
- F16: 128 scalar loads → 16 float4 loads（8x 减少）
- 虽然线程间 K 仍然不 coalesced，但每线程的内存事务数大幅减少

### ncu 优化后数据（F32）

| 指标 | 基线 | 优化 | 变化 |
|------|------|------|------|
| Duration | 64.32µs | **42.69µs** | **-34%** |
| Memory BW | 16.67 GB/s | 25.35 GB/s | **+52%** |
| DRAM Throughput | 6.14% | 9.34% | +52% |
| Compute (SM) | 3.50% | 4.12% | +18% |
| Excess Sectors | 76% | **33%** | **-57%** |
| F16 Duration | — | 37.41µs | — |

### 残留瓶颈与进一步优化方向

**Grid=12 是 decode 场景的固有限制**（qlen=1 × nhead=12），无法通过简单方法增加。两个高级优化方向：

1. **Split-KV 多 Block 协作**：将 total_len 分给多个 block，每 block 处理部分 t，再用全局归约合并 softmax。需要 Online Softmax 技巧（FlashAttention 的核心）。

2. **K 转置缓存**：在 KV cache 中将 K 转置为 `[kv_head, total_len, head_dim]` 布局，使 Phase 1 中线程间 K 访问变为 coalesced。需要额外的转置 kernel，但在长序列下收益巨大。

这些属于 FlashAttention 级别的优化，超出本项目教育范围。

---

## 9. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| Kernel Fusion | 多算子合并为一个 kernel，避免中间结果写全局内存 |
| Dynamic Shared Memory | `<<<grid, block, smem_size>>>` 动态分配 |
| 多阶段 Kernel | 同一 kernel 内通过 `__syncthreads()` 分隔不同计算阶段 |
| 线程映射切换 | 不同 Phase 用不同的线程→数据映射（Phase 1-3 → total_len，Phase 4 → v_head_dim） |
| Safe Softmax | 先减最大值再 exp，防止数值溢出 |
| Block Reduce (max + sum) | warp shuffle + shared memory 的标准两层归约 |
| GQA | 多 Q 头共享 KV 头，减少 KV cache 内存 |
| `extern __shared__` | 声明动态大小的 shared memory |
| float4 向量化 | 128-bit 批量数据读取，减少指令量和内存事务数 |
| 数据预加载到 smem | Q 等被频繁重用的数据加载到共享内存，减少全局内存/L1 压力 |
| Coalescing 分析 | 相邻线程访存模式决定带宽利用率 — 步长越大越浪费 |
| Scalar Tail | 向量化路径无法处理的尾部元素用标量 kernel 兜底 |
| Decode 瓶颈 | qlen=1 时 Grid 极小，SM 利用率低，是 Attention 优化的核心挑战 |
