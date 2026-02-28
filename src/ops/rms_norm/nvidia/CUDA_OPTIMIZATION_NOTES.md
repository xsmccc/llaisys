# RMSNorm 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: $Y_i = W_i \cdot X_i \cdot \frac{1}{\sqrt{\frac{1}{d}\sum_{j=0}^{d-1} X_j^2 + \epsilon}}$

分解为两步：
1. **归约（Reduction）**: 计算 `sum_sq = Σ X_j²`，然后 `inv_rms = rsqrt(sum_sq/d + eps)`
2. **逐元素（Elementwise）**: `out[j] = weight[j] * in[j] * inv_rms`

**类型**: 归约 + elementwise 混合型。这是 CUDA 编程中非常经典的模式。

**张量形状**:
- in: `[rows, cols]` — 输入
- weight: `[cols]` — 可学习的缩放参数
- out: `[rows, cols]` — 输出

---

## 2. 与 LayerNorm 的区别

| | LayerNorm | RMSNorm |
|---|----------|---------|
| 步骤 | 减均值 → 除标准差 → 缩放 + 偏移 | 除 RMS → 缩放 |
| 归约次数 | 2 次（求均值 + 求方差） | 1 次（求平方和） |
| 参数 | weight + bias | 只有 weight |
| 计算量 | 更多 | 更少 |

RMSNorm 是 LLaMA/Qwen 等模型默认使用的归一化，比 LayerNorm 快且效果相近。

---

## 3. 线程映射策略

```
grid:  (rows, 1, 1)   — 每个 block 处理一行
block: (256, 1, 1)     — 256 线程分工处理列
```

**为什么一个 block 处理一行？**
1. 归约需要 block 内同步（`__syncthreads`），所以归约范围 = block 范围
2. 每行独立归约，天然映射到一个 block
3. cols 通常是 1536~4096，256 线程通过循环覆盖

---

## 4. Block 归约实现：Warp Shuffle + Shared Memory

```cuda
__device__ float warp_reduce_sum(float val) {
    // 5 轮 shuffle，将 warp 内 32 个值求和到 lane 0
    for (int delta = 16; delta >= 1; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;  // lane 0 持有结果
}

__device__ float block_reduce_sum(float val) {
    __shared__ float s_partial[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // Layer 1: warp 内归约
    val = warp_reduce_sum(val);

    // Layer 2: warp 间通过 shared memory 传递
    if (lane == 0) s_partial[warp_id] = val;
    __syncthreads();

    // Layer 3: warp 0 做最终归约
    val = (lane < num_warps) ? s_partial[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    return val; // 仅 thread 0 持有正确总和
}
```

**与 Argmax 的 block_reduce_max 对比**:
- 结构完全相同（同构归约）
- 唯一区别：`max` 换成 `+`，初始值从 `-FLT_MAX` 换成 `0.0f`

---

## 5. Kernel 实现：两阶段

```cuda
template <typename T>
__global__ void rms_norm_kernel(
    T* out, const T* in, const T* weight,
    size_t rows, size_t cols, float eps
) {
    size_t row = blockIdx.x;
    const T* row_in = in + row * cols;
    T* row_out = out + row * cols;

    // ═══ Phase 1: 归约求平方和 ═══
    float sum_sq = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = to_float(row_in[j]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    // 广播 inv_rms（thread 0 → shared memory → 所有线程）
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // ═══ Phase 2: 逐元素归一化 ═══
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = to_float(row_in[j]);
        float w = to_float(weight[j]);
        row_out[j] = from_float<T>(v * w * inv_rms);
    }
}
```

**关键点**:
1. `rsqrtf` = 1/sqrt(x) — CUDA 硬件在 SFU 上 1 个周期完成，比 `1.0f/sqrtf(x)` 快
2. 广播用 shared memory 的 `s_inv_rms`，避免重复计算
3. F16/BF16 在计算时上转为 F32，结果回转，保持精度

---

## 6. 两次全局内存遍历

注意 kernel 中 `row_in[j]` 被读了 **两次**：
1. Phase 1: 读取计算平方和
2. Phase 2: 读取做归一化

**这是否浪费？**
- 如果 cols 小（< 4096），数据很可能驻留在 L1 cache 中
- 如果 cols 很大，可以考虑用 shared memory 缓存一行数据（但 48KB shared memory 最多放 ~12K floats）
- 对于 LLM 的典型维度（1536~4096），L1 cache 足够

---

## 7. ncu Profiling 与 float4 向量化优化

**测试环境**: RTX 4070 Laptop (36 SMs, sm_89), CUDA 12.6
**测试参数**: rows=512, cols=4096（模型级参数）

### 7.1 基线数据（标量访问）

| 指标 | F32 | F16 |
|------|-----|-----|
| DRAM Throughput | 66.59% | 49.10% |
| Duration | 48.00µs | 31.55µs |
| Memory BW | 180.94 GB/s | 133.35 GB/s |
| L2 Hit Rate | 64.27% | 54.44% |
| Occupancy | 86.86% | 89.75% |

**瓶颈分析**:
- 标量读写导致大量上字节内存事务（LD.32 或 LD.16），无法充分利用 128B 缓存行
- F16 更差：每请求只搬 2B，浪费率更高

### 7.2 优化：float4 向量化（LD.128 / ST.128）

- F32: `float4 = 4 × float = 16B`，vec_cols = cols / 4
- F16/BF16: `float4 = 8 × half = 16B`，vec_cols = cols / 8
- Phase 1（求平方和）和 Phase 2（归一化）都改为 float4 读写
- 带标量尾部处理，兼容任意 cols 值

### 7.3 优化后数据

| 指标 | F32 标量 | F32 vec4 | F16 标量 | F16 vec4 |
|------|----------|----------|----------|----------|
| DRAM | 66.59% | 79.84% | 49.10% | 75.45% |
| Duration | 48.00µs | 38.78µs | 31.55µs | 20.58µs |
| Memory BW | 180.94 | 216.86 | 133.35 | 204.56 |
| L2 Hit Rate | 64.27% | 64.48% | 54.44% | 58.26% |
| Occupancy | 86.86% | 91.76% | 89.75% | 89.44% |

**提升**:
- F32: Duration 48→38.78µs (**-19%**)，DRAM 67→80%
- F16: Duration 31.55→20.58µs (**-35%**)，DRAM 49→75%
- F16 提升更大，因为标量 LD.16 浪费率更高，vec4 改善更显著

### 7.4 为什么 DRAM 没到 90%+？

1. **两次全局内存遍历**: Phase 1 和 Phase 2 各读一次 in，虽然第二次大概率 L1 cache hit
   （L2 Hit Rate ~64% 验证了这点），但仍有 overhead
2. **归约同步开销**: `block_reduce_sum` 需要 `__syncthreads()` + warp shuffle，
   所有线程必须等待最慢的完成
3. **Compute 占比不为零**: RMSNorm 有 FMA（v*v, v*w*inv_rms）、rsqrtf 等计算，
   不是纯搬运如 Embedding

---

## 8. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| 两阶段 Kernel | 先归约再 elementwise，靠 `__syncthreads()` 分隔 |
| `rsqrtf` | 1/sqrt(x) 快速指令，映射到 SFU 硬件 |
| Shared Memory 广播 | thread 0 写入标量 → `__syncthreads()` → 所有线程读取 |
| 归约 + elementwise 融合 | 典型的"先缩后展"模式：N→1→N |
| L1 Cache 重利用 | 两次读同一数据，如果 working set 小于 L1 容量则几乎零开销 |
| **float4 向量化** | LD.128/ST.128 减少内存事务数 4x(F32)/8x(F16) |
| **模板 constexpr** | `sizeof(float4)/sizeof(T)` 编译期确定向量宽度 |
| **`#pragma unroll`** | 展开小循环，消除循环控制开销 |
