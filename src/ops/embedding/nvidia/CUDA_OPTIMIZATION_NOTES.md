# Embedding 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: `out[i, :] = weight[indices[i], :]`

**功能**: 从 Embedding 表（二维矩阵）中按索引查找对应行，拼成输出张量。在 Transformer 中用于将 token ID 转换为向量表示。

**类型**: 纯访存密集型（Memory Bound）— 没有任何计算，只是内存拷贝。

**张量形状**:
- weight: `[vocab_size, embedding_dim]` — 词嵌入表
- indices: `[num_indices]` — token ID 列表
- out: `[num_indices, embedding_dim]` — 输出

---

## 2. 访存模式分析

| 数据 | 访存模式 | 说明 |
|------|---------|------|
| indices | 连续读，一次性 | 顺序遍历 |
| weight | **不规则读**（gather） | index 决定行号，可能跳跃 |
| out | 连续写 | 按 index 顺序依次写入 |

**关键挑战**: weight 的访问是不规则的（取决于 indices 的值），可能导致 L2 cache miss。但 embedding_dim 通常很大（1536, 4096），每行数据足以让内存事务合并。

---

## 3. 实现策略：一个 Block 处理一行

```
grid:  (num_indices, 1, 1)  — 每个 block 负责复制一行
block: (256, 1, 1)          — 256 线程协作复制
```

**为什么这样映射？**
1. 每行长度 = embedding_dim（通常几千），一个 block 的 256 线程正好能高效处理
2. 行与行之间完全独立，天然并行
3. 同一 block 内的线程访问的是连续内存（同一行），利于合并访存

---

## 4. 向量化实现

### F32: float4（每线程 4 个 float = 16B）

```cuda
template <typename IndexT>
__global__ void embedding_kernel_f32(
    float* out, const IndexT* indices, const float* weight,
    size_t num_indices, size_t embedding_dim
) {
    size_t row = blockIdx.x;
    if (row >= num_indices) return;

    IndexT idx = indices[row];           // 读取当前行的 index
    const float* src = weight + idx * embedding_dim;  // weight 中对应行
    float* dst = out + row * embedding_dim;           // 输出中对应行

    // float4 向量化复制
    size_t vec_dim = embedding_dim / 4;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
        dst4[i] = src4[i];  // 16B 一次加载 + 存储
    }

    // 尾部标量处理
    size_t tail_start = vec_dim * 4;
    for (size_t i = tail_start + threadIdx.x; i < embedding_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}
```

### F16/BF16: float4（每线程 8 个 half = 16B）

对于 2 字节类型，一个 float4（16B）可以搬运 8 个元素：

```cuda
// F16: float4 = 16B = 8 × 2B
size_t vec_dim = embedding_dim / 8;
const float4* src4 = reinterpret_cast<const float4*>(src);
float4* dst4 = reinterpret_cast<float4*>(dst);
for (size_t i = threadIdx.x; i < vec_dim; i += blockDim.x) {
    dst4[i] = src4[i];  // 纯拷贝，不需要解包
}
```

**注意**: 这里没有做 half 的算术运算，只是拷贝，所以直接用 float4 搬运即可，不需要转换类型。

---

## 5. 模板参数：支持 I32/I64 索引

```cuda
template <typename IndexT>  // int32_t 或 int64_t
__global__ void embedding_kernel_f32(/* ... */) {
    IndexT idx = indices[row];  // 统一处理两种索引类型
}
```

在 dispatch 层根据 `index_dtype` 选择模板实例化：
```cpp
if (index_dtype == LLAISYS_DTYPE_I64)
    launch_embedding_typed<int64_t>(/* ... */);
else if (index_dtype == LLAISYS_DTYPE_I32)
    launch_embedding_typed<int32_t>(/* ... */);
```

---

## 6. ncu Profiling 数据

**测试环境**: RTX 4070 Laptop (36 SMs, sm_89), CUDA 12.6, ncu 2024.3.2
**模型级参数**: num_indices=512, vocab_size=151936, embedding_dim=1536

### 6.1 小数据测试（num_indices=50, embedding_dim=4096）

| 指标 | F32 |
|------|-----|
| Grid | 50 |
| DRAM Throughput | 56.09% |
| Duration | 5.25µs |
| Waves/SM | 0.23 |
| Achieved Occupancy | 22.36% |
| L2 Hit Rate | 52.56% |

**ncu 警告**: "kernel grid is too small to fill available resources, only 0.2 full waves"

**分析**: 50 个 block 远不够填满 36 SMs × 6 blocks/SM = 216 个槽位。
这不是 kernel 实现问题，而是**并行度不足**。小数据下的 DRAM 利用率低是正常的。

### 6.2 模型级参数测试（512 × 1536, vocab=151936）

| 指标 | F32 | F16 |
|------|-----|-----|
| Grid | 512 | 512 |
| DRAM Throughput | 81.62% | 70.67% |
| Duration | 14.27µs | 8.32µs |
| Waves/SM | 2.37 | 2.37 |
| Achieved Occupancy | 73.14% | 73.23% |
| L2 Hit Rate | 50.29% | 50.37% |
| Sector Utilization | 31.0/32 B (96.9%) | 30.2/32 B (94.4%) |

**ncu 警告**: "2 full waves and partial wave of 79 blocks → 33.3% tail effect"

**关键分析**:

1. **Tail Effect（尾部效应）**:
   - GPU 每次可调度 36 SMs × 6 blocks/SM = 216 blocks（一个 "wave"）
   - 512 blocks / 216 = 2.37 waves → 需要 3 次 wave 调度
   - 第 3 个 wave 只有 80 blocks，132/216 个槽位空闲 → **37% 利用率**
   - 整体浪费：约 21% 的 SM 时间

2. **F16 DRAM 低于 F32 的原因**:
   - F16: vec_dim = 1536 / 8 = 192 个 float4
   - F32: vec_dim = 1536 / 4 = 384 个 float4
   - F16 每行只有 192 个 float4，256 线程中有 64 个线程完全空闲（25%）
   - 空闲线程不搬运数据 → DRAM 利用率打折

3. **L2 Hit Rate ≈ 50%**: 因为 Gather 访问的 weight 行散布在 151936 行的表中，
   几乎不可能命中 L2（36 MB L2 vs 890 MB weight 表）。50% 中的命中主要来自 out 的写回。

---

## 7. Row-Stride 优化尝试及分析

### 7.1 优化思路

针对 Tail Effect，尝试 **Grid-Stride on rows**：
- 将 Grid 从 `num_indices` 缩小为 `min(num_indices, SM * 6)`
- 每个 block 通过 `for (row = blockIdx.x; row < num_indices; row += gridDim.x)` 循环处理多行
- 目标：消除第 3 个 wave 的低利用率

```cuda
// 修改前：Grid = num_indices（可能产生不满的末尾 wave）
int blocks = num_indices;  // 512 blocks → 3 waves, 尾部 37% 利用率

// 修改后：Grid = min(SM*6, num_indices)，每 block 循环处理多行
int num_sm = 0;
cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
int max_blocks = num_sm * 6;  // 36 * 6 = 216
int blocks = min(max_blocks, (int)num_indices);  // Grid = 216

// kernel 内部：
for (size_t row = blockIdx.x; row < num_indices; row += gridDim.x) {
    // ... 处理每一行
}
```

### 7.2 优化后 ncu 数据

| 指标 | F32 原始 | F32 Row-Stride | F16 原始 | F16 Row-Stride |
|------|----------|----------------|----------|----------------|
| Grid | 512 | 216 | 512 | 216 |
| Waves/SM | 2.37 | 1.0 | 2.37 | 1.0 |
| DRAM | 81.62% | 79.75% | 70.67% | 69.77% |
| Duration | 14.27µs | 14.78µs | 8.32µs | 8.35µs |
| Occupancy | 73.14% | 67.71% | 73.23% | 72.63% |

### 7.3 为什么 Row-Stride 没有改善性能？

**关键洞察：Tail Effect 对于均匀工作负载是不可避免的！**

**原始方案（512 blocks, 3 waves）**:
```
Wave 1: blocks 0-215   → 216/216 SM 满载，每 block 1 行
Wave 2: blocks 216-431 → 216/216 SM 满载，每 block 1 行
Wave 3: blocks 432-511 → 80/216 SM 在工作，136 个空闲
总时间 = 3 × T_per_row
```

**Row-Stride 方案（216 blocks, 1 wave）**:
```
Wave 1: 216 blocks 同时启动
  blocks 0-79:   处理行 0, 216, 432 = 3 行 → 3 × T_per_row
  blocks 80-215: 处理行 80, 296    = 2 行 → 2 × T_per_row
总时间 = max(3, 2) × T_per_row = 3 × T_per_row  ← 完全相同！
```

**数学证明**:
- 原始: 时间 = ⌈512 / 216⌉ × T_per_row = 3 × T_per_row
- Row-Stride: 时间 = ⌈512 / 216⌉ × T_per_row = 3 × T_per_row
- **两者的总时间完全相同！**

这是因为 Embedding 的每一行工作量完全相同（同样的 embedding_dim），不存在工作负载不均衡。
Tail Effect 只是换了个形式：从 "wave 间不均" 变成了 "block 间不均"。

**Row-Stride 真正有效的场景**:
1. **每行工作量不同**（如 SpMV）：Grid-Stride 能更好地负载均衡
2. **Wave 切换开销大**：减少 wave 数可以减少调度开销（但现代 GPU 调度开销极低）
3. **需要跨行共享数据**（如 reduction）：同一 block 处理多行，可利用 shared memory

### 7.4 核心结论

**Embedding 算子在当前实现下已接近最优**：

| 特性 | 状态 | 说明 |
|------|------|------|
| 向量化 | ✅ 已优化 | float4 LD.128，所有 dtype |
| 合并访存 | ✅ 已优化 | Sector 利用率 94-97% |
| 并行度 | ✅ 足够 | 模型参数下 Waves/SM > 2 |
| DRAM 利用率 | ~80% (F32) | Gather 访问的理论上限 |
| L2 利用率 | ~50% | 受限于 vocab_size 远大于 L2 容量 |

**瓶颈是物理限制，不是实现问题**：
- random gather 的 L2 miss 是不可避免的
- DRAM bandwidth 是硬件天花板
- Embedding 是**纯搬运**，没有计算可以融合来 amortize 访存开销

---

## 8. 潜在优化方向（超出 kernel 级别）

1. **Quantized Embedding**: 用 INT8/INT4 存储 weight，kernel 内解压（减少 DRAM 搬运量）
2. **与 RMSNorm 融合**: Embedding 后通常接 RMSNorm，融合后可减少一次全局内存写+读
3. **预排序 indices**: 排序后相邻的 index 可能访问相邻的 weight 行，提高 L2 命中率

---

## 9. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| Block-per-row 映射 | 一个 block 处理二维数据的一行，适合行内连续、行间独立的场景 |
| Gather 访存 | 按索引从大表中读取，访存不规则，依赖 L2 cache |
| 模板多态 `<IndexT>` | 用 CUDA 模板在编译期分发不同索引类型 |
| `__restrict__` | 告诉编译器指针无别名，允许更激进的优化 |
| float4 搬运技巧 | 对于纯拷贝，可以用 float4 搬运任何 16B 对齐的数据 |
| Tail Effect | 最后一个 wave 填不满所有 SM，造成资源浪费 |
| Waves/SM | ncu 指标，衡量并行度是否充分利用 GPU |
| Grid-Stride Loop | block 循环处理多个工作单元，适合负载不均匀场景 |
| **优化≠改善** | Row-Stride 对均匀负载无效 — 理解"为什么无效"比盲目优化更重要 |
