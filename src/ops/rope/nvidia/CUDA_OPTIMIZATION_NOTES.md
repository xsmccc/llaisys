# RoPE 算子 CUDA 优化学习笔记

## 1. 算子概述

**全称**: Rotary Position Embedding（旋转位置编码）

**公式**: 对输入向量的前后半部分 `(a, b)` 应用二维旋转：
$$a'_j = a_j \cos(\phi) - b_j \sin(\phi)$$
$$b'_j = b_j \cos(\phi) + a_j \sin(\phi)$$

其中角度 $\phi_{i,j} = p_i / \theta^{2j/d}$，$p_i$ 是位置 ID，$\theta$ 是基础频率（通常 10000）。

**功能**: 将位置信息编码到 Q/K 向量中，使得内积自然反映相对位置关系。

**类型**: 计算密集型 elementwise — sin/cos 计算较贵（映射到 SFU）。

**张量形状**:
- in/out: `[seqlen, nhead, head_dim]`
- pos_ids: `[seqlen]` — 每个 token 的绝对位置 ID

---

## 2. 数学原理

RoPE 的核心是**复数旋转**。将实数对 `(a, b)` 视为复数 `a + bi`，乘以旋转因子 `e^{iφ}`：

```
(a + bi) × (cos φ + i sin φ) = (a cos φ - b sin φ) + (b cos φ + a sin φ)i
```

不同频率的旋转让每个维度对编码不同尺度的位置信息：
- 低维度（j 小）→ 频率高 → 编码近距离位置差异
- 高维度（j 大）→ 频率低 → 编码远距离位置差异

---

## 3. 线程映射：1D 展平策略

```
总工作量 = seqlen × nhead × (head_dim / 2)  // 每对 (a,b) 算一次
```

将三维索引展平为一维 `tid`：

```cuda
for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
     tid < total;
     tid += blockDim.x * gridDim.x)
{
    // 从 tid 反推三维索引
    size_t j = tid % half_dim;               // dim 内位置
    size_t h = (tid / half_dim) % n_heads;   // head 索引
    size_t i = tid / (half_dim * n_heads);   // seq 索引
}
```

**为什么不用 2D/3D Grid？**
1. 展平后用 Grid-Stride Loop 更灵活，不受 gridDim 限制
2. 三维可以通过整数除法和取模高效反推
3. 代码更简洁统一

---

## 4. 精度处理：double 精度角度计算

```cuda
// 用 double 精度避免大 pos_id 下的精度损失
int64_t p = pos_ids[i];
double freq = 1.0 / pow(static_cast<double>(theta), 2.0 * j / head_dim);
double angle = static_cast<double>(p) * freq;
float cos_val = static_cast<float>(cos(angle));
float sin_val = static_cast<float>(sin(angle));
```

**为什么用 double？**
- `theta^(2j/d)` 可能非常大（theta=10000, 2j/d 接近 1 时 → 10000）
- `pos_id` 在长序列中可能达到数千甚至数万
- `pos_id × freq` 的乘积在 float 精度下误差会累积
- 角度的微小误差会被 sin/cos 放大（尤其在高频分量）

**性能影响**:
- double 精度 sin/cos 比 float 慢 2-4x
- 但 RoPE 的工作量小（只有 seqlen × nhead × head_dim/2），不是推理瓶颈

---

## 5. 内存访问模式

```cuda
size_t offset = i * n_heads * head_dim + h * head_dim;
float a = to_float(in[offset + j]);               // 前半部分
float b = to_float(in[offset + j + half_dim]);     // 后半部分
```

**特点**:
- a 和 b 在内存中间隔 `half_dim` 个元素
- 同一 warp 内相邻线程访问的 j 连续 → `a` 的读取是合并的
- `b` 的读取也是合并的（偏移固定为 half_dim）
- 写回同理

**没有 Bank Conflict**: 不使用 shared memory，完全是全局内存读写。

---

## 6. Grid 配置

```cuda
constexpr int THREADS = 256;
size_t total = seq_len * n_heads * (head_dim / 2);
int blocks = min((total + 255) / 256, 65535);
```

简单的 "一线程一元素对" 映射。由于 RoPE 中 sin/cos 计算是瓶颈（而非内存），不需要复杂的 Grid-Stride 或向量化优化。

---

## 7. ncu Profiling 与 float 精度优化

**测试环境**: RTX 4070 Laptop (36 SMs, sm_89), CUDA 12.6
**测试参数**: seqlen=512, nhead=4, head_dim=4096, pos_ids=[512, 1024)

### 7.1 基线数据（double 精度 pow + sin + cos）

| 指标 | F32 |
|------|-----|
| DRAM Throughput | 7.14% |
| Compute Throughput | **84.10%** |
| Duration | **5.38 ms** |
| SM Busy | 86.92% |
| Memory BW | 16.97 GB/s |

**严重的 Compute-Bound！** 原因：
- `pow(double, double)` → ~400 cycle（FP64 = FP32 吞吐量的 1/64）
- `sin(double)` → ~200 cycle
- `cos(double)` → ~200 cycle
- 每线程总计 ~800 cycle，全部在 FP64 单元上执行

### 7.2 优化：全 float 精度 + sincosf

| 优化项 | 原始 | 优化后 | 加速 |
|--------|------|--------|------|
| 频率计算 | `pow(double, double)` ~400 cycle | `powf(float, float)` ~16 cycle | **25x** |
| sin/cos | `sin(double)` + `cos(double)` ~400 cycle | `sincosf(float)` ~8 cycle | **50x** |
| 每线程总计 | ~800 cycle | ~24 cycle | **33x** |

**精度分析**:
- `powf` 精度 ~1e-6 相对误差
- `sincosf` 精度 ~1e-6 相对误差
- 对 angle 的 range reduction 精确到 2^23 (~8M 弧度)
- LLM 推理中 pos_id < 128K → 完全安全
- **关键**: 匹配 PyTorch 的计算路径（`pos / theta^exp` 而非 `pos * 1/theta^exp`）
  以保持浮点舍入一致性

### 7.3 优化后数据

| 指标 | F32 原始 | F32 优化 | 提升 |
|------|----------|----------|------|
| DRAM | 7.14% | 88.35% | 12x |
| Compute | 84.10% | 61.60% | (不再是瓶颈) |
| Duration | 5380µs | 207.81µs | **25.9x** |
| Memory BW | 16.97 | 240.05 | 14x |
| 性质 | Compute-Bound | **Memory-Bound** | 根本转变 |

F16 优化后: Duration 155.62µs, Compute 83%, DRAM 58%
（F16 数据量减半但 sin/cos 计算量不变 → 仍然 compute-bound）

### 7.4 为什么这是 CUDA 优化中最重要的教训之一

> **消费级 GPU 的 FP64 吞吐量仅为 FP32 的 1/64**（Ada Lovelace 架构）

这意味着一个 `sin(double)` 比 `sincosf(float)` 慢 **50-100x**。
在工程实践中，除非有严格的精度要求（如科学计算），CUDA kernel 应**始终使用 float**。

---

## 8. 潜在进一步优化

| 优化 | 方法 | 效果 |
|------|------|------|
| 预计算频率表 | 将 freq[] 提前算好存到 device memory | 避免每线程做 powf |
| `__sincosf` 快速版 | 用 CUDA fast math intrinsic | 再快 ~2x，但精度降低 |
| 与 Attention 融合 | RoPE 后接 Attention，可融合减少读写 | 减少全局内存一次往返 |
| float2 向量化 | (a, b) 用 float2 一起 load/store | 减少内存事务数 |

---

## 9. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| 1D 展平策略 | 将多维问题展平为一维，用取模和除法反推索引 |
| SFU | Special Function Unit，执行 sin/cos/exp/rsqrt 等超越函数 |
| **FP64 vs FP32 吞吐量** | 消费级 GPU FP64 = FP32 的 1/64（致命性能陷阱！）|
| `sincosf` | 单条指令同时计算 sin 和 cos（比分开调用快） |
| `powf` vs `pow` | float 版 ~16 cycle vs double 版 ~400 cycle |
| Grid-Stride + 上限裁剪 | `min(blocks, 65535)` 防止超过 CUDA 最大 grid 维度 |
| **Compute→Memory Bound 转变** | 优化计算后瓶颈从 SFU 转移到 DRAM |
