# Argmax 算子 CUDA 优化学习笔记

## 1. 算子概述

**功能**: 在一维张量中找到最大值及其索引。

**类型**: 归约操作（Reduction）— 将 N 个元素合并为 1 个结果。

**输入/输出**:
- 输入: vals `[N]` — 一维张量
- 输出: max_val（标量）, max_idx（标量）

---

## 2. GPU 归约的核心挑战

CPU 上归约是 O(N) 的串行遍历，GPU 上要做并行归约：

```
N 个元素 → log2(N) 轮比较 → 1 个结果
```

GPU 归约分三层：

| 层级 | 方法 | 特点 |
|------|------|------|
| **Warp 内** | `__shfl_down_sync` | 寄存器级，无内存开销，最快 |
| **Block 内** | Shared Memory + Warp Shuffle | 需要 `__syncthreads()` |
| **Grid 级** | 多 kernel / Atomic | 需要全局同步 |

---

## 3. Warp Shuffle 详解

`__shfl_down_sync(mask, val, delta)` — warp 内线程间直接交换寄存器值。

**工作原理**:
```
lane i 读取 lane (i + delta) 的 val
不经过内存，直接通过 warp 内的 crossbar 网络传输
```

**示例（8 线程找最大值）**:

```
初始:     [3, 7, 1, 9, 5, 2, 8, 4]
          
Round 1 (delta=4):  
  lane0 = max(lane0, lane4) = max(3,5) = 5
  lane1 = max(lane1, lane5) = max(7,2) = 7
  lane2 = max(lane2, lane6) = max(1,8) = 8
  lane3 = max(lane3, lane7) = max(9,4) = 9
结果:     [5, 7, 8, 9, -, -, -, -]

Round 2 (delta=2):
  lane0 = max(lane0, lane2) = max(5,8) = 8
  lane1 = max(lane1, lane3) = max(7,9) = 9
结果:     [8, 9, -, -, -, -, -, -]

Round 3 (delta=1):
  lane0 = max(lane0, lane1) = max(8,9) = 9
结果:     [9, -, -, -, -, -, -, -]  ✅ lane0 持有全局最大值
```

**CUDA 代码**:
```cuda
__device__ void warp_reduce_max(float& val, size_t& idx) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, val, delta);
        size_t other_idx = __shfl_down_sync(0xffffffff, idx, delta);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // 归约完成: lane 0 持有 warp 内的最大值和索引
}
```

**关键参数**:
- `0xffffffff` — 32 位 mask，全部 32 个 lane 参与
- `delta` 从 16 → 8 → 4 → 2 → 1，共 5 轮，归约 32 → 1

---

## 4. Block 级归约：Warp → Shared Memory → Warp 0

```cuda
// 256 线程 = 8 个 warp
__shared__ float s_val[32];   // 每 warp 一个槽位
__shared__ size_t s_idx[32];

// Step 1: Warp 内归约
warp_reduce_max(local_max, local_idx);

// Step 2: 每 warp 的 lane 0 写入 shared memory
if (lane == 0) {
    s_val[warp_id] = local_max;
    s_idx[warp_id] = local_idx;
}
__syncthreads();

// Step 3: Warp 0 读取所有 warp 结果，再做一轮归约
if (warp_id == 0) {
    float val = (lane < num_warps) ? s_val[lane] : -FLT_MAX;
    size_t idx = (lane < num_warps) ? s_idx[lane] : 0;
    warp_reduce_max(val, idx);
    // lane 0 持有全局最大值
}
```

**数据流**:
```
256 线程
  ↓ warp shuffle (5 轮)
8 个 warp 的 lane 0
  ↓ 写 shared memory
8 个值
  ↓ warp 0 读取 + warp shuffle (5 轮)
1 个最终结果（thread 0）
```

---

## 5. 单 Block 策略

当前实现使用 **单 block（256 线程）+ Grid-Stride Loop**：

```cuda
argmax_kernel<<<1, 256>>>(max_idx, max_val, vals, numel);
```

**适用场景**: numel 不太大（测试中最大 4096 元素）。

**如果 numel 很大怎么办？**
- 方案 A: 多 block → 每 block 输出一个局部最大值 → 第二个 kernel 归约
- 方案 B: 多 block + `atomicCAS` 直接汇总
- 方案 C: CUB 库的 `cub::DeviceReduce::ArgMax`（生产级方案）

---

## 6. 性能分析与优化

### 6.1 ncu 基线数据（单 Block, numel=151936, F32）

| 指标 | 值 |
|------|-----|
| Grid | 1 |
| DRAM Throughput | 1.21% |
| Memory Throughput | 3.30 GB/s |
| Duration | 185.70 µs |
| Waves/SM | 0.00 |
| Achieved Occupancy | 16.70% |
| SM Busy | 5.64% |

**问题**: 单 block = 1/36 SMs 在工作，其余 35 个 SM 完全空闲！
对于 vocab_size=151936（LLM 推理的实际场景），这是严重的并行度不足。

### 6.2 优化：多 Block 两阶段归约

**策略**: 将归约拆为两个 phase：
```
Phase 1: 144 blocks 并行归约 → 每 block 输出 1 个局部 (val, idx)
Phase 2: 1 block 归约 144 个结果 → 最终答案

数据流:
  vals[151936] ──Phase1(144 blocks)──→ tmp[144] ──Phase2(1 block)──→ max_val, max_idx
```

Block 数量 = SM × 4 = 36 × 4 = 144：
- 每 SM 4 个 block → 充分隐藏访存延迟
- 每 block 处理 151936/144 ≈ 1055 个元素

临时缓冲区使用**持久化 static 变量**，避免每次调用 cudaMalloc/cudaFree 的 ~1ms 开销。

### 6.3 优化后 ncu 数据

| 指标 | 单 Block | 多 Block (Phase1) | 提升 |
|------|----------|-------------------|------|
| Grid | 1 | 144 | 144x |
| DRAM | 1.21% | 47.30% | 39x |
| Duration | 185.70µs | 4.83µs | **38.5x** |
| Waves/SM | 0.00 | 0.67 | ∞ |
| Occupancy | 16.70% | 60.14% | 3.6x |
| Memory BW | 3.30 GB/s | 126.70 GB/s | 38x |

### 6.4 为什么 DRAM 没到 90%+？

Argmax 的内存访问有**重度分支（branch）**：
```cuda
if (v > local_max) {   // 每次比较都可能分支
    local_max = v;
    local_idx = i;
}
```
- L2 Hit Rate 仅 5.33%（数据 593KB > L1 但 < L2 36MB，应该命中才对）
  → 因为 144 blocks × 256 threads 同时访问不同区域，L2 压力大
- 分支预测失败导致 warp 执行效率下降
- 数据量小（593KB），kernel launch overhead 占比较高

### 6.5 核心结论

Argmax 是**轻量算子**（593KB 数据，4.83µs），在 LLM 推理中耗时占比极低。
多 Block 优化已将其从 186µs 降到 4.83µs，进一步优化（float4 向量化读取）的收益有限。

---

## 7. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| `__shfl_down_sync` | Warp 内寄存器交换，归约操作的核心原语 |
| Warp Shuffle Reduce | 5 轮 shuffle 将 32 个值归约为 1 个 |
| Shared Memory | Block 内线程通信的桥梁，用于跨 warp 传递数据 |
| `__syncthreads()` | Block 级同步屏障，确保 shared memory 写入完成 |
| `-FLT_MAX` | float 最小值常量，归约初值 |
| Lane / Warp / Block | GPU 线程层级：lane ∈ warp (32线程) ∈ block |
| Two-phase Reduction | 先 warp 内归约，再 warp 间归约，是 GPU 归约的标准模式 |
| **Multi-Block 归约** | 大数据用多 block 并行 → 临时缓冲区 → 二次归约，利用全部 SM |
| **持久化临时缓冲区** | static 变量避免反复 cudaMalloc/cudaFree，只增不缩 |
| **SM 并行度** | 单 block 浪费 35/36 SMs → 多 block 可提速 36x |
