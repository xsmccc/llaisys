# SwiGLU 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: `out = up * SiLU(gate)`，其中 `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`

**类型**: 访存密集型（Memory Bound）— 虽然有 exp 计算，但 GPU 的 SFU（Special Function Unit）执行 exp 很快，整体仍受带宽限制。

**输入/输出**: gate[N], up[N] → out[N]，总访存量 = 3N 个元素（读2写1）。

---

## 2. 优化迭代全过程（含 ncu 数据）

这是我们做得最深入的一个算子，经历了三轮优化，**最终得出了重要的性能分析结论**。

### 2.1 版本 0: Naive（每线程1个元素）

```cuda
__global__ void swiglu_naive(float* out, const float* gate, const float* up, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        out[idx] = up[idx] * g / (1.0f + __expf(-g));
    }
}
```

**ncu 分析结果**:
| 指标 | 值 |
|------|-----|
| Duration | 68.86 µs |
| DRAM Throughput | **95.32%** |
| Compute (SM) Throughput | ~30% |
| Waves Per SM | 37.93 |

**结论**: 已经达到 95% 的 DRAM 峰值带宽！Memory Bound 明确。

### 2.2 版本 1: float4 向量化（错误尝试）

**思路**: 用 float4 让每个线程处理 4 个元素 → Grid 缩小 4 倍。

```cuda
// Grid 从 8192 blocks 缩小为 2048 blocks
size_t numel_vec = (numel + 3) / 4;
int blocks = (numel_vec + threads - 1) / threads;
```

**ncu 分析结果**:
| 指标 | 值 | 变化 |
|------|-----|------|
| Duration | **96.35 µs** | ❌ 变慢 40% |
| DRAM Throughput | ~70% | ❌ 下降 |
| Waves Per SM | **9.48** | ❌ 从 37.93 暴跌 |

**根因分析**:
- Grid 缩小了 4 倍 → 总 block 数从 8192 降到 2048
- RTX 4070 Laptop 有 36 个 SM，2048 / 36 ≈ 57 blocks/SM，但每个 block 的 warp 变少
- **Waves Per SM 从 37.93 降到 9.48** → SM 上没有足够的 warp 来隐藏内存延迟
- GPU 大量时间在 stall 等待内存返回数据

> **教训**: 向量化减少了指令数，但如果同时减少了 GPU 的并行度（warp/waves），可能反而更慢！

### 2.3 版本 2: Grid-Stride Loop + float4（最终版本）

**核心思路**: 保持 Grid 大小不变（高并行度），让每个线程用循环处理多个 float4 块。

```cuda
__global__ void swiglu_f32_vec4(float* out, const float* gate, const float* up, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;  // 总线程数

    // 每线程用循环跳步处理多个 float4
    for (size_t i = tid; (i + 1) * 4 <= n; i += stride) {
        size_t base = i * 4;
        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up + base);
        float4 o4;
        o4.x = u4.x * silu(g4.x);
        o4.y = u4.y * silu(g4.y);
        o4.z = u4.z * silu(g4.z);
        o4.w = u4.w * silu(g4.w);
        *reinterpret_cast<float4*>(out + base) = o4;
    }
}
```

**Grid 大小计算**:
```cuda
int num_sm = 0;
cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);  // 36
int BLOCKS = num_sm * 8;  // 288 blocks × 256 threads = 73728 总线程
```

**ncu 分析结果**:
| 指标 | 值 | 变化 |
|------|-----|------|
| Duration | **69.02 µs** | ≈ 持平 naive |
| DRAM Throughput | **95%+** | ≈ 同 naive |
| Waves Per SM | 充足 | ✅ 恢复正常 |

### 2.4 最终结论

**SwiGLU 在 naive 版本就已经达到了 ~95% DRAM 带宽利用率**，这意味着：

1. 内存带宽已经是硬件瓶颈的天花板 — **没有 kernel 级优化空间了**
2. 向量化没有提升，因为 naive 的 sector 利用率本身就很高（连续访存模式）
3. **唯一能提升的方式是 Kernel Fusion** — 比如把 SwiGLU 和前序/后续操作融合，减少全局内存读写次数

**理论峰值计算**:
```
数据量 = 3 × 2^20 个 float × 4B = 12 MB（1M 元素的 swiglu）
RTX 4070 Laptop 峰值带宽 ≈ 256 GB/s
理论最优时间 = 12MB / 256GB/s ≈ 46.9 µs
实际: 69 µs → 效率 ≈ 68%（考虑开销后合理）
```

---

## 3. Grid-Stride Loop 模式详解

这是 CUDA 编程中最重要的模式之一：

```
线程数 = blocks × threads_per_block  （固定）
stride = 线程数

for (i = tid; i < total_work; i += stride) {
    // 处理第 i 个工作单元
}
```

**优势**:
1. Grid 大小与数据大小解耦 — 不用担心 65535 blocks 限制
2. 可以精确控制 occupancy — 选择最优的 blocks/SM
3. 与向量化组合后既有高并行度又有宽指令

**Grid 大小经验公式**:
```
blocks = num_SM × (target_warps_per_SM / warps_per_block)
       = 36 × (48 / 8)  // RTX 4070: 36 SM, 目标 48 warp/SM, 256 线程/block = 8 warp
       = 36 × 6 = 216    // 或者更保守地 36 × 8 = 288
```

---

## 4. 关键 ncu 指标速查

| 指标 | 看什么 | 含义 |
|------|--------|------|
| Duration | kernel 耗时 | 最终性能指标 |
| DRAM Throughput | 带宽利用率 % | Memory-bound 算子的核心指标 |
| SM Throughput | 计算利用率 % | Compute-bound 算子的核心指标 |
| Waves Per SM | 每 SM 的波次数 | < 4 说明并行度不足 |
| Achieved Occupancy | 活跃 warp / 最大 warp | 影响 latency hiding |
| L1TEX Throughput | L1 缓存吞吐 | 高 → 可能存在 sector 浪费 |

**判断瓶颈**:
- DRAM > 70%, SM < 50% → **Memory Bound** → 优化访存 / 减少数据量 / kernel fusion
- SM > 70%, DRAM < 50% → **Compute Bound** → 减少计算 / 用更快的指令
- 两者都低 → **Latency Bound** → 增加并行度 / 提高 occupancy

---

## 5. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| Grid-Stride Loop | 固定 Grid 大小，线程循环处理数据，解耦并行度与数据大小 |
| Waves Per SM | 描述 SM 上有多少"波"的 warp 可以执行，代表 latency hiding 能力 |
| `__expf` | CUDA 快速数学 exp，精度略低（~2 ULP）但速度快 |
| `cudaDeviceGetAttribute` | 运行时查询 GPU 属性（SM 数、最大线程数等） |
| Memory Bandwidth Ceiling | 当 DRAM 利用率 > 90%，说明已经触碰硬件极限 |
| Kernel Fusion | 多个算子合并为一个 kernel，减少全局内存访问,是突破带宽瓶颈的关键 |

---

## 6. 本次优化 Session 详细 ncu 数据

### 6.1 F32 Grid-Stride + float4（当前版本）

测试条件：shape=(512, 4096), numel=2,097,152, RTX 4070 Laptop (36 SM)

```
ncu --set full --kernel-name regex:"swiglu_kernel.*" --launch-skip 5 --launch-count 1
```

| 指标 | 值 |
|------|-----|
| **Duration** | **65.50 µs** |
| **DRAM Throughput** | **94.91%** |
| Memory Throughput | 257.68 GB/s |
| Compute (SM) Throughput | 8.13% |
| SM Busy | 8.70% |
| Executed IPC Active | 0.33 inst/cycle |
| Issue Slots Busy | 8.38% |
| Active Warps Per Scheduler | 10.18 / 12 |
| Eligible Warps Per Scheduler | 0.10 |
| L1TEX stall | 114.3 cycles (94.9%) |
| Registers Per Thread | 36 |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 83.97% |
| Grid Size | 288 |
| Block Size | 256 |
| Executed Instructions | 1,499,654 |

**分析**：DRAM 94.91% → 在带宽天花板上。SM 只有 8.7%，计算不是瓶颈。
94.9% 的 stall 在等 L1TEX（全局内存访问）→ 典型 Memory-Bound。

### 6.2 F16 half2 版本（优化前基线）

| 指标 | 值 |
|------|-----|
| **Duration** | **34.34 µs** |
| **DRAM Throughput** | **90.23%** |
| Memory Throughput | 244.54 GB/s |
| SM Busy | **22.30%** |
| Executed IPC Active | 0.89 |
| Active Warps Per Scheduler | 8.88 |
| **Executed Instructions** | **2,035,770** |
| Registers Per Thread | 28 |
| L1TEX stall | 39.4 cycles (86.0%) |

**分析**：
- Duration ≈ F32 的一半（数据量减半）
- DRAM 90.23% → 比 F32 低 5%，说明有优化空间
- SM Busy 22.30% → 比 F32 的 8.7% 高很多 → 大量类型转换指令
- 指令总数 2,035,770 → 比 F32 的 1,499,654 多 36%！
  → 原因：half2 = LD.32 每线程只搬 4B，需要更多指令
  → 加上 half↔float 的类型转换指令开销

### 6.3 F16 float4 版本（优化后）

优化内容：
- `half2`（LD.32, 4B/线程）→ `float4`（LD.128, 16B/线程 = 8 个 half）
- 添加 `__restrict__` 指针提示
- 尾部处理从 "奇数 1 个" 改为 "最多 7 个"

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| **Duration** | 34.34 µs | **34.24 µs** | -0.3% (持平) |
| **DRAM Throughput** | 90.23% | **90.43%** | +0.2% |
| Memory Throughput | 244.54 GB/s | 245.34 GB/s | +0.3% |
| **Executed Instructions** | 2,035,770 | **1,482,693** | **-27.2%** ✅ |
| SM Busy | 22.30% | **16.87%** | -24.3% ✅ |
| Executed IPC | 0.89 | 0.64 | -28.1% |
| Registers/Thread | 28 | 36 | +28.6% |
| Active Warps/Scheduler | 8.88 | 10.18 | +14.6% ✅ |
| L1TEX stall | 39.4 cyc | 53.5 cyc | +35.8% |
| Block Limit Registers | 8 | 6 | 退化 |

**分析**：
1. **指令数减少 27%** — float4 宽搬运确实减少了 LD/ST 指令数量
2. **SM Busy 降低 24%** — 更少的指令 = SM 更轻松
3. **Duration 几乎不变** — DRAM 仍在 90%+，Memory-Bound 天花板
4. **寄存器从 28→36** — 每线程同时持有 8 个 gate + 8 个 up + 中间变量
   - Block Limit Registers 从 8→6，但 Theoretical Occupancy 仍是 100%
5. **Active Warps 提升** — 更均匀的工作分配

**结论**：float4 优化在指令层面有显著改善（-27%），但在DRAM带宽天花板下对 Duration 影响微乎其微。这再次验证了 **Memory-Bound 算子只有 Kernel Fusion 能显著提升性能** 的结论。

---

## 7. 三个数据类型对比总结

| 指标 | F32 | F16 (优化后) | 理论关系 |
|------|-----|-------------|---------|
| Duration | 65.50 µs | 34.24 µs | F16 ≈ F32/2 ✓（数据量减半）|
| DRAM Throughput | 94.91% | 90.43% | F32 更高（float4 = 原生匹配）|
| Registers/Thread | 36 | 36 | 相同（都用 float4）|
| Instructions | 1,499,654 | 1,482,693 | F16 略少（数据量小→循环轮次少）|
| SM Busy | 8.70% | 16.87% | F16 更高（half↔float 转换开销）|

---

## 8. ncu 实用命令模板

```bash
# F32 profiling
sudo ncu --set full \
  --kernel-name regex:"swiglu_kernel.*" \
  --launch-skip 5 --launch-count 1 \
  /path/to/python scripts/Swiglu/profile_swiglu.py --dtype f32 --iterations 10

# F16 profiling
sudo ncu --set full \
  --kernel-name regex:"swiglu_kernel.*" \
  --launch-skip 5 --launch-count 1 \
  /path/to/python scripts/Swiglu/profile_swiglu.py --dtype f16 --iterations 10

# BF16 profiling
sudo ncu --set full \
  --kernel-name regex:"swiglu_kernel.*" \
  --launch-skip 5 --launch-count 1 \
  /path/to/python scripts/Swiglu/profile_swiglu.py --dtype bf16 --iterations 10
```

---

## 9. 核心优化经验总结

### 三条规律（Add + SwiGLU 验证）

1. **DRAM > 90% 时，kernel 内优化（向量化、减指令）对 Duration 影响 <1%**
   - Add F16: 33.76µs → 33.47µs（-0.9%）
   - SwiGLU F16: 34.34µs → 34.24µs（-0.3%）

2. **float4 宽搬运的真正价值是减少指令数（-27~37%），而非减少 Duration**
   - 减少的 issue slot pressure 在更复杂的 pipeline 中可能间接受益

3. **突破 Memory-Bound 天花板唯一方法：Kernel Fusion**
   - 例如：Linear → SwiGLU 可以融合，避免 SwiGLU 独立读写 DRAM
   - 节省 2N 元素的 DRAM 读写（Linear 的输出不需要写回再被 SwiGLU 读取）
