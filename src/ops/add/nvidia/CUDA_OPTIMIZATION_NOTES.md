# Add 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: `C[i] = A[i] + B[i]`

**类型**: 纯访存密集型（Memory Bound）— 零计算（只有加法），性能完全取决于内存带宽。

**输入/输出**: 三个形状相同的连续张量 A、B、C，支持 F32 / F16 / BF16。

---

## 2. Naive 版本分析

最简单的实现：每个线程处理 1 个元素。

```cuda
__global__ void add_kernel_naive(float* c, const float* a, const float* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

**问题**: 每个 float 是 4 字节，GPU 内存总线宽度通常是 32 字节（256 位）。每次只读 4 字节意味着内存事务利用率只有 12.5%。

---

## 3. ncu 瓶颈分析

使用 `ncu` 分析 naive 版本时的关键指标：

| 指标 | 含义 |
|------|------|
| **DRAM Throughput** | 实际内存带宽利用率。Add 是纯访存，理想应接近峰值 |
| **L1TEX Throughput** | L1 缓存吞吐，高说明有大量细粒度内存事务 |
| **Achieved Occupancy** | SM 上活跃 warp 的比例，影响 latency hiding |
| **Memory ↔ Compute** | `Speed Of Light` 面板显示瓶颈在哪侧 |

**典型发现**: naive 版本的 L1TEX sector 利用率偏低（每事务只需要 4B 但占用 32B sector），可以通过向量化改善。

---

## 4. 优化方法：向量化访存

### 核心思路

将多次小内存访问合并为一次大访问，利用 GPU 的 **LD.128 / ST.128** 指令：

| 数据类型 | 向量化类型 | 每线程每步 | 内存事务宽度 |
|----------|-----------|-----------|------------|
| F32      | `float4`  | 4 个 float = 16B | 128-bit |
| F16      | `half2`   | 2 个 half = 4B   | 32-bit （half2 原生指令） |
| BF16     | `bfloat162`| 2 个 bf16 = 4B  | 32-bit （bfloat162 原生指令） |

### F32 float4 实现

```cuda
__global__ void add_kernel_f32_vec(float *c, const float *a, const float *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 4;

    if (base + 3 < numel) {
        // 一次加载 16 字节 → 编译为 LD.128 指令
        float4 a4 = *reinterpret_cast<const float4 *>(a + base);
        float4 b4 = *reinterpret_cast<const float4 *>(b + base);
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        *reinterpret_cast<float4 *>(c + base) = c4;
    } else {
        // 尾部标量处理
        for (size_t i = base; i < numel && i < base + 4; ++i)
            c[i] = a[i] + b[i];
    }
}
```

### F16 half2 实现

```cuda
// half2 可以直接用硬件指令 __hadd2 做并行加法
__half2 a2 = *reinterpret_cast<const __half2 *>(a + base);
__half2 b2 = *reinterpret_cast<const __half2 *>(b + base);
__half2 c2 = __hadd2(a2, b2);  // 一条指令完成两个 half 加法
```

**关键点**: `__hadd2` 是 SM 硬件原生支持的 SIMD 指令，一条指令处理两个 FP16 值，吞吐量是标量 `__hadd` 的两倍。

---

## 5. Grid 配置策略

```
threads = 256
numel_vec = (numel + 3) / 4     // F32: 每线程处理4个元素
blocks = (numel_vec + 255) / 256
```

这是最简单的 "一线程一元素组" 策略，适合 Add 这种均匀负载的算子。

---

## 6. 进一步优化方向

由于 Add 算子已经在带宽天花板附近，进一步优化只能通过：

1. **Kernel Fusion（算子融合）**: 将 Add 与后续操作（如 ReLU）合并，减少额外的全局内存读写
2. **异步拷贝 + 流水线**: 对很大的张量，可以用 `cp.async` 将加载与计算重叠

---

## 7. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| `float4` / `half2` | CUDA 内置向量类型，映射到宽内存指令 |
| `reinterpret_cast` | 将 `float*` 视为 `float4*`，告诉编译器生成宽加载指令 |
| `LD.128` / `ST.128` | 128 位内存指令，一次搬运 16 字节 |
| Memory Bound | 性能瓶颈在内存带宽而非计算 |
| Sector 利用率 | 每个 32B sector 中实际使用的字节比例 |

---

## 8. ncu/nsys 实战分析指南

### 8.1 ncu Profiling 命令

```bash
# 激活虚拟环境
source venv/bin/activate

# 完整 profiling（推荐首次使用）
ncu --set full -o add_profile python test/ops/add.py --device nvidia

# 只看内存相关指标（更快）
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__time_duration.avg \
    python test/ops/add.py --device nvidia

# 查看生成的汇编代码（验证 LD.128/ST.128）
ncu --print-source sass python test/ops/add.py --device nvidia 2>&1 | grep -E "LD\.|ST\."
```

### 8.2 关键指标解读

| 指标 | 含义 | Add 算子目标 |
|------|------|-------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 | **>80%** = 优秀 |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 缓存吞吐 | 参考值 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算利用率 | **<30%**（Memory-Bound 特征） |
| `gpu__time_duration.avg` | Kernel 耗时 (ns) | 越低越好 |

### 8.3 判断瓶颈

```
DRAM > 70%, SM < 50%  → ✅ Memory Bound（符合预期）
SM > 70%, DRAM < 50%  → ❌ Compute Bound（不应出现）
两者都 < 50%          → ⚠️ Latency Bound（并行度不足）
```

### 8.4 Sector 效率分析

```bash
ncu --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    python test/ops/add.py --device nvidia
```

计算方法：
```
实际 sectors/request = sectors.sum / requests.sum
理想值（float4）= 4（因为 128B float4 / 32B sector = 4）
效率 = 理想值 / 实际值 × 100%
```

### 8.5 nsys 系统级分析

```bash
# 生成 timeline
nsys profile -o add_timeline python test/ops/add.py --device nvidia

# 查看报告
nsys stats add_timeline.nsys-rep
```

关注点：
- Kernel Launch Overhead（短 kernel 时占比大）
- 意外的 D2H/H2D 内存传输
- cudaDeviceSynchronize 阻塞

---

## 9. 理论性能计算与效率分析

### 9.1 理论带宽需求

```
设 numel = 1,048,576 (1M 元素)
数据量 = (读A + 读B + 写C) = 3 × 1M × 4B = 12 MB (F32)

RTX 4070 Laptop 峰值带宽 ≈ 256 GB/s
理论最小时间 = 12 MB / 256 GB/s ≈ 46.9 µs
```

### 9.2 效率计算

```
效率 = 理论时间 / 实际时间 × 100%

例如 ncu 测得 Duration = 60 µs:
效率 = 46.9 / 60 × 100% ≈ 78%

78% 对于简单 kernel 是合理的（launch 开销、尾部处理等）
```

### 9.3 实测数据模板（回学校后填写）

| 测试配置 | 值 |
|---------|-----|
| GPU | RTX 4070 Laptop |
| numel | ___________ |
| dtype | F32 / F16 / BF16 |

| 指标 | F32 | F16 | BF16 |
|------|-----|-----|------|
| Duration (µs) | ___ | ___ | ___ |
| DRAM Throughput (%) | ___ | ___ | ___ |
| SM Throughput (%) | ___ | ___ | ___ |
| 实际带宽 (GB/s) | ___ | ___ | ___ |
| 效率 (%) | ___ | ___ | ___ |

---

## 10. 进阶优化代码示例

### 10.1 Grid-Stride Loop 版本

```cuda
__global__ void add_kernel_f32_grid_stride(
    float* __restrict__ c,
    const float* __restrict__ a, 
    const float* __restrict__ b,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;  // 总线程数
    
    // 每线程循环处理多个 float4
    for (size_t i = tid; (i + 1) * 4 <= numel; i += stride) {
        size_t base = i * 4;
        float4 a4 = *reinterpret_cast<const float4*>(a + base);
        float4 b4 = *reinterpret_cast<const float4*>(b + base);
        float4 c4 = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
        *reinterpret_cast<float4*>(c + base) = c4;
    }
    
    // 尾部处理
    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;
        for (size_t i = tail_start; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

// Launch 配置
int num_sm;
cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
int blocks = num_sm * 8;  // 固定 grid 大小
add_kernel_f32_grid_stride<<<blocks, 256>>>(c, a, b, numel);
```

**优势**: 
- Grid 大小固定 → 无论数据大小，launch 开销恒定
- 小数据时保持高 occupancy
- 大数据时避免过大 grid

### 10.2 F16 用 float4 宽搬运

```cuda
// 当前: half2 一次搬 4B（2 个 half）
// 优化: float4 一次搬 16B（8 个 half）

__global__ void add_kernel_f16_wide(
    __half* c, const __half* a, const __half* b, size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = tid * 8;  // 每线程 8 个 half
    
    if (base + 7 < numel) {
        // LD.128 搬 16 字节 = 8 个 half
        float4 a_chunk = *reinterpret_cast<const float4*>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4*>(b + base);
        
        // 拆成 4 个 half2 做向量加法
        __half2* a_h2 = reinterpret_cast<__half2*>(&a_chunk);
        __half2* b_h2 = reinterpret_cast<__half2*>(&b_chunk);
        __half2 c_h2[4];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c_h2[i] = __hadd2(a_h2[i], b_h2[i]);
        }
        
        // ST.128 写回
        *reinterpret_cast<float4*>(c + base) = *reinterpret_cast<float4*>(c_h2);
    }
}
```

**预期收益**: 带宽利用率 ×4（从 4B/线程 → 16B/线程）

---

## 11. 当前代码的改进空间总结

| 改进点 | 当前状态 | 改进方案 | 预期收益 |
|-------|---------|---------|---------|
| Grid 配置 | 线性增长 | Grid-Stride Loop | 小数据性能 +10~30% |
| F16/BF16 宽度 | half2 (4B) | float4 (16B) | 带宽利用率 +4x |
| 指针标注 | 无 `__restrict__` | 添加标注 | 编译器优化 |
| Kernel Fusion | 无 | 与后续 op 融合 | 减少内存访问 |
