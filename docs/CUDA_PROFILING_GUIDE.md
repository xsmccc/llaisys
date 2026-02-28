# CUDA 算子性能分析指南

本指南介绍如何对 llaisys 项目中的 CUDA 算子进行性能分析。

## 目录
1. [快速性能测试](#1-快速性能测试)
2. [内置 Benchmark](#2-内置-benchmark)
3. [使用 nsys 进行时间线分析](#3-使用-nsys-进行时间线分析)
4. [使用 ncu 进行 Kernel 级分析](#4-使用-ncu-进行-kernel-级分析)
5. [性能指标解读](#5-性能指标解读)

---

## 1. 快速性能测试

### 使用测试脚本的 --profile 选项

最简单的方式是使用测试脚本内置的 profile 功能：

```bash
cd /home/xsmccc/llaisys/llaisys
source venv/bin/activate

# 对 add 算子进行 profile
PYTHONPATH=python:test:$PYTHONPATH python test/ops/add.py --device nvidia --profile
```

**输出示例：**
```
Testing Ops.add on nvidia
   shape (2, 3) dtype <f32>
        Torch time: 0.01234 ms 
        LLAISYS time: 0.01056 ms
   shape (2, 3) dtype <f16>
        Torch time: 0.01189 ms 
        LLAISYS time: 0.00987 ms
```

---

## 2. 内置 Benchmark

### 使用 benchmark_add.py 进行详细对比

```bash
# 默认测试所有数据类型
python scripts/benchmark_add.py

# 只测试特定数据类型
python scripts/benchmark_add.py --dtypes f32 f16

# 自定义张量大小
python scripts/benchmark_add.py --shape 1024 4096

# 调整测试参数
python scripts/benchmark_add.py --warmup 100 --iterations 2000
```

**输出示例：**
```
======================================================================
Add Operator Benchmark
Shape: (512, 4096) (2,097,152 elements)
Warmup: 50, Iterations: 1000
======================================================================
DType      Time (ms)       Bandwidth (GB/s)     Speedup   
----------------------------------------------------------------------
f32        0.015294        1532.43              1.00      x
f16        0.011205        1045.81              1.36      x
bf16       0.012374        947.07               1.24      x
======================================================================
```

---

## 3. 使用 nsys 进行时间线分析

### Nsight Systems 适合：
- 查看整体执行时间线
- 发现 CPU-GPU 同步瓶颈
- 分析 kernel 启动开销
- 查看内存拷贝操作

### 基础使用

```bash
# 1. 生成 profile 数据
nsys profile --stats=true \
     -o add_profile_f32 \
     python scripts/profile_add_nsys.py --dtype f32

# 2. 查看统计信息
nsys stats add_profile_f32.nsys-rep

# 3. 使用 GUI 查看（需要图形界面）
nsys-ui add_profile_f32.nsys-rep
```

### 高级选项

```bash
# 只收集 CUDA API 和 kernel 信息
nsys profile --trace=cuda,nvtx \
     -o add_f16_trace \
     python scripts/profile_add_nsys.py --dtype f16 --iterations 5000

# 只分析特定时间段（从第2秒开始，持续3秒）
nsys profile --delay=2 --duration=3 \
     -o add_profile \
     python scripts/profile_add_nsys.py --dtype f32
```

### 对比不同数据类型

```bash
# 分别生成三种数据类型的 profile
for dtype in f32 f16 bf16; do
    nsys profile --stats=true \
         -o add_${dtype} \
         python scripts/profile_add_nsys.py --dtype $dtype --iterations 1000
    nsys stats add_${dtype}.nsys-rep > add_${dtype}_stats.txt
done

# 对比结果
grep "CUDA Kernel Statistics" add_*_stats.txt
```

---

## 4. 使用 ncu 进行 Kernel 级分析

### Nsight Compute 适合：
- 详细的 kernel 性能指标
- 内存访问模式分析
- 寄存器使用情况
- warp 执行效率
- 瓶颈识别（内存 vs 计算）

### 基础分析

```bash
# 1. 完整分析（包含所有指标，较慢）
ncu --set full \
    -o add_kernel_f32 \
    python scripts/profile_add_ncu.py --dtype f32

# 2. 查看报告
ncu --import add_kernel_f32.ncu-rep

# 3. 使用 GUI（需要图形界面）
ncu-ui add_kernel_f32.ncu-rep
```

### 快速分析（只看关键指标）

```bash
# 只看内存和计算吞吐
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
    python scripts/profile_add_ncu.py --dtype f16

# 只看内存合并访问
ncu --metrics \
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    python scripts/profile_add_ncu.py --dtype f32
```

### 对比不同 kernel

```bash
# 分析所有数据类型的 kernel
for dtype in f32 f16 bf16; do
    echo "Profiling $dtype kernel..."
    ncu --set full \
        -o add_kernel_${dtype} \
        python scripts/profile_add_ncu.py --dtype $dtype --iterations 50
done

# 生成对比报告
ncu --import add_kernel_f32.ncu-rep add_kernel_f16.ncu-rep add_kernel_bf16.ncu-rep
```

### 常用指标说明

```bash
# 内存带宽分析
ncu --query-metrics | grep dram

# 推荐的内存带宽指标
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python scripts/profile_add_ncu.py --dtype f16

# SM 利用率分析
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__average_warps_issue_stalled_no_instructions.pct \
    python scripts/profile_add_ncu.py --dtype f32
```

---

## 5. 性能指标解读

### Add 算子性能特征

Add 算子是**内存受限（memory-bound）**操作：
- 计算量：1 次加法/元素
- 内存访问：3 次（读 a, 读 b, 写 c）
- 算术强度：1/12 flops/byte (F32) 或 1/6 flops/byte (F16)

### 关键指标

#### 1. **时间 (Latency)**
```
目标：< 0.02 ms (对于 512×4096 的张量)
```

#### 2. **带宽利用率**
```
理论峰值：取决于 GPU 型号
- RTX 3090: ~936 GB/s
- A100 (40GB): ~1555 GB/s
- H100: ~2000+ GB/s

Add 算子带宽计算：
bandwidth = (numel × 3 × dtype_size) / time

良好性能：达到理论峰值的 60-80%
```

#### 3. **向量化效果**
```
期望加速比：
- F32 (float4): ~2-3x vs scalar
- F16 (half2): ~1.5-2x vs scalar
- BF16 (bfloat162): ~1.5-2x vs scalar
```

#### 4. **内存合并访问效率**
```
ncu 指标：
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld

目标：> 80%
说明：越接近 100%，说明内存访问越规整（coalesced）
```

### 性能优化检查清单

- [ ] **带宽利用率** > 60%
- [ ] **内存合并访问** > 80%
- [ ] **向量化实现** (F32用float4, F16用half2, BF16用bfloat162)
- [ ] **边界处理高效** (处理非对齐尺寸)
- [ ] **SM利用率合理** (对于内存受限操作，可能只有 20-40%)

### 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 带宽利用率低 (< 40%) | 1. 未向量化<br>2. 数据太小<br>3. 启动开销大 | 1. 使用向量化指令<br>2. 增大测试规模<br>3. 批处理多个操作 |
| 内存合并效率低 | 1. 非连续访问<br>2. 未对齐访问 | 1. 确保连续内存布局<br>2. 使用对齐的数据 |
| F16/BF16 未加速 | 1. 未用向量化<br>2. 类型转换开销 | 1. 使用 half2/bfloat162<br>2. 避免多余转换 |
| 时间不稳定 | 1. GPU未热身<br>2. 频率调节 | 1. 增加warmup次数<br>2. 锁定GPU频率 |

---

## 附录：脚本参数说明

### benchmark_add.py
```bash
python scripts/benchmark_add.py \
    --dtypes f32 f16 bf16 \    # 测试的数据类型
    --shape 512 4096 \          # 张量形状
    --warmup 50 \               # 预热次数
    --iterations 1000           # 测试迭代次数
```

### profile_add_nsys.py
```bash
python scripts/profile_add_nsys.py \
    --dtype f32 \               # 数据类型
    --shape 512 4096 \          # 张量形状
    --warmup 10 \               # 预热次数
    --iterations 1000           # 测试迭代次数
```

### profile_add_ncu.py
```bash
python scripts/profile_add_ncu.py \
    --dtype f16 \               # 数据类型
    --shape 512 4096 \          # 张量形状
    --iterations 100            # kernel 启动次数
```

---

## 参考资料

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
