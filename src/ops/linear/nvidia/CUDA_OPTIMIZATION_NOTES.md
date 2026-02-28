# Linear 算子 CUDA 优化学习笔记

## 1. 算子概述

**公式**: `out = in @ weight.T + bias`

**形状**:
- in: `[M, K]` — M 行 × K 列（rows × in_features）
- weight: `[N, K]` — N 行 × K 列（out_features × in_features）
- bias: `[N]` — 可选偏置
- out: `[M, N]` — M 行 × N 列

**本质**: **矩阵乘法（GEMM）**— 这是深度学习中计算量最大的操作，Transformer 的大部分 FLOPS 都在这里。

**实现选择**: 使用 **cuBLAS** 库而非手写 kernel。

---

## 2. 为什么用 cuBLAS 而非手写 GEMM？

| 角度 | cuBLAS | 手写 |
|------|--------|------|
| 性能 | 接近理论峰值（NVIDIA 工程师持续优化） | 很难超过 cuBLAS |
| 工作量 | 几行 API 调用 | 数百行复杂代码 |
| 硬件适配 | 自动选择最优 kernel（Tensor Core / CUDA Core） | 需要手动适配不同架构 |
| 精度选项 | 支持 F32/F16/BF16/INT8/FP8 Tensor Core | 每种都要写 |
| 实际工程 | PyTorch/TensorRT 底层都用 cuBLAS | 学习目的可以手写 |

**结论**: Linear 用 cuBLAS 是最佳实践。手写 GEMM 是面试/学习的好练习，但生产代码用库。

---

## 3. cuBLAS 行主序适配（最关键的知识点）

### 问题
cuBLAS 的所有矩阵都假设 **列主序（Column-Major）** 存储，但我们的数据是 **行主序（Row-Major）**。

### 关键洞察

行主序矩阵 A`[M,K]` 在内存中的布局 = 列主序矩阵 A^T`[K,M]` 的布局。

也就是说：
> **cuBLAS 看到的行主序 A`[M,K]` 就是列主序的 A^T`[K,M]`**

### 推导

我们需要计算: `out[M,N] = in[M,K] × weight^T[K,N]`

1. cuBLAS 看到的 `in` 指针 → 列主序 `in^T[K,M]`
2. cuBLAS 看到的 `weight` 指针 → 列主序 `weight^T[K,N]`
3. cuBLAS 看到的 `out` 指针 → 列主序 `out^T[N,M]`

我们需要让 cuBLAS 算出 `out^T[N,M]`：

```
out^T[N,M] = weight[N,K] × in^T[K,M]
```

cuBLAS 拿到的是 `weight^T[K,N]`，需要转置 → `transa = CUBLAS_OP_T`
cuBLAS 拿到的是 `in^T[K,M]`，不需要转置 → `transb = CUBLAS_OP_N`

### 最终调用

```cuda
cublasSgemm(
    handle,
    CUBLAS_OP_T,                    // transa: 转置 weight
    CUBLAS_OP_N,                    // transb: in 不转置
    N,                              // m: 结果的行数（列主序视角）
    M,                              // n: 结果的列数
    K,                              // k: 内积维度
    &alpha,                         // alpha = 1.0
    weight, K,                      // A = weight, lda = K（转置前的行数）
    in, K,                          // B = in, ldb = K
    &beta,                          // beta = 0.0
    out, N                          // C = out, ldc = N
);
```

**记忆口诀**: 行主序 `Y = X @ W^T` → cuBLAS `(OP_T, OP_N, N, M, K, W, K, X, K, Y, N)`

---

## 4. 混合精度 GEMM

### F16: `cublasGemmEx`

```cuda
cublasGemmEx(
    handle, CUBLAS_OP_T, CUBLAS_OP_N,
    N, M, K,
    &alpha,                         // float alpha（不是 half！）
    weight, CUDA_R_16F, K,          // A 类型
    in, CUDA_R_16F, K,              // B 类型
    &beta,
    out, CUDA_R_16F, N,             // C 类型
    CUBLAS_COMPUTE_32F,             // 累加精度：F32
    CUBLAS_GEMM_DEFAULT             // 算法选择
);
```

**为什么用 `CUBLAS_COMPUTE_32F` 而非 `CUBLAS_COMPUTE_16F`？**
- Tensor Core 的 FP16 乘法天然产生 FP32 中间结果
- F32 累加避免大矩阵的精度损失（多次 FP16 加法误差会累积）
- 与 PyTorch 默认行为一致

### BF16: 同样用 `CUBLAS_COMPUTE_32F`

```cuda
// BF16 输入输出，F32 内部累加
weight, CUDA_R_16BF, K,
in, CUDA_R_16BF, K,
out, CUDA_R_16BF, N,
CUBLAS_COMPUTE_32F,
```

---

## 5. Bias 加法优化：float4 向量化

GEMM 后需要广播加 bias：`out[i,j] += bias[j]`

### 基线版本（标量访存）

```cuda
__global__ void add_bias_f32(float* out, const float* bias, size_t rows, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * N;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        out[idx] += bias[idx % N];  // 按列广播
    }
}
```

### ncu 基线数据 (512×4096, F32)

| 指标 | F32 | F16 |
|------|-----|-----|
| Grid | 8192 | 8192 |
| Duration | 41.89µs | 25.25µs |
| DRAM Throughput | 92.06% | 61.41% |
| Memory BW | 249.91 GB/s | 166.59 GB/s |
| Compute (SM) | 29.92% | 49.75% |

**F32 观察**: 已接近硬件峰值（92% DRAM），但 Compute 29.92% 说明标量指令数较多。
**F16 瓶颈**: DRAM 只有 61%，Compute 却达 49.75% — 标量 `__half` 操作指令密集，成为了指令吞吐瓶颈。

### 优化：float4 (128-bit) 向量化

```cuda
// F32: float4 = 4 个 float
__global__ void add_bias_f32(float* out, const float* bias, size_t rows, size_t N) {
    const size_t vec_cols = N / 4;
    float4* out4  = reinterpret_cast<float4*>(out);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    for (size_t i = ...; i < rows * vec_cols; ...) {
        float4 o = out4[i], b = bias4[i % vec_cols];
        o.x += b.x; o.y += b.y; o.z += b.z; o.w += b.w;
        out4[i] = o;
    }
}

// F16: float4 = 8 个 __half，用 __half2 SIMD
float4 o = out4[i], b = bias4[i % vec_cols];
__half2* oh = reinterpret_cast<__half2*>(&o);
const __half2* bh = reinterpret_cast<const __half2*>(&b);
for (int k = 0; k < 4; ++k) oh[k] = __hadd2(oh[k], bh[k]);
```

**核心技巧**:
- F32: 每次 `float4` 读写 = 128-bit = 4 个 float → 指令减少 4x
- F16: 每次 `float4` = 128-bit = 8 个 `__half`，用 `__half2` SIMD 加法 → 指令减少 8x
- BF16: 同 F16，用 `__nv_bfloat162` SIMD
- 小尺寸（N 不能被 VEC 整除）自动回退到标量 kernel

### ncu 优化后数据

| 指标 | F32 基线 | F32 优化 | F16 基线 | F16 优化 |
|------|---------|---------|---------|---------|
| Grid | 8192 | 2048 | 8192 | 1024 |
| Duration | 41.89µs | 45.60µs | 25.25µs | **17.89µs** |
| DRAM | 92.06% | 93.86% | 61.41% | **86.92%** |
| Memory BW | 249.91 GB/s | 254.96 GB/s | 166.59 GB/s | **235.24 GB/s** |
| Compute | 29.92% | **7.54%** | 49.75% | **9.75%** |

**结果分析**:
- **F32**: 已处于带宽极限（92→94%），向量化消除了 3/4 指令（Compute 30%→8%），但时间无明显变化（带宽瓶就在上限）
- **F16**: 大幅改善！DRAM 61→87%（+42%），时间 25.25→17.89µs（**-29%**），__half2 SIMD 解决了标量指令瓶颈
- **教训**: 向量化对已经带宽饱和的 F32 帮助不大，但对指令吞吐受限的 F16 效果显著

### Bias 融合的替代方案

可以用 `cublasSgeam` 或把 bias 预扩展为 `[M, N]` 矩阵在 GEMM 中用 `beta=1` 一步完成。但对于 LLM 推理场景（bias_add 仅占总时间的 ~3%），单独 kernel 的开销可以忽略。

---

## 6. cuBLAS Handle 管理

```cuda
// 全局唯一 handle，懒初始化，线程安全
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        cublasCreate(&handle);
    });
    return handle;
}
```

**为什么用全局 handle？**
- cuBLAS handle 创建开销大（分配 GPU 端工作空间）
- 复用同一个 handle 可以让 cuBLAS 缓存最优算法选择
- `std::call_once` 保证多线程安全

---

## 7. 性能分析

### GEMM 的计算复杂度

```
FLOPs = 2 × M × N × K（乘加各一次）

例如 (512, 4096) × (4096, 4096)^T = (512, 4096):
  FLOPs = 2 × 512 × 4096 × 4096 ≈ 17.2 GFLOP
  
RTX 4070 Laptop FP32 峰值: ~23 TFLOPS
理论最优时间: 17.2G / 23T ≈ 0.75 ms
```

### ncu 实测 cuBLAS GEMM 数据 (512×4096, F32)

| 指标 | 数值 |
|------|------|
| Kernel 名称 | `ampere_sgemm_128x64_tn` |
| Grid | (32, 8, 7) |
| Block | 128 |
| Duration | 1.51 ms |
| Compute (SM) | 73.78% |
| FMA Pipeline | 61.5% (highest-utilized) |
| DRAM Throughput | 20.78% |
| L2 Cache Throughput | 68.96% |
| FP32 Peak | 61% |

**分析**:
- cuBLAS 选择了 `ampere_sgemm_128x64_tn` kernel（128×64 tile, TN 布局）
- Compute 与 Memory 平衡良好（73.78% vs 68.96%）
- 达到 FP32 理论峰值的 61% — 对于面向消费级 GPU 这是很好的成绩
- DRAM 只有 20.78% 是因为 cuBLAS 充分利用了 L2 cache 做数据复用（GEMM 特有的 blocking/tiling 策略）
- **结论**: cuBLAS GEMM 已接近最优，没有手动优化空间

### Tensor Core 加速

| 架构 | FP32 TFLOPS | FP16 Tensor Core TFLOPS | 加速比 |
|------|-------------|------------------------|--------|
| RTX 4070 Laptop (Ada) | ~23 | ~184 | 8x |

使用 FP16 + Tensor Core 可以获得巨大的加速！cuBLAS 的 `cublasGemmEx` 会自动利用 Tensor Core。

---

## 8. 学到的 CUDA 概念

| 概念 | 说明 |
|------|------|
| cuBLAS | NVIDIA 的 BLAS 库，GEMM 性能接近理论峰值 |
| 列主序 vs 行主序 | cuBLAS 用列主序，通过转置恒等式适配行主序 |
| `cublasSgemm` | F32 GEMM |
| `cublasGemmEx` | 泛型 GEMM，支持混合精度和 Tensor Core |
| `CUBLAS_COMPUTE_32F` | 用 F32 精度做中间累加 |
| Tensor Core | NVIDIA 专用矩阵运算硬件，每周期完成 4×4 矩阵乘 |
| cublasHandle_t | cuBLAS 上下文，管理内部状态和工作空间 |
| `std::once_flag` | C++ 线程安全的一次性初始化原语 |
| float4 向量化 | 128-bit 批量读写，减少指令数量和内存事务数 |
| `__half2` SIMD | 两个 FP16 值打包执行同一条算术指令 |
| `__nv_bfloat162` | 两个 BF16 值打包执行 SIMD 运算 |
| 标量 vs 向量化瓶颈 | F32 已带宽饱和时向量化无提升；F16 标量指令受限时向量化大幅提升 |
