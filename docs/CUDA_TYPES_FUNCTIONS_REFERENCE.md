# CUDA 类型和函数参考手册

本文档整理了 `add_nvidia.cu` 中使用的所有 CUDA 特定类型和函数的详细说明。

---

## 目录
1. [CUDA 修饰符](#1-cuda-修饰符)
2. [CUDA 向量类型](#2-cuda-向量类型)
3. [半精度浮点类型 (FP16)](#3-半精度浮点类型-fp16)
4. [BFloat16 类型](#4-bfloat16-类型)
5. [CUDA 内置函数](#5-cuda-内置函数)
6. [类型转换技术](#6-类型转换技术)
7. [线程索引变量](#7-线程索引变量)

---

## 1. CUDA 修饰符

### `__device__`
```cpp
__device__ void my_func() { ... }
```
- **作用**：声明函数只能在 GPU 设备上调用
- **调用限制**：只能从其他 `__device__` 或 `__global__` 函数调用，不能从 CPU 调用
- **使用场景**：GPU 端的辅助函数、内联函数

### `__global__`
```cpp
__global__ void my_kernel(...) { ... }
```
- **作用**：声明 CUDA kernel 函数
- **调用方式**：从 CPU 端使用 `<<<grid, block>>>` 语法启动
- **返回值**：必须是 `void`

### `__forceinline__`
```cpp
__device__ __forceinline__ float my_func() { ... }
```
- **作用**：强制编译器内联该函数
- **优点**：消除函数调用开销，提高性能
- **使用场景**：频繁调用的小函数、类型转换函数

---

## 2. CUDA 向量类型

### `float4`
```cpp
float4 a4;
a4.x = 1.0f;  // 第1个元素
a4.y = 2.0f;  // 第2个元素
a4.z = 3.0f;  // 第3个元素
a4.w = 4.0f;  // 第4个元素
```
- **定义**：包含 4 个 float 的向量类型
- **大小**：16 字节（4 × 4 字节）
- **对齐**：16 字节对齐
- **用途**：向量化内存访问，一次读写 4 个 float
- **其他类型**：`float2`, `float3` 也可用（分别包含 2、3 个元素）

**使用示例**：
```cpp
// 向量化读取
float4 data = *reinterpret_cast<const float4 *>(ptr);

// 向量化运算
float4 result;
result.x = data.x + 1.0f;
result.y = data.y + 1.0f;
result.z = data.z + 1.0f;
result.w = data.w + 1.0f;

// 向量化写回
*reinterpret_cast<float4 *>(output_ptr) = result;
```

---

## 3. 半精度浮点类型 (FP16)

### `__half`
```cpp
__half h = __float2half(1.5f);
```
- **定义**：CUDA 的半精度浮点类型（16位）
- **头文件**：`<cuda_fp16.h>`
- **精度**：1 符号位 + 5 指数位 + 10 尾数位
- **范围**：约 ±65504，最小正数 ~6e-8
- **用途**：节省内存、提高带宽，适合深度学习

**注意事项**：
- 某些 GPU 架构对 FP16 有硬件支持（Tensor Cores）
- 计算精度较低，可能需要混合精度训练

### `__half2`
```cpp
__half2 h2 = __halves2half2(__float2half(1.0f), __float2half(2.0f));
```
- **定义**：包含 2 个 `__half` 的向量类型
- **大小**：4 字节（2 × 2 字节）
- **用途**：向量化 FP16 运算，一条指令同时处理 2 个半精度数
- **性能**：比标量 `__half` 运算快约 2 倍

**向量化访问示例**：
```cpp
// 直接读取两个连续的半精度数为 __half2
__half2 data = *reinterpret_cast<const __half2 *>(fp16_ptr);

// 向量化加法（一条指令）
__half2 result = __hadd2(data, other);

// 写回
*reinterpret_cast<__half2 *>(output_ptr) = result;
```

---

## 4. BFloat16 类型

### `__nv_bfloat16`
```cpp
__nv_bfloat16 bf = __float2bfloat16(3.14f);
```
- **定义**：NVIDIA 的 BFloat16 类型（16位）
- **头文件**：`<cuda_bf16.h>`
- **精度**：1 符号位 + 8 指数位 + 7 尾数位
- **特点**：
  - 与 FP32 有相同的指数范围
  - 尾数精度较低（7位 vs FP32的23位）
  - 转换成本低（直接截断 FP32）
- **用途**：深度学习训练，平衡精度和性能

**FP16 vs BF16 对比**：
| 类型 | 指数位 | 尾数位 | 范围 | 精度 |
|------|--------|--------|------|------|
| FP16 | 5 | 10 | ±65504 | 高 |
| BF16 | 8 | 7 | ±3.4e38 | 中 |
| FP32 | 8 | 23 | ±3.4e38 | 最高 |

### `__nv_bfloat162`
```cpp
__nv_bfloat162 bf2;
```
- **定义**：包含 2 个 `__nv_bfloat16` 的向量类型
- **大小**：4 字节
- **用途**：向量化 BF16 运算

---

## 5. CUDA 内置函数

### FP16 算术运算

#### `__hadd(__half a, __half b)`
```cpp
__half result = __hadd(a, b);  // a + b
```
- **作用**：半精度加法
- **返回**：`__half` 类型
- **硬件支持**：在支持的 GPU 上使用专用硬件单元

#### `__hadd2(__half2 a, __half2 b)`
```cpp
__half2 result = __hadd2(a2, b2);  // 同时计算两个加法
```
- **作用**：向量化半精度加法，一次计算 2 个
- **返回**：`__half2` 类型
- **性能**：一条 SIMD 指令完成
- **等价于**：
  ```cpp
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  ```

**其他 FP16 运算函数**：
```cpp
__hsub(a, b)      // a - b
__hmul(a, b)      // a * b
__hdiv(a, b)      // a / b
__hfma(a, b, c)   // a * b + c (融合乘加)

// 向量化版本
__hsub2(a2, b2)
__hmul2(a2, b2)
__hfma2(a2, b2, c2)
```

### FP16 类型转换

#### `__float2half(float x)`
```cpp
__half h = __float2half(3.14f);
```
- **作用**：将 float 转换为 `__half`
- **后缀 `_rn`**：round to nearest（四舍五入）
- **其他舍入模式**：
  - `__float2half_rz`：向零舍入
  - `__float2half_rd`：向下舍入
  - `__float2half_ru`：向上舍入

#### `__half2float(__half h)`
```cpp
float f = __half2float(h);
```
- **作用**：将 `__half` 转换为 float
- **精度**：无损转换（FP16 完全可以表示在 FP32 内）

### FP16 向量操作

#### `__halves2half2(__half a, __half b)`
```cpp
__half2 h2 = __halves2half2(h_low, h_high);
```
- **作用**：将两个 `__half` 打包成一个 `__half2`
- **参数**：
  - `a`：低位元素（`.x`）
  - `b`：高位元素（`.y`）

#### `__low2half(__half2 h2)`
```cpp
__half low = __low2half(h2);
```
- **作用**：提取 `__half2` 的低位元素（`.x`）

#### `__high2half(__half2 h2)`
```cpp
__half high = __high2half(h2);
```
- **作用**：提取 `__half2` 的高位元素（`.y`）

### BF16 算术运算

#### `__hadd(__nv_bfloat16 a, __nv_bfloat16 b)`
```cpp
__nv_bfloat16 result = __hadd(a, b);
```
- **作用**：BFloat16 加法（标量）
- **注意**：函数名与 FP16 相同，通过类型重载区分

#### `__hadd2(__nv_bfloat162 a, __nv_bfloat162 b)`
```cpp
__nv_bfloat162 result = __hadd2(a2, b2);
```
- **作用**：向量化 BF16 加法（2个元素）

### BF16 类型转换

#### `__float2bfloat16(float x)`
```cpp
__nv_bfloat16 bf = __float2bfloat16(1.5f);
```
- **作用**：float → BFloat16
- **实现**：通常是直接截断 FP32 的低 16 位

#### `__bfloat162float(__nv_bfloat16 bf)`
```cpp
float f = __bfloat162float(bf);
```
- **作用**：BFloat16 → float

---

## 6. 类型转换技术

### `reinterpret_cast`
```cpp
__half2 h2 = *reinterpret_cast<const __half2 *>(ptr);
```
- **作用**：在不改变内存位模式的情况下重新解释数据类型
- **使用场景**：
  1. 向量化访问（将连续内存解释为向量类型）
  2. 自定义类型与 CUDA 类型互转
- **注意事项**：
  - 必须确保内存对齐
  - 不进行任何数值转换
  - 需要理解底层内存布局

**示例：向量化读取**
```cpp
// 原始指针指向连续的 4 个 float
float *ptr = ...;

// 重新解释为 float4 指针，一次读取 4 个元素
float4 vec = *reinterpret_cast<const float4 *>(ptr);

// 等价于：
float4 vec;
vec.x = ptr[0];
vec.y = ptr[1];
vec.z = ptr[2];
vec.w = ptr[3];
```

**示例：自定义类型转换**
```cpp
// 假设 llaisys::fp16_t 内部存储为 uint16_t _v
__device__ __half to_cuda_half(llaisys::fp16_t v) {
    // 直接将 uint16_t 的位模式重新解释为 __half
    return *reinterpret_cast<const __half *>(&v._v);
}
```

---

## 7. 线程索引变量

### 内置变量

#### `blockIdx.x`
- **类型**：`uint3`（3维向量，通常只用 `.x`）
- **含义**：当前线程所在的 block 编号（grid 维度）
- **范围**：`0` 到 `gridDim.x - 1`

#### `blockDim.x`
- **类型**：`dim3`（3维向量）
- **含义**：每个 block 包含的线程数
- **示例**：如果启动时指定 `256` 个线程/block，则 `blockDim.x = 256`

#### `threadIdx.x`
- **类型**：`uint3`
- **含义**：线程在 block 内的编号
- **范围**：`0` 到 `blockDim.x - 1`

### 全局线程索引计算

```cpp
size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**理解方式**：
```
Grid:     [Block 0] [Block 1] [Block 2] ...
          ↓         ↓         ↓
Threads:  0-255     256-511   512-767   ...

对于 Block 1 中的 Thread 5：
  idx = 1 * 256 + 5 = 261
```

**可视化示例**：
```
假设：blockDim.x = 256, gridDim.x = 4
总共线程数：256 * 4 = 1024

Block 0: threadIdx [0-255]   → idx [0-255]
Block 1: threadIdx [0-255]   → idx [256-511]
Block 2: threadIdx [0-255]   → idx [512-767]
Block 3: threadIdx [0-255]   → idx [768-1023]
```

---

## 8. 代码中的具体应用

### 示例 1：F32 向量化加法

```cpp
__global__ void add_kernel_f32_vec(float *c, const float *a, const float *b, size_t numel) {
    // 1. 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 每个线程处理 4 个元素
    size_t base = idx * 4;

    if (base + 3 < numel) {
        // 3. 向量化读取：一次读 4 个 float (16 字节)
        float4 a4 = *reinterpret_cast<const float4 *>(a + base);
        float4 b4 = *reinterpret_cast<const float4 *>(b + base);
        
        // 4. 向量运算（编译器可能使用 SIMD 指令）
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        
        // 5. 向量化写回：一次写 4 个 float
        *reinterpret_cast<float4 *>(c + base) = c4;
    }
}
```

**性能分析**：
- 内存访问次数：3 次（读 a, 读 b, 写 c）
- 每次访问 16 字节（4 个 float）
- 相比标量版本，减少了 4 倍的内存事务

### 示例 2：FP16 向量化加法

```cpp
__global__ void add_kernel_f16_vec(llaisys::fp16_t *c, const llaisys::fp16_t *a, 
                                    const llaisys::fp16_t *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 2;  // 每线程处理 2 个元素

    if (base + 1 < numel) {
        // 1. 直接读取为 __half2（4字节）
        __half2 a2 = *reinterpret_cast<const __half2 *>(a + base);
        __half2 b2 = *reinterpret_cast<const __half2 *>(b + base);
        
        // 2. 使用硬件指令同时计算 2 个加法
        __half2 c2 = __hadd2(a2, b2);
        
        // 3. 向量化写回
        *reinterpret_cast<__half2 *>(c + base) = c2;
    } 
    else if (base < numel) {
        // 标量路径：处理最后一个奇数元素
        __half ha = to_cuda_half(a[base]);
        __half hb = to_cuda_half(b[base]);
        c[base] = from_cuda_half(__hadd(ha, hb));
    }
}
```

**关键点**：
- `__hadd2` 是一条硬件指令，性能比两次 `__hadd` 好
- 需要处理元素个数为奇数的情况（最后一个元素用标量处理）

---

## 9. 常见问题

### Q1: 为什么使用 `reinterpret_cast` 而不是直接访问？

**A:** 向量化访问的优势：
```cpp
// 标量访问（慢）：4 次内存事务
float a0 = a[i];
float a1 = a[i+1];
float a2 = a[i+2];
float a3 = a[i+3];

// 向量化访问（快）：1 次内存事务
float4 a4 = *reinterpret_cast<const float4 *>(&a[i]);
```

### Q2: `__half` 和 `float` 性能差异？

**A:** 
- **内存带宽**：FP16 是 FP32 的 2 倍（相同带宽下传输 2 倍数据）
- **计算速度**：取决于 GPU 架构
  - 有 Tensor Cores：FP16 显著更快
  - 无硬件支持：FP16 可能更慢（需要模拟）

### Q3: 什么时候用 BF16 而不是 FP16？

**A:**
- **BF16 优势**：
  - 与 FP32 有相同的数值范围（指数位相同）
  - 转换成本低
  - 训练稳定性更好
- **FP16 优势**：
  - 精度更高（尾数位多）
  - 推理场景更优

---

## 10. 参考资料

### 官方文档
- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- [CUDA C++ Programming Guide - Half Precision](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

### 头文件位置
```bash
# 安装 CUDA 后查看
ls /usr/local/cuda/include/cuda_fp16.h
ls /usr/local/cuda/include/cuda_bf16.h
```

### 快速查询命令
```bash
# 查看所有 FP16 函数
grep -r "__half" /usr/local/cuda/include/cuda_fp16.h | grep "inline"

# 查看所有 BF16 函数  
grep -r "__nv_bfloat16" /usr/local/cuda/include/cuda_bf16.h | grep "inline"
```

---

## 总结速查表

| 类型/函数 | 用途 | 性能特点 |
|----------|------|---------|
| `float4` | 4个float向量 | 一次访问16字节 |
| `__half` | 半精度浮点 | 节省50%内存 |
| `__half2` | 2个half向量 | 一条指令处理2个 |
| `__nv_bfloat16` | BFloat16标量 | FP32范围，低精度 |
| `__nv_bfloat162` | 2个BF16向量 | 向量化BF16运算 |
| `__hadd(a,b)` | 标量加法 | FP16/BF16通用 |
| `__hadd2(a2,b2)` | 向量加法 | 2倍吞吐量 |
| `reinterpret_cast` | 类型重解释 | 零成本转换 |
| `blockIdx.x` | Block编号 | 计算全局索引 |
| `threadIdx.x` | 线程编号 | 计算全局索引 |
