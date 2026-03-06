/**
 * @file swiglu_nvidia.cu
 * @brief SwiGLU 算子的 CUDA 实现
 *
 * ═══════════════════════════════════════════════════════════════
 *  SwiGLU 公式
 * ═══════════════════════════════════════════════════════════════
 *
 *   out = up ⊙ SiLU(gate)
 *   SiLU(x) = x / (1 + exp(-x)) = x · σ(x)
 *
 *   其中 gate 和 up 是两个同形状的张量（来自 FFN 的两个线性投影），
 *   ⊙ 是逐元素乘法（Hadamard product），σ(x) 是 Sigmoid 函数。
 *
 * ═══════════════════════════════════════════════════════════════
 *  算术强度分析（Arithmetic Intensity）
 * ═══════════════════════════════════════════════════════════════
 *
 *   每个元素的计算量：
 *     - exp(-x)      : 1 FLOP（特殊函数，实际硬件代价 ~20 cycle，但 FLOP 算 1）
 *     - 1 + exp(-x)  : 1 FLOP
 *     - x / (...)    : 1 FLOP
 *     - up * silu_val : 1 FLOP
 *     合计: ~4 FLOP / element
 *
 *   每个元素的访存量（F32）：
 *     - 读 gate[i]   : 4 B
 *     - 读 up[i]     : 4 B
 *     - 写 out[i]    : 4 B
 *     合计: 12 B / element
 *
 *   算术强度 = 4 / 12 ≈ 0.33 FLOP/B
 *
 *   RTX 4070 Laptop 的 Ridge Point ≈ 29.15 TF / 256 GB/s ≈ 113.87 FLOP/B
 *   0.33 << 113.87  →  **Memory-Bound 算子**
 *
 *   虽然 SwiGLU 比 Add（0.08 FLOP/B）算术强度高 4 倍，
 *   但仍然远低于 Ridge Point，瓶颈依然在内存带宽。
 *
 * ═══════════════════════════════════════════════════════════════
 *  优化迭代历史
 * ═══════════════════════════════════════════════════════════════
 *
 *   版本 0: Naive（每线程 1 个元素）
 *     - ncu: Duration = 68.86 µs, DRAM = 95.32%
 *     - 已经跑满了 DRAM 带宽，但指令效率低
 *
 *   版本 1: Naive + float4（错误尝试）
 *     - 想法：float4 让每线程处理 4 元素 → Grid 缩小 4×
 *     - ncu: Duration = 96.35 µs ← 反而变慢了！
 *     - 根因：Grid 从 8192 → 2048，Waves Per SM 从 37.93 → 9.48
 *       SM 没有足够的 warp 来隐藏内存延迟（latency hiding 不足）
 *
 *   版本 2: Grid-Stride Loop + float4（当前 F32 版本）
 *     - 固定 Grid = SM 数 × 8，保证 Occupancy
 *     - 每线程用 float4 向量化，但通过 stride 循环覆盖所有数据
 *     - ncu: Duration = 69.02 µs, DRAM ≈ 95%+ → 性能恢复且指令更高效
 *
 *   教训：向量化 ≠ 减小 Grid。必须保持足够的 warp 数来隐藏延迟。
 *
 * ═══════════════════════════════════════════════════════════════
 *  代码结构
 * ═══════════════════════════════════════════════════════════════
 *
 *   1. 类型转换辅助函数（LLAISYS 自定义类型 ↔ CUDA 内置类型）
 *   2. silu_f32()：SiLU 激活函数（float 精度）
 *   3. Naive 版本 Kernel（仅供学习对比，未在生产路径中调用）
 *   4. Grid-Stride + 向量化 Kernel（当前使用）
 *      - F32: float4（每线程 4 × 4B = 16B = LD.128）
 *      - F16: half2 （每线程 2 × 2B = 4B  = LD.32）← 有优化空间
 *      - BF16: bfloat162（每线程 2 × 2B = 4B = LD.32）← 有优化空间
 *   5. launch_swiglu_kernel()：Grid/Block 配置 + dispatch
 *   6. 对外接口
 */

#include "swiglu_nvidia.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <iostream>

namespace {

// ============ 错误检查 ============
// cudaError_t 是 CUDA 所有 API 的返回码类型，cudaSuccess 表示成功
// 在 kernel launch 后必须调用 cudaGetLastError() 检查异步错误
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============ 类型转换辅助函数 ============
// LLAISYS 框架定义了 fp16_t 和 bf16_t 自定义类型，内部存储用 uint16_t
// CUDA 内置类型为 __half 和 __nv_bfloat16
// 二者二进制格式完全相同，所以可以安全地 reinterpret_cast
//
// __device__：只能在 GPU 上执行的函数
// __forceinline__：强制内联，避免函数调用开销（寄存器压入/弹出）
// 对于这种简单的类型转换，内联后编译器不会生成任何额外指令

// LLAISYS fp16 → CUDA half
__device__ __forceinline__ __half to_cuda_half(llaisys::fp16_t v) {
    return *reinterpret_cast<const __half*>(&v._v);
}

// CUDA half → LLAISYS fp16
__device__ __forceinline__ llaisys::fp16_t from_cuda_half(__half h) {
    return *reinterpret_cast<const llaisys::fp16_t*>(&h);
}

// LLAISYS bf16 → CUDA bfloat16
__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(llaisys::bf16_t v) {
    return *reinterpret_cast<const __nv_bfloat16*>(&v._v);
}

// CUDA bfloat16 → LLAISYS bf16
__device__ __forceinline__ llaisys::bf16_t from_cuda_bfloat16(__nv_bfloat16 b) {
    return *reinterpret_cast<const llaisys::bf16_t*>(&b);
}

// ============ SiLU 激活函数 ============
//
// SiLU(x) = x / (1 + exp(-x)) = x · sigmoid(x)
//
// 为什么用 __expf 而不是 expf？
//   - __expf 是 CUDA 内置快速数学函数（intrinsic）
//   - 通过查表+插值实现，比标准 expf 快约 2-3×
//   - 精度：最大 ULP 误差 2（即最后 2 个有效位可能不同）
//   - 对于 AI 推理完全够用，训练也一般可以接受
//
// 为什么只写 float 版本？
//   - CUDA 没有 __expf 的 half/bfloat16 版本
//   - 半精度的 exp 范围太小（FP16 最大值 65504），容易溢出
//   - 所以 F16/BF16 kernel 统一上转 float 计算，结果再转回
//   - 这是 AI 框架的标准做法（PyTorch、TensorRT 都这样）
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║              NAIVE 版本 KERNELS（仅供学习对比）               ║
// ╚═══════════════════════════════════════════════════════════════╝
//
// 最简单的 CUDA 实现模式：1 thread → 1 element
//
// 线程映射方式：
//   idx = blockIdx.x × blockDim.x + threadIdx.x
//   if (idx < numel) → 处理 gate[idx], up[idx] → out[idx]
//
// Grid 大小 = ceil(numel / blockDim.x) 个 block
//   例如 numel = 2,097,152 (512×4096), blockDim = 256
//   → Grid = 8192 blocks
//
// 访存模式分析：
//   - 每个线程读 1 个 gate + 1 个 up = 2 × 4B = 8B（F32）
//   - 连续线程读连续地址 → Coalesced（合并访问）
//   - 一个 warp（32 线程）发起 1 次 128B 读取事务（刚好一个 cache line）
//   - 访存效率 100%，但是**每条 load 指令只搬 4B**
//
// 性能参考（512×4096, F32）：
//   Duration = 68.86 µs, DRAM Throughput = 95.32%
//   已经接近带宽上限，说明 Coalesced 访问足够高效
//
// 那为什么还需要优化？
//   → 指令层面：每个元素需要独立的 LD.32 指令
//   → 向量化后用 LD.128 可以 4 个元素只发 1 条指令
//   → 减少指令调度开销，在更大规模数据时收益更明显

/**
 * @brief F32 Naive Kernel
 *
 * 每个线程：1 次 LD.32 (gate) + 1 次 LD.32 (up) + SiLU 计算 + 1 次 ST.32 (out)
 * 总共 3 条内存指令 / element
 */
__global__ void swiglu_kernel_f32_naive(
    float* out,
    const float* gate,
    const float* up,
    size_t numel
) {
    // 全局线程 ID = 第几个 block × 每个 block 多少线程 + block 内的线程号
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查：numel 可能不是 blockDim 的整数倍，最后一个 block 会有多余线程
    if (idx < numel) {
        float g = gate[idx];              // LD.32：从全局内存加载 4B
        float u = up[idx];                // LD.32：从全局内存加载 4B
        float silu_val = silu_f32(g);     // 计算 SiLU(gate) ← 包含 __expf
        out[idx] = u * silu_val;          // ST.32：写回 4B 到全局内存
    }
}

/**
 * @brief F16 Naive Kernel
 *
 * 与 F32 流程相同，但多了类型转换的步骤：
 *   1. 从全局内存加载 fp16_t（2B）
 *   2. 转换为 CUDA __half 类型
 *   3. 上转为 float 进行 SiLU 计算（半精度下 exp 不安全）
 *   4. float 结果转回 __half
 *   5. 转回 fp16_t 写入全局内存
 *
 * 每个元素：LD.16 + 类型转换 + 计算 + 类型转换 + ST.16
 */
__global__ void swiglu_kernel_f16_naive(
    llaisys::fp16_t* out,
    const llaisys::fp16_t* gate,
    const llaisys::fp16_t* up,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        // 类型桥接：框架类型 → CUDA 类型 → float（计算精度）
        __half g_half = to_cuda_half(gate[idx]);    // LD.16 + reinterpret
        __half u_half = to_cuda_half(up[idx]);      // LD.16 + reinterpret
        
        // FP16 → FP32 提升计算精度（exp 在半精度下容易溢出/下溢）
        float g = __half2float(g_half);   // CUDA intrinsic: 1 cycle
        float u = __half2float(u_half);
        
        float silu_val = silu_f32(g);     // 在 float 精度下计算 SiLU
        float result = u * silu_val;
        
        // FP32 → FP16 并写回
        out[idx] = from_cuda_half(__float2half(result));  // ST.16
    }
}

/**
 * @brief BF16 Naive Kernel
 *
 * BF16 比 FP16 更需要上转 float：
 *   - FP16: 5 bit 指数 + 10 bit 尾数 → 精度尚可
 *   - BF16: 8 bit 指数 + 7 bit 尾数  → 范围大但精度低
 *   - BF16 的 7 位尾数意味着有效精度仅 ~2 个十进制位
 *   - 直接在 BF16 上做 exp 会有很大的数值误差
 */
__global__ void swiglu_kernel_bf16_naive(
    llaisys::bf16_t* out,
    const llaisys::bf16_t* gate,
    const llaisys::bf16_t* up,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        // 类型桥接：框架类型 → CUDA 类型 → float
        __nv_bfloat16 g_bf16 = to_cuda_bfloat16(gate[idx]);  // LD.16 + reinterpret
        __nv_bfloat16 u_bf16 = to_cuda_bfloat16(up[idx]);    // LD.16 + reinterpret
        
        // BF16 → FP32（BF16 只有 7 位尾数，必须上转）
        float g = __bfloat162float(g_bf16);
        float u = __bfloat162float(u_bf16);
        
        float silu_val = silu_f32(g);
        float result = u * silu_val;
        
        // FP32 → BF16 写回（截断低 16 位尾数）
        out[idx] = from_cuda_bfloat16(__float2bfloat16(result));  // ST.16
    }
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║     向量化优化版本 KERNELS（Grid-Stride Loop）               ║
// ╚═══════════════════════════════════════════════════════════════╝
//
// ── Grid-Stride Loop 模式详解 ─────────────────────────────────
//
//  传统模式：Grid 足够大 → 1 thread : 1 element → 一次循环都不需要
//    问题：向量化后 numel/4 个线程就够了，Grid 变小 → SM 利用率下降
//
//  Grid-Stride 模式：
//    - 固定 Grid 大小（通常 = SM 数 × 每 SM 的目标 block 数）
//    - 每个线程通过循环处理多个元素
//
//    图解（4 个线程，8 个数据块 → 每线程处理 2 个）：
//
//      数据: [A][B][C][D] [E][F][G][H]
//                                                   stride = 4
//      Thread0: [A]...............[E]    ← tid=0, 0+stride=4
//      Thread1: ....[B]...............[F] ← tid=1, 1+stride=5
//      Thread2: ........[C]...............[G]
//      Thread3: ............[D]...............[H]
//                第 1 轮                 第 2 轮
//
//    for (i = tid; i < num_vecs; i += stride) {
//        // 处理第 i 个 vec4
//    }
//
//  为什么用固定 Grid？
//    1. GPU SM 数量是固定的（4070L = 36 SM）
//    2. 每 SM 最多 48 个 warp = 1536 个线程
//    3. Grid 再大也不会增加硬件并行度，只增加调度开销
//    4. Grid-Stride 让每个线程多做几轮，但 warp 始终保持满负荷
//
//  关键参数：
//    BLOCKS = num_sm × 8 = 288
//    THREADS = 256 (= 8 warp)
//    总线程 = 288 × 256 = 73,728
//    每 SM 分配 = 288/36 = 8 block（但每 SM 最多同时跑 6 个）
//    多出的 2 个 block 排队等候，保证 SM 不会空闲
//
// ── 第一次 vec4 优化失败的教训 ─────────────────────────────────
//
//   错误做法：Grid = ceil(numel/4/256)  ← 直接把 Grid 缩小 4 倍
//     numel = 2,097,152 → Grid = 2,048 blocks
//     之前 naive Grid = 8,192 blocks
//     Waves Per SM 从 37.93 降到 9.48 ← 波次不足
//
//   正确做法（当前）：Grid 保持 288 blocks，用 stride 循环
//     numel = 2,097,152, vec4 个数 = 524,288
//     每个线程处理 = 524,288 / 73,728 ≈ 7.1 轮
//     每轮处理 4 个 float = LD.128 → 高效

/**
 * @brief F32 Grid-Stride + float4 向量化 Kernel
 *
 * 每个线程每步处理 4 个 float（16B），对应 PTX 指令 LD.E.128
 * 这是 F32 能达到的最大单线程访存粒度
 *
 * 数据流（每轮每线程）：
 *   LD.128 (gate) → 4 个 silu_f32 → LD.128 (up) → 4 个乘法 → ST.128 (out)
 *   访存 = 3 × 16B = 48B/轮
 *   计算 = 4 × (exp + div + mul) + 4 × mul = ~16 FLOP/轮
 */
__global__ void swiglu_kernel_f32_vec4(
    float* out,
    const float* gate,
    const float* up,
    size_t numel
) {
    // tid: 全局线程编号（在所有 block 的所有线程中的唯一 ID）
    // stride: 总线程数（grid 一共有多少个线程）
    // 单位是 "vec4 个"：每个线程每步跳 stride 个 vec4 位置
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // ── Grid-Stride 主循环 ──
    // 条件 (i + 1) * 4 <= numel 保证读取 4 个完整的 float 不会越界
    // i 的含义：第 i 个 vec4 组，对应的起始元素下标 = i × 4
    for (size_t i = tid; (i + 1) * 4 <= numel; i += stride) {
        size_t base = i * 4;  // 元素级偏移

        // reinterpret_cast<const float4*>：告诉编译器这里有一个 128 位对齐的指针
        // 编译器会生成 LD.E.128 指令，一次搬运 16B = 4 个 float
        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        // float4 是 struct { float x, y, z, w; }
        // 逐分量计算 SwiGLU
        float4 o4;
        o4.x = u4.x * silu_f32(g4.x);  // 每个 silu 内部：__expf + div + mul
        o4.y = u4.y * silu_f32(g4.y);
        o4.z = u4.z * silu_f32(g4.z);
        o4.w = u4.w * silu_f32(g4.w);

        // ST.E.128：一次写回 16B
        *reinterpret_cast<float4*>(out + base) = o4;
    }

    // ── 尾部标量处理 ──
    // numel 不一定是 4 的整数倍，剩余 1~3 个元素需要逐个处理
    // 只让 tid=0 的线程处理，避免多线程重复写入
    // 尾部元素极少（最多 3 个），性能影响可忽略
    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;  // 向下取整到 4 的倍数
        for (size_t i = tail_start; i < numel; ++i) {
            out[i] = up[i] * silu_f32(gate[i]);
        }
    }
}

/**
 * @brief F16 Grid-Stride + float4 向量化 Kernel（优化版）
 *
 * 优化前：half2 = 4B = LD.32，每线程处理 2 个 half
 * 优化后：float4 = 16B = LD.128，每线程处理 8 个 half
 *
 * ── 为什么能用 float4 加载 half 数据？──────────────────────
 *   float4 只是一个 16 字节的容器，里面的二进制内容可以任意解释。
 *   我们用 float4 发起 LD.128 指令搬运 16B，然后在寄存器中用
 *   reinterpret_cast 把每个 float（4B）重新解释为 __half2（两个 half）。
 *
 *   float4 { .x, .y, .z, .w } 每个分量 = 4B = 2 个 half
 *   → 一个 float4 = 4 × 2 = 8 个 half
 *
 * ── 数据流（每轮每线程）────────────────────────────────────
 *   LD.128 (gate)   → 获得 8 个 half  ← 1 条指令
 *   LD.128 (up)     → 获得 8 个 half  ← 1 条指令
 *   拆分 4 个 half2 → 8 个 float（上转）
 *   8 × silu_f32 + 8 × mul（在 float 精度下计算）
 *   8 × float → half（下转）→ 打包为 4 个 half2
 *   ST.128 (out)    → 写回 8 个 half  ← 1 条指令
 *
 *   优化前：每 2 元素需要 3 条 LD/ST.32 → 每 8 元素需要 12 条
 *   优化后：每 8 元素只需 3 条 LD/ST.128 → 指令数减少 4×
 */
__global__ void swiglu_kernel_f16_vec4(
    llaisys::fp16_t* __restrict__ out,
    const llaisys::fp16_t* __restrict__ gate,
    const llaisys::fp16_t* __restrict__ up,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // numel_vec: float4 向量的个数（每个 float4 = 8 个 half）
    size_t numel_vec = numel / 8;

    // ── Grid-Stride 主循环（float4 宽搬运）──
    for (size_t i = tid; i < numel_vec; i += stride) {
        size_t base = i * 8;  // 元素级偏移（8 个 half / float4）

        // LD.128: 一次加载 16B = 8 个 half
        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        // 把每个 float（4B）重新解释为 half2（2 个 half）
        // g4.x → 包含 gate[base+0], gate[base+1]
        // g4.y → 包含 gate[base+2], gate[base+3]
        // g4.z → 包含 gate[base+4], gate[base+5]
        // g4.w → 包含 gate[base+6], gate[base+7]
        __half2 g_01 = *reinterpret_cast<const __half2*>(&g4.x);
        __half2 g_23 = *reinterpret_cast<const __half2*>(&g4.y);
        __half2 g_45 = *reinterpret_cast<const __half2*>(&g4.z);
        __half2 g_67 = *reinterpret_cast<const __half2*>(&g4.w);

        __half2 u_01 = *reinterpret_cast<const __half2*>(&u4.x);
        __half2 u_23 = *reinterpret_cast<const __half2*>(&u4.y);
        __half2 u_45 = *reinterpret_cast<const __half2*>(&u4.z);
        __half2 u_67 = *reinterpret_cast<const __half2*>(&u4.w);

        // 上转 float，计算 SwiGLU，下转回 half2
        // 每个 half2 拆成 lo/hi → 2 个 float → silu → mul → 转回 half → 打包
        __half2 o_01 = __halves2half2(
            __float2half(__half2float(__low2half(u_01)) * silu_f32(__half2float(__low2half(g_01)))),//low
            __float2half(__half2float(__high2half(u_01)) * silu_f32(__half2float(__high2half(g_01)))));//high
        __half2 o_23 = __halves2half2(
            __float2half(__half2float(__low2half(u_23)) * silu_f32(__half2float(__low2half(g_23)))),
            __float2half(__half2float(__high2half(u_23)) * silu_f32(__half2float(__high2half(g_23)))));
        __half2 o_45 = __halves2half2(
            __float2half(__half2float(__low2half(u_45)) * silu_f32(__half2float(__low2half(g_45)))),
            __float2half(__half2float(__high2half(u_45)) * silu_f32(__half2float(__high2half(g_45)))));
        __half2 o_67 = __halves2half2(
            __float2half(__half2float(__low2half(u_67)) * silu_f32(__half2float(__low2half(g_67)))),
            __float2half(__half2float(__high2half(u_67)) * silu_f32(__half2float(__high2half(g_67)))));

        // 把 4 个 half2 打包回 float4，用 ST.128 写回
        float4 o4;
        *reinterpret_cast<__half2*>(&o4.x) = o_01;
        *reinterpret_cast<__half2*>(&o4.y) = o_23;
        *reinterpret_cast<__half2*>(&o4.z) = o_45;
        *reinterpret_cast<__half2*>(&o4.w) = o_67;

        // ST.128: 一次写回 16B = 8 个 half
        *reinterpret_cast<float4*>(out + base) = o4;
    }

    // ── 尾部处理（numel 不是 8 的倍数时，最多 7 个元素）──
    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __half2float(to_cuda_half(gate[i]));
            float u = __half2float(to_cuda_half(up[i]));
            out[i] = from_cuda_half(__float2half(u * silu_f32(g)));
        }
    }
}

/**
 * @brief BF16 Grid-Stride + float4 向量化 Kernel（优化版）
 *
 * 和 F16 完全相同的结构，只是类型换成 bfloat16。
 * 同样使用 float4 宽搬运：1 个 float4 = 16B = 8 个 bf16
 */
__global__ void swiglu_kernel_bf16_vec4(
    llaisys::bf16_t* __restrict__ out,
    const llaisys::bf16_t* __restrict__ gate,
    const llaisys::bf16_t* __restrict__ up,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t numel_vec = numel / 8;

    for (size_t i = tid; i < numel_vec; i += stride) {
        size_t base = i * 8;

        // LD.128: 加载 8 个 bf16
        float4 g4 = *reinterpret_cast<const float4*>(gate + base);
        float4 u4 = *reinterpret_cast<const float4*>(up   + base);

        // 重解释为 bfloat162
        __nv_bfloat162 g_01 = *reinterpret_cast<const __nv_bfloat162*>(&g4.x);
        __nv_bfloat162 g_23 = *reinterpret_cast<const __nv_bfloat162*>(&g4.y);
        __nv_bfloat162 g_45 = *reinterpret_cast<const __nv_bfloat162*>(&g4.z);
        __nv_bfloat162 g_67 = *reinterpret_cast<const __nv_bfloat162*>(&g4.w);

        __nv_bfloat162 u_01 = *reinterpret_cast<const __nv_bfloat162*>(&u4.x);
        __nv_bfloat162 u_23 = *reinterpret_cast<const __nv_bfloat162*>(&u4.y);
        __nv_bfloat162 u_45 = *reinterpret_cast<const __nv_bfloat162*>(&u4.z);
        __nv_bfloat162 u_67 = *reinterpret_cast<const __nv_bfloat162*>(&u4.w);

        // 上转 float，计算 SwiGLU，下转回 bfloat162
        __nv_bfloat162 o_01 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_01)) * silu_f32(__bfloat162float(__low2bfloat16(g_01)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_01)) * silu_f32(__bfloat162float(__high2bfloat16(g_01)))));
        __nv_bfloat162 o_23 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_23)) * silu_f32(__bfloat162float(__low2bfloat16(g_23)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_23)) * silu_f32(__bfloat162float(__high2bfloat16(g_23)))));
        __nv_bfloat162 o_45 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_45)) * silu_f32(__bfloat162float(__low2bfloat16(g_45)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_45)) * silu_f32(__bfloat162float(__high2bfloat16(g_45)))));
        __nv_bfloat162 o_67 = __halves2bfloat162(
            __float2bfloat16(__bfloat162float(__low2bfloat16(u_67)) * silu_f32(__bfloat162float(__low2bfloat16(g_67)))),
            __float2bfloat16(__bfloat162float(__high2bfloat16(u_67)) * silu_f32(__bfloat162float(__high2bfloat16(g_67)))));

        // 打包回 float4 并 ST.128
        float4 o4;
        *reinterpret_cast<__nv_bfloat162*>(&o4.x) = o_01;
        *reinterpret_cast<__nv_bfloat162*>(&o4.y) = o_23;
        *reinterpret_cast<__nv_bfloat162*>(&o4.z) = o_45;
        *reinterpret_cast<__nv_bfloat162*>(&o4.w) = o_67;

        *reinterpret_cast<float4*>(out + base) = o4;
    }

    // 尾部处理
    if (tid == 0) {
        size_t tail_start = numel_vec * 8;
        for (size_t i = tail_start; i < numel; ++i) {
            float g = __bfloat162float(to_cuda_bfloat16(gate[i]));
            float u = __bfloat162float(to_cuda_bfloat16(up[i]));
            out[i] = from_cuda_bfloat16(__float2bfloat16(u * silu_f32(g)));
        }
    }
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║                 Kernel 启动函数                               ║
// ╚═══════════════════════════════════════════════════════════════╝
//
// 职责：
//   1. 动态查询 GPU 的 SM 数量（不同型号 GPU 不同）
//   2. 计算最优的 Grid/Block 配置
//   3. 根据数据类型分派到对应的 kernel
//
// Grid 大小策略：
//   BLOCKS = num_sm × 8, THREADS = 256（每 block 8 个 warp）
//   → 每 SM 分到 8 个 block
//   → 但 SM 最多同时驻留 48 warp = 6 个 block（256 线程 = 8 warp）
//     实际 6 × 8 = 48 warp = 满载
//   → 多出的 2 个 block 在就绪队列中排队，当有 block 完成时立即补上
//   → 保证 SM 始终有工作可做，避免"tail effect"（尾部效应）
//
// 尾部效应：最后一批 block 不够填满所有 SM
//   例如数据只需要 40 blocks，36 SM → 第一轮 36 blocks → 第二轮只有 4 blocks
//   → 32 个 SM 空闲！Grid-Stride 模式通过固定 Grid 大小完全避免此问题
void launch_swiglu_kernel(
    std::byte* out,
    const std::byte* gate,
    const std::byte* up,
    llaisysDataType_t dtype,
    size_t numel
) {
    constexpr int THREADS = 256;  // 每 block 256 线程 = 8 warp

    // 运行时查询当前 GPU 的 SM 数量
    // cudaDevAttrMultiProcessorCount = SM（Streaming Multiprocessor）个数
    // RTX 4070 Laptop (AD106) = 36 SM
    // RTX 4090 (AD102) = 128 SM
    // 这使得同一份代码在不同 GPU 上都能自动适配最优 Grid
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);

    // 每 SM 安排 8 个 block
    // 实际同时驻留 6 个（受 48 warp/SM 限制），剩余 2 个排队备用
    int BLOCKS = num_sm * 8;

    // Kernel Launch 是**异步**的（火箭发射）：
    // CPU 把 kernel 放入 GPU 命令队列就立即返回
    // GPU 在自己的时间线上执行 kernel
    // 所以下面的 cudaGetLastError() 只检查"launch 是否成功"（参数合法性）
    // 不检查"kernel 执行是否正确"
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        // F32: float4 向量化, 每线程 16B = LD.128
        swiglu_kernel_f32_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(gate),
            reinterpret_cast<const float*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_F16:
        // F16: float4 向量化, 每线程 16B = LD.128（8 个 half）
        swiglu_kernel_f16_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(gate),
            reinterpret_cast<const llaisys::fp16_t*>(up),
            numel
        );
        break;
    case LLAISYS_DTYPE_BF16:
        // BF16: float4 向量化, 每线程 16B = LD.128（8 个 bf16）
        swiglu_kernel_bf16_vec4<<<BLOCKS, THREADS>>>(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(gate),
            reinterpret_cast<const llaisys::bf16_t*>(up),
            numel
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for CUDA swiglu");
    }

    // 检查 launch 是否成功（参数错误、Grid 超限等会在这里被捕获）
    checkCuda(cudaGetLastError(), "Failed to launch swiglu kernel");
}

} // anonymous namespace

// ╔═══════════════════════════════════════════════════════════════╗
// ║                     对外接口                                   ║
// ╚═══════════════════════════════════════════════════════════════╝
// C++ 层调用入口，由 Python 端通过 ctypes 间接调用
// 参数全部是 raw 指针 + 元数据，不涉及任何框架抽象
namespace llaisys::ops::nvidia {

void swiglu(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* gate_ptr,
    const std::byte* up_ptr,
    size_t numel
) {
    launch_swiglu_kernel(out_ptr, gate_ptr, up_ptr, dtype, numel);
}

} // namespace llaisys::ops::nvidia
