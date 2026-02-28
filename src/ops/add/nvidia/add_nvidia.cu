/**
 * ============================================================================
 * Add 算子 CUDA 实现 - 向量化版本
 * ============================================================================
 * 
 * 【算子类型】访存密集型 (Memory-Bound)
 * 
 * 【计算特点】
 *   - 每个元素只做一次加法（1 FLOP）
 *   - 每个元素需要 2 次读 + 1 次写 = 3 次内存访问
 *   - 算术强度 = 1 FLOP / (3 × 4B) ≈ 0.083 FLOP/Byte (F32)
 *   - RTX 4070 Laptop: ~256 GB/s 带宽, ~12 TFLOPS FP32
 *   - 平衡点 = 12T / 256G ≈ 47 FLOP/Byte
 *   - 0.083 << 47 → 典型的 Memory-Bound
 * 
 * 【优化策略】
 *   1. 向量化访存：float4/half2/bfloat162，减少 memory transaction 数量
 *   2. 合并访存(Coalesced Access)：相邻线程访问相邻内存地址
 *   3. 使用原生向量加法指令：__hadd2 可在单指令内完成 2 个 half 加法
 * 
 * 【当前版本的不足】
 *   - 缺少 Grid-Stride Loop（大数据时 grid 过大、小数据时 wave 不足）
 *   - 缺少 __restrict__ 指针标注（编译器可能无法优化别名分析）
 *   - F16/BF16 只用 half2 (4B)，可改用 float4 搬运 8 个 half (16B)
 * 
 * ============================================================================
 */

#include "add_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_runtime.h>   // CUDA 运行时 API (cudaGetLastError, cudaGetErrorString 等)
#include <cuda_fp16.h>      // FP16/half 类型支持 (__half, __half2, __hadd, __hadd2)
#include <cuda_bf16.h>      // BFloat16 类型支持 (__nv_bfloat16, __nv_bfloat162)

#include <stdexcept>
#include <iostream>

namespace {

/**
 * CUDA 错误检查宏
 * 每次 kernel launch 后都应检查错误（kernel 是异步的，错误会延迟报告）
 */
inline void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

// ============================================================================
// 类型转换辅助函数
// ============================================================================
// 
// 
//   __forceinline__ 的作用
//   强制编译器内联此函数，避免函数调用开销
//   对于这种只有一行的类型转换函数，内联是必须的

__device__ __forceinline__ __half to_cuda_half(llaisys::fp16_t v) {
    // reinterpret_cast: 不改变内存内容，只改变类型解释
    // &v._v: 取底层 uint16_t 的地址
    return *reinterpret_cast<const __half *>(&v._v);
}

__device__ __forceinline__ llaisys::fp16_t from_cuda_half(__half h) {
    return *reinterpret_cast<const llaisys::fp16_t *>(&h);
}

__device__ __forceinline__ __nv_bfloat16 to_cuda_bfloat16(llaisys::bf16_t v) {
    return *reinterpret_cast<const __nv_bfloat16 *>(&v._v);
}

__device__ __forceinline__ llaisys::bf16_t from_cuda_bfloat16(__nv_bfloat16 b) {
    return *reinterpret_cast<const llaisys::bf16_t *>(&b);
}

// ============================================================================
// F32 向量化 Kernel
// ============================================================================
/**
 * 【float4 向量化原理】
 * 
 * GPU 内存系统的最小传输单位是 32 字节（一个 "sector"）。
 * 
 * 不使用向量化时：
 *   - 每线程读 1 个 float (4B)
 *   - 32 个线程组成一个 warp，一次 warp 访问 32×4B = 128B
 *   - 需要 4 个 memory transaction（每个 32B）
 * 
 * 使用 float4 向量化时：
 *   - 每线程读 1 个 float4 (16B)
 *   - 编译器会生成 LD.128 指令（一次 load 128 bits = 16 bytes）
 *   - Warp 32 线程 × 16B = 512B，但编译器会合并为更少的 transaction
 * 
 * 【关键洞察】
 *   向量化的真正好处不是"减少指令数"，而是：
 *   1. 提高 sector 利用率（减少浪费的带宽）
 *   2. 更好地利用 L1/L2 cache line（64B/128B）
 *   3. 减少指令发射压力
 * 
 * 【线程映射】
 *   假设 numel = 1024，threads = 256，则 blocks = 1
 *   线程 0 处理 [0,1,2,3]，线程 1 处理 [4,5,6,7]，...
 *   相邻线程访问不相邻的 float4，但整体仍是合并访问
 */
__global__ void add_kernel_f32_vec(float *c, const float *a, const float *b, size_t numel) {
    // 计算当前线程的全局索引
    // blockIdx.x: 当前 block 在 grid 中的索引 (0 ~ gridDim.x-1)
    // blockDim.x: 每个 block 的线程数 (这里是 256)
    // threadIdx.x: 当前线程在 block 内的索引 (0 ~ 255)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理 4 个连续的 float，计算起始位置
    size_t base = idx * 4;

    // --- 向量化路径：一次处理 4 个 float ---
    // 条件：确保 [base, base+3] 都在有效范围内
    if (base + 3 < numel) {
        // reinterpret_cast 将 float* 解释为 float4*
        // 这告诉编译器生成 LD.128 / ST.128 指令
        // 
        // 【内存对齐要求】
        //   float4 需要 16 字节对齐。如果输入数组是动态分配的（cudaMalloc），
        //   CUDA 保证至少 256 字节对齐，所以 base=0 是安全的。
        //   但 base=1 时地址不对齐，会导致未定义行为或性能下降。
        //   当前代码假设输入已对齐（实际 tensor 从偏移 0 开始）
        float4 a4 = *reinterpret_cast<const float4 *>(a + base);
        float4 b4 = *reinterpret_cast<const float4 *>(b + base);
        
        // float4 是结构体 {float x, y, z, w}
        // 逐分量相加（编译器可能向量化为 FADD.F32x4，取决于体系结构）
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        
        // 将结果写回全局内存（ST.128 指令）
        *reinterpret_cast<float4 *>(c + base) = c4;
    } 
    // --- 标量路径：处理尾部 0~3 个剩余元素 ---
    // 当 numel 不是 4 的倍数时，最后几个元素只能标量处理
    else {
        for (size_t i = base; i < numel && i < base + 4; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

// ============================================================================
// F16 向量化 Kernel —— 使用 float4 宽搬运 (LD.128)
// ============================================================================
/**
 * 【优化思路：搬运和计算分离】
 * 
 * 之前版本：half2 搬运 + half2 计算
 *   → 每线程 LD.32 (4B) = 2 个 half → 指令多、load 窄
 * 
 * 优化版本：float4 搬运 + half2 计算
 *   → 每线程 LD.128 (16B) = 8 个 half → 指令少、load 宽
 *   → 然后把 float4 拆成 4 个 half2 做 __hadd2
 * 
 * 【内存布局】
 *   float4 (16B) = [half_0 half_1] [half_2 half_3] [half_4 half_5] [half_6 half_7]
 *                   ─── half2_0 ──  ─── half2_1 ──  ─── half2_2 ──  ─── half2_3 ──
 * 
 * 【预期效果】
 *   - load/store 指令数减少 4 倍
 *   - DRAM Throughput 可能从 92% → 接近 95%
 *   - Grid Size 减小 4 倍（从 4096 → 1024）
 */
__global__ void add_kernel_f16_vec(
    llaisys::fp16_t * __restrict__ c,
    const llaisys::fp16_t * __restrict__ a,
    const llaisys::fp16_t * __restrict__ b,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 8;  // 每线程处理 8 个 half (= 16B = 1 个 float4)

    // --- float4 宽搬运路径：一次处理 8 个 half ---
    if (base + 7 < numel) {
        // LD.128：一次从全局内存搬运 16 字节
        // 把 fp16_t* 地址解释为 float4*，读入 4 个 float（= 8 个 half）
        float4 a_chunk = *reinterpret_cast<const float4 *>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4 *>(b + base);

        // 把 float4 拆解为 4 个 half2 做向量加法
        // reinterpret_cast<__half2*> 不改变内存内容，只改变类型解释
        __half2 *a_h2 = reinterpret_cast<__half2 *>(&a_chunk);
        __half2 *b_h2 = reinterpret_cast<__half2 *>(&b_chunk);

        // 4 次 __hadd2 = 8 次 half 加法
        __half2 c_h2[4];
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]);  // half[0,1]
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]);  // half[2,3]
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]);  // half[4,5]
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]);  // half[6,7]

        // ST.128：一次写回 16 字节
        *reinterpret_cast<float4 *>(c + base) = *reinterpret_cast<float4 *>(c_h2);
    }
    // --- 标量尾部处理：剩余 0~7 个 half ---
    else {
        for (size_t i = base; i < numel && i < base + 8; ++i) {
            __half ha = to_cuda_half(a[i]);
            __half hb = to_cuda_half(b[i]);
            c[i] = from_cuda_half(__hadd(ha, hb));
        }
    }
}

// ============================================================================
// BF16 向量化 Kernel
// ============================================================================
/**
 * 【BFloat16 简介】
 * 
 * BF16 是 Google Brain 提出的 16 位浮点格式：
 *   - 1 位符号 + 8 位指数 + 7 位尾数
 *   - 对比 FP16: 1 + 5 + 10
 *   - 对比 FP32: 1 + 8 + 23
 * 
 * BF16 的特点：
 *   - 指数范围与 FP32 相同（±3.4e38），不易溢出
 *   - 精度比 FP16 低（7 vs 10 位尾数），但对深度学习足够
 *   - FP32 ↔ BF16 转换只需截断/补零尾数，无需重新归一化
 * 
 * 【硬件支持】
 *   - NVIDIA Ampere (sm_80+) 开始支持原生 BF16 运算
 *   - __hadd2 对 bfloat162 类型同样有效（重载函数）
 * 
 * 【与 F16 kernel 的对称性】
 *   代码结构完全相同，只是类型从 half → bfloat16
 */
__global__ void add_kernel_bf16_vec(
    llaisys::bf16_t * __restrict__ c,
    const llaisys::bf16_t * __restrict__ a,
    const llaisys::bf16_t * __restrict__ b,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = idx * 8;  // 每线程处理 8 个 bf16 (= 16B = 1 个 float4)

    // --- float4 宽搬运路径 ---
    if (base + 7 < numel) {
        float4 a_chunk = *reinterpret_cast<const float4 *>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4 *>(b + base);

        __nv_bfloat162 *a_h2 = reinterpret_cast<__nv_bfloat162 *>(&a_chunk);
        __nv_bfloat162 *b_h2 = reinterpret_cast<__nv_bfloat162 *>(&b_chunk);

        __nv_bfloat162 c_h2[4];
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]);
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]);
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]);
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]);

        *reinterpret_cast<float4 *>(c + base) = *reinterpret_cast<float4 *>(c_h2);
    }
    // --- 标量尾部处理 ---
    else {
        for (size_t i = base; i < numel && i < base + 8; ++i) {
            __nv_bfloat16 ba = to_cuda_bfloat16(a[i]);
            __nv_bfloat16 bb = to_cuda_bfloat16(b[i]);
            c[i] = from_cuda_bfloat16(__hadd(ba, bb));
        }
    }
}

// ============================================================================
// Kernel 启动器
// ============================================================================
/**
 * 【Grid/Block 配置策略】
 * 
 * threads = 256 是经典选择：
 *   - 256 = 8 个 warp（每 warp 32 线程）
 *   - 大多数 GPU 每 SM 最多 2048 线程，256 线程/block 允许 8 个 block 并发
 *   - 占用率(occupancy)通常在 50%~100% 之间，取决于寄存器/shared memory 使用
 * 
 * blocks 计算：向上取整确保覆盖所有元素
 *   blocks = ceil(numel_vec / threads)
 * 
 * 【当前实现的问题】
 *   当 numel 很大（如 100M 元素）时：
 *     F32: numel_vec = 25M, blocks = 97657
 *   这会导致：
 *   1. kernel launch 开销增加
 *   2. 可能超过 1D grid 的最大 block 数限制（65535 on some configs）
 * 
 * 【更好的做法：Grid-Stride Loop】
 *   固定 grid 大小（如 SM数 × 8），让每个线程用循环处理多个元素。
 *   参见 SwiGLU 的实现。
 */
void launch_add_kernel(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    constexpr int threads = 256;  // 每 block 256 线程 = 8 warps
    
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        // float4 向量化：每线程处理 4 个元素
        // numel_vec = 需要的"向量化单元"数量
        size_t numel_vec = (numel + 3) / 4;  // 向上取整
        int blocks = static_cast<int>((numel_vec + threads - 1) / threads);
        
        // <<<blocks, threads>>> 是 CUDA kernel launch 语法
        // blocks: grid 中的 block 数量（1D）
        // threads: 每个 block 的线程数（1D）
        add_kernel_f32_vec<<<blocks, threads>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel
        );
        break;
    }
    case LLAISYS_DTYPE_F16: {
        // 【优化后】float4 宽搬运：每线程处理 8 个 half (16B)
        size_t numel_vec = (numel + 7) / 8;  // 向上取整到 8 的倍数
        int blocks = static_cast<int>((numel_vec + threads - 1) / threads);
        add_kernel_f16_vec<<<blocks, threads>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel
        );
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        // 【优化后】float4 宽搬运：每线程处理 8 个 bf16 (16B)
        size_t numel_vec = (numel + 7) / 8;
        int blocks = static_cast<int>((numel_vec + threads - 1) / threads);
        add_kernel_bf16_vec<<<blocks, threads>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b),
            numel
        );
        break;
    }
    default:
        throw std::invalid_argument("Unsupported dtype for CUDA add");
    }

    // 【Kernel 异步执行与错误检查】
    // CUDA kernel 是异步的：<<<>>> 立即返回，kernel 在 GPU 上排队执行
    // cudaGetLastError() 检查最近一次 kernel launch 是否有配置错误
    // 注意：这不会等待 kernel 完成，运行时错误需要 cudaDeviceSynchronize() 后才能捕获
    checkCuda(cudaGetLastError(), "Failed to launch add kernel");
}
} // namespace

// ============================================================================
// 对外接口
// ============================================================================
namespace llaisys::ops::nvidia {

/**
 * Add 算子入口点
 * 
 * 被 src/ops/add/op.cpp 中的 dispatch 逻辑调用
 * 当 device == NVIDIA 时路由到此函数
 */
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    launch_add_kernel(c, a, b, type, numel);
}

} // namespace llaisys::ops::nvidia
