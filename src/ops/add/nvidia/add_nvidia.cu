/**
 * ============================================================================
 * Add 算子 CUDA 实现 - 向量化版本
 * ============================================================================
 * 
 * 算子类型访存密集型 (Memory-Bound)
 * 
 * 计算特点
 *   - 每个元素只做一次加法（1 FLOP）
 *   - 每个元素需要 2 次读 + 1 次写 = 3 次内存访问
 *   - 算术强度 = 1 FLOP / (3 × 4B) ≈ 0.083 FLOP/Byte (F32)
 *   - RTX 4070 Laptop: ~256 GB/s 带宽, ~12 TFLOPS FP32
 *   - 平衡点 = 12T / 256G ≈ 47 FLOP/Byte
 *   - 0.083 << 47 → 典型的 Memory-Bound
 * 
 * 优化策略
 *   1. 向量化访存：float4/half2/bfloat162，减少 memory transaction 数量
 *   2. 合并访存(Coalesced Access)：相邻线程访问相邻内存地址
 *   3. 使用原生向量加法指令：__hadd2 可在单指令内完成 2 个 half 加法
 * 
 * 优化历史
 *   v1: 朴素标量 → v2: float4 向量化 → v3 (当前): Grid-Stride Loop + __restrict__
 *   - Grid-Stride Loop: 固定 grid 大小，线程循环处理数据，解耦并行度与数据量
 *   - __restrict__: 所有指针添加别名限定，允许编译器优化 load/store 重排
 *   - F16/BF16: float4 宽搬运 (LD.128) + __hadd2 硬件向量加法
 * 
 * ============================================================================
 */

#include "add_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../core/context/context.hpp"

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
 * float4 向量化原理
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
 * 设计要点
 *   向量化的真正好处不是"减少指令数"，而是：
 *   1. 提高 sector 利用率（减少浪费的带宽）
 *   2. 更好地利用 L1/L2 cache line（64B/128B）
 *   3. 减少指令发射压力
 * 
 * 线程映射 (Grid-Stride)
 *   固定 grid = SM数×8 个 block，每个 block 256 线程
 *   stride = 总线程数，每个线程通过 i += stride 循环处理多组 float4
 *   例：总线程 69632，数据 100M float4
 *   线程 0: [0], [69632], [139264], ...
 *   线程 1: [1], [69633], [139265], ...
 *   相邻线程在同一轮访问相邻 float4，仍是合并访问
 */
__global__ void add_kernel_f32_vec(
    float * __restrict__ c,
    const float * __restrict__ a,
    const float * __restrict__ b,
    size_t numel
) {
    // Grid-Stride Loop:
    //   tid    = 当前线程的全局唯一 ID
    //   stride = grid 中的总线程数（固定值），每轮循环跳过这么多 vec4 组
    //   这样无论数据有多大，都只用固定数量的 block，线程循环复用
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // --- Grid-Stride 主循环：每轮处理 1 个 float4 (= 4 个 float, 16B) ---
    // (i + 1) * 4 <= numel 保证读取 4 个完整 float 不越界
    for (size_t i = tid; (i + 1) * 4 <= numel; i += stride) {
        size_t base = i * 4;

        // LD.128：一次从 HBM 搬运 16 字节
        // reinterpret_cast 告诉编译器生成宽 load 指令
        // 对齐cudaMalloc 保证至少 256B 对齐，base=0 安全
        float4 a4 = *reinterpret_cast<const float4 *>(a + base);
        float4 b4 = *reinterpret_cast<const float4 *>(b + base);

        // float4 = struct {float x, y, z, w}，逐分量加法
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;

        // ST.128：一次写回 16 字节
        *reinterpret_cast<float4 *>(c + base) = c4;
    }

    // --- 尾部标量处理：numel 不是 4 的倍数时，最多剩 3 个元素 ---
    // 只让 tid==0 处理，避免多线程重复写入
    if (tid == 0) {
        size_t tail_start = (numel / 4) * 4;
        for (size_t i = tail_start; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

// ============================================================================
// F16 向量化 Kernel —— 使用 float4 宽搬运 (LD.128)
// ============================================================================
/**
 * 优化思路：搬运和计算分离
 * 
 * 之前版本：half2 搬运 + half2 计算
 *   → 每线程 LD.32 (4B) = 2 个 half → 指令多、load 窄
 * 
 * 优化版本：float4 搬运 + half2 计算
 *   → 每线程 LD.128 (16B) = 8 个 half → 指令少、load 宽
 *   → 然后把 float4 拆成 4 个 half2 做 __hadd2
 * 
 * 内存布局
 *   float4 (16B) = [half_0 half_1] [half_2 half_3] [half_4 half_5] [half_6 half_7]
 *                   ─── half2_0 ──  ─── half2_1 ──  ─── half2_2 ──  ─── half2_3 ──
 * 
 * 预期效果
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
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride  = blockDim.x * gridDim.x;

    // --- Grid-Stride 主循环：每轮处理 8 个 half (= 1 个 float4 = 16B) ---
    for (size_t i = tid; (i + 1) * 8 <= numel; i += stride) {
        size_t base = i * 8;

        // LD.128：一次从全局内存搬运 16 字节 = 8 个 half
        float4 a_chunk = *reinterpret_cast<const float4 *>(a + base);
        float4 b_chunk = *reinterpret_cast<const float4 *>(b + base);

        // 把 float4 拆解为 4 个 half2 做 __hadd2 硬件向量加法
        __half2 *a_h2 = reinterpret_cast<__half2 *>(&a_chunk);
        __half2 *b_h2 = reinterpret_cast<__half2 *>(&b_chunk);

        __half2 c_h2[4];
        c_h2[0] = __hadd2(a_h2[0], b_h2[0]);  // half[0,1]
        c_h2[1] = __hadd2(a_h2[1], b_h2[1]);  // half[2,3]
        c_h2[2] = __hadd2(a_h2[2], b_h2[2]);  // half[4,5]
        c_h2[3] = __hadd2(a_h2[3], b_h2[3]);  // half[6,7]

        // ST.128：一次写回 16 字节
        *reinterpret_cast<float4 *>(c + base) = *reinterpret_cast<float4 *>(c_h2);
    }

    // --- 尾部标量处理 ---
    if (tid == 0) {
        size_t tail_start = (numel / 8) * 8;
        for (size_t i = tail_start; i < numel; ++i) {
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
 * BFloat16 简介
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
 * 硬件支持
 *   - NVIDIA Ampere (sm_80+) 开始支持原生 BF16 运算
 *   - __hadd2 对 bfloat162 类型同样有效（重载函数）
 * 
 * 与 F16 kernel 的对称性
 *   代码结构完全相同，只是类型从 half → bfloat16
 */
__global__ void add_kernel_bf16_vec(
    llaisys::bf16_t * __restrict__ c,
    const llaisys::bf16_t * __restrict__ a,
    const llaisys::bf16_t * __restrict__ b,
    size_t numel
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // --- Grid-Stride 主循环：每轮处理 8 个 bf16 (= 1 个 float4 = 16B) ---
    for (size_t i = tid; (i + 1) * 8 <= numel; i += stride) {
        size_t base = i * 8;

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

    // --- 尾部标量处理 ---
    if (tid == 0) {
        size_t tail_start = (numel / 8) * 8;
        for (size_t i = tail_start; i < numel; ++i) {
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
 * Grid/Block 配置策略 —— Grid-Stride Loop
 * 
 * threads = 256 是经典选择：
 *   - 256 = 8 个 warp（每 warp 32 线程）
 *   - 大多数 GPU 每 SM 最多 2048 线程，256 线程/block 允许 8 个 block 并发
 *   - 占用率(occupancy)通常在 50%~100% 之间，取决于寄存器/shared memory 使用
 * 
 * blocks = NUM_SM × 8（固定值）：
 *   - 不随数据量变化，线程通过循环复用（Grid-Stride Loop）
 *   - NUM_SM 通过 cudaDeviceGetAttribute 在首次调用时查询并缓存
 *   - RTX 4060 Ti: 34 SM × 8 = 272 blocks，总线程 = 69,632
 *   - A100: 108 SM × 8 = 864 blocks，总线程 = 221,184
 * 
 * 优势（vs 旧版按数据量算 blocks）：
 *   1. 大数据：不会产生 10 万个 block 的调度开销
 *   2. 小数据：grid 本身就足够小，不浪费
 *   3. 永远不会超过 grid 维度限制
 */
/**
 * 获取当前 GPU 的 SM 数量（首次调用时查询并缓存）
 */
int get_num_sm() {
    static int num_sm = 0;
    if (num_sm == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    }
    return num_sm;
}

void launch_add_kernel(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    constexpr int threads = 256;             // 每 block 256 线程 = 8 warps
    const int blocks = get_num_sm() * 8;     // 固定 grid 大小 = SM数 × 8
    
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        add_kernel_f32_vec<<<blocks, threads, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel
        );
        break;
    }
    case LLAISYS_DTYPE_F16: {
        add_kernel_f16_vec<<<blocks, threads, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel
        );
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        add_kernel_bf16_vec<<<blocks, threads, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            
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

    // Kernel 异步执行与错误检查
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
