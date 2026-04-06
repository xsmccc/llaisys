/**
 * @file linear_quantized_nvidia.cu
 * @brief W8A16 / W4A16 量化 Linear 算子的 CUDA 实现 (优化版)
 *
 * ── 量化方案 ──────────────────────────────────────────────────
 *   Per-channel 对称量化 (absmax):
 *     scale[n] = max(|W_fp32[n, :]|) / 127.0
 *     W_int8[n, k] = round(W_fp32[n, k] / scale[n])
 *
 * ── W8A16 流程 ──────────────────────────────────────────────────
 *   1. Activation: FP16 输入 (整个模型 pipeline 运行在 FP16)
 *   2. Weight: INT8 → dequant to FP16 (persistent cache)
 *   3. GEMM: cublasGemmEx FP16×FP16 (FP32 accumulation, Tensor Core)
 *   4. Output: FP16 (或 FP32, 由 output tensor dtype 决定)
 *   兼容 FP32 activation 输入 (内部自动转换为 FP16)
 *
 * ── 优化策略 ──────────────────────────────────────────────────
 *   1. FP16 权重持久缓存: 首次调用时 dequant INT8→FP16 并缓存，后续复用
 *      消除 decode 阶段每次 forward 196 次 dequant 的开销
 *   2. FP16 Tensor Core GEMM: cublasGemmEx (FP16×FP16, FP32 accumulate)
 *      RTX 4060 Ti sm_89: 176.5 TFLOPS FP16 TC vs 22 TFLOPS FP32 = 8x
 *   3. FP16 activation: 跳过 F32→F16 转换，减少 kernel + 带宽开销
 *   4. 消除所有 per-call cudaMalloc/cudaFree
 *
 * ── 性能 (RTX 4060 Ti, DeepSeek-R1-1.5B) ────────────────────
 *   W8A16: 57+ tok/s (1.7x faster than FP32 33.6 tok/s)
 *   FP16 KV Cache 节省 50% 显存
 */

#include "linear_quantized_nvidia.hpp"
#include "../../../utils.hpp"
#include "../../../core/context/context.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <mutex>
#include <unordered_map>

namespace {

// ============================================================
//  全局 cuBLAS Handle
// ============================================================
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("[cuBLAS] Failed to create handle");
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    });
    return handle;
}

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

inline void checkCublas(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(msg) + ": status=" + std::to_string(static_cast<int>(st)));
}

static int get_sm_version() {
    static int sm = -1;
    if (sm >= 0) return sm;
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    sm = prop.major * 10 + prop.minor;
    return sm;
}

// ============================================================
//  FP16 权重持久缓存 (线程安全)
//  key = INT8 权重的 GPU 指针地址 (唯一标识每个权重矩阵)
//  value = dequant 后的 FP16 权重 (生命周期与模型相同)
// ============================================================
static std::unordered_map<uintptr_t, __half*> g_weight_fp16_cache;
static std::mutex g_cache_mutex;  // 保护所有权重缓存的并发访问

// 输入 FP16 转换缓存 (仅 FP32 兼容路径使用)
struct CachedBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    void ensure(size_t needed) {
        if (needed <= size) return;
        if (ptr) cudaFree(ptr);
        checkCuda(cudaMalloc(&ptr, needed), "CachedBuffer::ensure");
        size = needed;
    }
    ~CachedBuffer() { if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; } }
};
static CachedBuffer g_input_fp16_buf;

// ============================================================
//  CUDA Kernels
// ============================================================

// Dequant INT8 → FP16 (向量化 4x)
// 向量化类型转换
__global__ void dequant_int8_to_fp16_vec4(
    __half* __restrict__ out,
    const int8_t* __restrict__ w,
    const float* __restrict__ s,
    size_t N, size_t K // N是行数，K是列数（特征维度）
) {
    const size_t K4 = K / 4;
    const size_t total = N * K4; // 每个线程处理一个 char4 (4 个 INT8 权重)
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引
         i < total; i += gridDim.x * blockDim.x) { // grobal stride loop
        float sc = s[i / K4];
        char4 v = reinterpret_cast<const char4*>(w)[i];
        __half* d = out + i * 4;
        d[0] = __float2half(float(v.x) * sc);
        d[1] = __float2half(float(v.y) * sc);
        d[2] = __float2half(float(v.z) * sc);
        d[3] = __float2half(float(v.w) * sc);
    }
}

// INT8 → FP16 标量版本 (处理 K 不被 4 整除的情况)
__global__ void dequant_int8_to_fp16_scalar(
    __half* __restrict__ out,
    const int8_t* __restrict__ w,
    const float* __restrict__ s,
    size_t N, size_t K
) {
    size_t total = N * K;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
        out[i] = __float2half(float(w[i]) * s[i / K]);
}


// ============================================================
//  INT4 Group Dequant: unpack uint8 → 2×int4 → FP16
//  Pack format: byte = (high_nibble << 4) | (low_nibble & 0xF)
//  Scales: FP16 [N, num_groups], group_size typically 128
// 每个线程处理一个 uint8_t，解出两个 int4 权重并乘以对应的 scale 转 FP16
// ============================================================
__global__ void dequant_int4_to_fp16_group(
    __half* __restrict__ out,
    const uint8_t* __restrict__ w_packed,
    const __half* __restrict__ scales,
    size_t N, size_t K_packed,
    size_t num_groups, size_t group_size
) {
    const size_t total = N * K_packed;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
    {
        const size_t row = i / K_packed;
        const size_t col_packed = i % K_packed;
        const size_t col_lo = col_packed * 2;
        const size_t col_hi = col_lo + 1;

        const size_t grp_lo = col_lo / group_size;
        const size_t grp_hi = col_hi / group_size;

        float s_lo = __half2float(scales[row * num_groups + grp_lo]);
        float s_hi = __half2float(scales[row * num_groups + grp_hi]);

        uint8_t byte_val = w_packed[i];
        int lo = (int)(byte_val & 0xF);
        if (lo >= 8) lo -= 16;
        int hi = (int)(byte_val >> 4);
        if (hi >= 8) hi -= 16;

        const size_t K_orig = K_packed * 2;
        out[row * K_orig + col_lo] = __float2half((float)lo * s_lo);
        out[row * K_orig + col_hi] = __float2half((float)hi * s_hi);
    }
}

// FP32 → FP16 输入转换 (兼容 FP32 输入路径)
__global__ void f32_to_fp16(
    __half* __restrict__ out,
    const float* __restrict__ in,
    size_t total
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
        out[i] = __float2half(in[i]);
}

// Bias 加法 (FP32)
__global__ void add_bias_f32(
    float* __restrict__ out,
    const float* __restrict__ bias,
    size_t rows, size_t N
) {
    size_t total = rows * N;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
        out[i] += bias[i % N];
}

// Bias 加法 (FP16 output, FP16 bias)
__global__ void add_bias_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ bias,
    size_t rows, size_t N
) {
    size_t total = rows * N;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
        out[i] = __hadd(out[i], bias[i % N]);
}

// Bias 加法 (FP16 output, FP32 bias — 自动转换)
__global__ void add_bias_fp16_from_f32(
    __half* __restrict__ out,
    const float* __restrict__ bias,
    size_t rows, size_t N
) {
    size_t total = rows * N;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += gridDim.x * blockDim.x)
        out[i] = __hadd(out[i], __float2half(bias[i % N]));
}

// ============================================================
//  Custom INT8 GEMV Kernel (M=1 Decode)
//  直接从 INT8 权重读取, 寄存器内 on-the-fly dequant
//  减少 2x HBM 带宽 (1 byte vs 2 bytes per weight)
//
//  Grid: (ceil(N / WARPS_PER_BLOCK), 1)
//  Block: (WARPS_PER_BLOCK * 32, 1)  [256 threads]
//  Shared mem: K * sizeof(__half) bytes for input vector
//
//  每个 warp 计算一个输出: out[n] = sum_k(W_int8[n,k] * scale[n] * x[k])
// ============================================================
static constexpr int GEMV_WARPS_PER_BLOCK = 8;  // 每个block的warp数量
static constexpr int GEMV_BLOCK_DIM = GEMV_WARPS_PER_BLOCK * 32;  // 256线程per block

__global__ void int8_gemv_kernel(
    void* __restrict__ out,
    const __half* __restrict__ x,       // [K] 输入
    const int8_t* __restrict__ W,       // [N, K] row-major 权重
    const float* __restrict__ scales,   // [N] per-channel
    int N, int K,
    bool out_fp16                       // true: __half output, false: float output
) {
    extern __shared__ __half x_smem[]; //动态shared memory存储输入向量x的FP16版本

    // 所有线程协作加载输入向量到 shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        x_smem[i] = x[i];
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp_id; // warp负责第n行

    if (n >= N) return;

    const int8_t* w_row = W + (size_t)n * K;
    const float scale = scales[n];

    float sum = 0.0f;

    // char4 向量化读取: 一次 4 个 INT8 权重 + half2 读输入
    const int K4 = K / 4;
    for (int i = lane_id; i < K4; i += 32) {
        char4 w4 = reinterpret_cast<const char4*>(w_row)[i];
        const int base = i * 4;
        __half2 x01 = *reinterpret_cast<const __half2*>(&x_smem[base]);
        __half2 x23 = *reinterpret_cast<const __half2*>(&x_smem[base + 2]);
        float2 fx01 = __half22float2(x01);
        float2 fx23 = __half22float2(x23);
        sum += float(w4.x) * fx01.x + float(w4.y) * fx01.y
             + float(w4.z) * fx23.x + float(w4.w) * fx23.y;
    }
    // 处理 K 不被 4 整除的尾部
    for (int k = K4 * 4 + lane_id; k < K; k += 32)
        sum += float(w_row[k]) * __half2float(x_smem[k]);

    sum *= scale;  // per-channel dequant 计算之后再乘scale

    // Warp shuffle reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane_id == 0) {
        if (out_fp16)
            reinterpret_cast<__half*>(out)[n] = __float2half(sum);
        else
            reinterpret_cast<float*>(out)[n] = sum;
    }
}

// ============================================================
//  Custom INT4 GEMV Kernel (M=1 Decode)
//  直接从 packed INT4 权重读取, 寄存器内 unpack + on-the-fly dequant
//  减少 4x HBM 带宽 (0.5 byte vs 2 bytes per weight)
//
//  权重格式: uint8_t [N, K/2], 每 byte 存 2 个 signed int4
//    lo = byte & 0xF (signed: if >= 8, -= 16)
//    hi = byte >> 4  (signed: if >= 8, -= 16)
//  Scale: __half [N, num_groups], per-group dequant (group_size=128)
// ============================================================
__global__ void int4_gemv_kernel(
    void* __restrict__ out,
    const __half* __restrict__ x,          // [K_orig]
    const uint8_t* __restrict__ W_packed,  // [N, K_orig/2]
    const __half* __restrict__ scales,     // [N, num_groups]
    int N, int K_orig,
    int num_groups, int group_size,
    bool out_fp16
) {
    extern __shared__ __half x_smem[];

    for (int i = threadIdx.x; i < K_orig; i += blockDim.x)
        x_smem[i] = x[i];
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS_PER_BLOCK + warp_id;

    if (n >= N) return;

    const int K_packed = K_orig / 2;
    const uint8_t* w_row = W_packed + (size_t)n * K_packed;
    const __half* s_row = scales + (size_t)n * num_groups;

    float sum = 0.0f;

    // uint32_t 向量化: 一次读 4 bytes = 8 个 INT4 值
    const int K4 = K_packed / 4;
    for (int i = lane_id; i < K4; i += 32) {
        uint32_t packed4 = reinterpret_cast<const uint32_t*>(w_row)[i];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
            int lo = (int)(byte_val & 0xF);
            if (lo >= 8) lo -= 16;
            int hi = (int)(byte_val >> 4);
            if (hi >= 8) hi -= 16;

            const int col_lo = (i * 4 + j) * 2;
            const int col_hi = col_lo + 1;

            float s_lo = __half2float(s_row[col_lo / group_size]);
            float s_hi = __half2float(s_row[col_hi / group_size]);

            __half2 x2 = *reinterpret_cast<const __half2*>(&x_smem[col_lo]);
            float2 fx2 = __half22float2(x2);

            sum += (float)lo * s_lo * fx2.x;
            sum += (float)hi * s_hi * fx2.y;
        }
    }
    // 处理尾部
    for (int i = K4 * 4 + lane_id; i < K_packed; i += 32) {
        uint8_t byte_val = w_row[i];
        int lo = (int)(byte_val & 0xF);
        if (lo >= 8) lo -= 16;
        int hi = (int)(byte_val >> 4);
        if (hi >= 8) hi -= 16;
        const int col_lo = i * 2;
        const int col_hi = col_lo + 1;
        float s_lo = __half2float(s_row[col_lo / group_size]);
        float s_hi = __half2float(s_row[col_hi / group_size]);
        sum += (float)lo * s_lo * __half2float(x_smem[col_lo]);
        sum += (float)hi * s_hi * __half2float(x_smem[col_hi]);
    }

    // Warp shuffle reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane_id == 0) {
        if (out_fp16)
            reinterpret_cast<__half*>(out)[n] = __float2half(sum);
        else
            reinterpret_cast<float*>(out)[n] = sum;
    }
}

// ============================================================
//  权重缓存 (首次 dequant + 持久缓存)
// ============================================================
__half* get_or_create_fp16_weight(
    const int8_t* weight_int8,
    const float* scales,
    size_t N, size_t K
) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    uintptr_t key = reinterpret_cast<uintptr_t>(weight_int8);
    auto it = g_weight_fp16_cache.find(key);
    if (it != g_weight_fp16_cache.end())
        return it->second;

    __half* w_fp16 = nullptr;
    checkCuda(cudaMalloc(&w_fp16, N * K * sizeof(__half)), "alloc FP16 weight");

    constexpr int T = 256;
    if (K % 4 == 0) {
        size_t vecs = N * (K / 4);
        dequant_int8_to_fp16_vec4<<<(int)((vecs + T - 1) / T), T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            w_fp16, weight_int8, scales, N, K);
    } else {
        size_t tot = N * K;
        dequant_int8_to_fp16_scalar<<<(int)((tot + T - 1) / T), T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            w_fp16, weight_int8, scales, N, K);
    }
    checkCuda(cudaGetLastError(), "dequant kernel");
    checkCuda(cudaDeviceSynchronize(), "dequant sync");

    g_weight_fp16_cache[key] = w_fp16;
    return w_fp16;
}


// ============================================================
//  INT4 权重缓存 (首次 dequant + 持久缓存)
//  与 INT8 缓存共用 g_cache_mutex
// ============================================================
static std::unordered_map<uintptr_t, __half*> g_weight_int4_fp16_cache;

__half* get_or_create_fp16_weight_int4(
    const uint8_t* weight_packed,
    const __half* scales,
    size_t N, size_t K_packed,
    size_t num_groups, size_t group_size
) {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    uintptr_t key = reinterpret_cast<uintptr_t>(weight_packed);
    auto it = g_weight_int4_fp16_cache.find(key);
    if (it != g_weight_int4_fp16_cache.end())
        return it->second;

    size_t K_orig = K_packed * 2;
    __half* w_fp16 = nullptr;
    checkCuda(cudaMalloc(&w_fp16, N * K_orig * sizeof(__half)), "alloc FP16 weight (INT4)");

    constexpr int T = 256;
    size_t total = N * K_packed;
    dequant_int4_to_fp16_group<<<(int)((total + T - 1) / T), T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
        w_fp16, weight_packed, scales, N, K_packed, num_groups, group_size);
    checkCuda(cudaGetLastError(), "dequant_int4 kernel");
    checkCuda(cudaDeviceSynchronize(), "dequant_int4 sync");

    g_weight_int4_fp16_cache[key] = w_fp16;
    return w_fp16;
}

// ============================================================
//  统一入口 — W8A16 (支持 FP16/FP32 activation)
// ============================================================
void linear_quantized_impl(
    std::byte* out,
    const std::byte* in,
    const int8_t* weight_int8,
    const float* scales,
    const std::byte* bias,
    size_t K, size_t N, size_t M,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
) {
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, (cudaStream_t)llaisys::core::context().runtime().stream());

    if (get_sm_version() >= 70) {
        // 获取 FP16 输入指针
        const __half* in_fp16 = nullptr;
        if (in_dtype == LLAISYS_DTYPE_F16) {
            in_fp16 = reinterpret_cast<const __half*>(in);
        } else {
            size_t in_count = M * K;
            g_input_fp16_buf.ensure(in_count * sizeof(__half));
            __half* buf = static_cast<__half*>(g_input_fp16_buf.ptr);
            constexpr int T = 256;
            f32_to_fp16<<<(int)((in_count + T - 1) / T), T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                buf, reinterpret_cast<const float*>(in), in_count);
            checkCuda(cudaGetLastError(), "f32->fp16");
            in_fp16 = buf;
        }

        if (M == 1) {
            // ── Custom GEMV: 直接读 INT8 权重, on-the-fly dequant (2x 带宽节省) ──
            int grid = ((int)N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK;
            size_t smem = K * sizeof(__half);
            bool out_is_fp16 = (out_dtype == LLAISYS_DTYPE_F16);
            int8_gemv_kernel<<<grid, GEMV_BLOCK_DIM, smem, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                out, in_fp16, weight_int8, scales, (int)N, (int)K, out_is_fp16);
            checkCuda(cudaGetLastError(), "int8_gemv_kernel");
        } else {
            // ── FP16 Tensor Core 路径 (Volta+, M>1 prefill) ──
            __half* w_fp16 = get_or_create_fp16_weight(weight_int8, scales, N, K);
            cudaDataType_t c_type = (out_dtype == LLAISYS_DTYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
            const float alpha = 1.0f, beta = 0.0f;
            checkCublas(
                cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)N, (int)M, (int)K,
                    &alpha,
                    w_fp16, CUDA_R_16F, (int)K,
                    in_fp16, CUDA_R_16F, (int)K,
                    &beta,
                    out, c_type, (int)N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx FP16 TC");
        }
    } else {
        throw std::runtime_error(
            "linear_quantized requires sm >= 70 for FP16 Tensor Core. "
            "Current device has sm " + std::to_string(get_sm_version()));
    }

    // Bias 加法 — 根据 output/bias dtype 选择 kernel
    if (bias) {
        constexpr int T = 256;
        size_t total = M * N;
        int blocks = (int)((total + T - 1) / T);
        if (out_dtype == LLAISYS_DTYPE_F16) {
            if (bias_dtype == LLAISYS_DTYPE_F16) {
                add_bias_fp16<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                    reinterpret_cast<__half*>(out),
                    reinterpret_cast<const __half*>(bias), M, N);
            } else {
                add_bias_fp16_from_f32<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                    reinterpret_cast<__half*>(out),
                    reinterpret_cast<const float*>(bias), M, N);
            }
        } else {
            add_bias_f32<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                reinterpret_cast<float*>(out),
                reinterpret_cast<const float*>(bias), M, N);
        }
        checkCuda(cudaGetLastError(), "bias add");
    }
}


// ============================================================
//  INT4 统一入口 — W4A16 (支持 FP16/FP32 activation)
// ============================================================
void linear_quantized_int4_impl(
    std::byte* out,
    const std::byte* in,
    const uint8_t* weight_packed,
    const __half* scales,
    const std::byte* bias,
    size_t K_orig, size_t N, size_t M,
    size_t num_groups, size_t group_size,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
) {
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, (cudaStream_t)llaisys::core::context().runtime().stream());
    size_t K_packed = K_orig / 2;

    // 获取 FP16 输入指针
    const __half* in_fp16 = nullptr;
    if (in_dtype == LLAISYS_DTYPE_F16) {
        in_fp16 = reinterpret_cast<const __half*>(in);
    } else {
        size_t in_count = M * K_orig;
        g_input_fp16_buf.ensure(in_count * sizeof(__half));
        __half* buf = static_cast<__half*>(g_input_fp16_buf.ptr);
        constexpr int T = 256;
        f32_to_fp16<<<(int)((in_count + T - 1) / T), T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            buf, reinterpret_cast<const float*>(in), in_count);
        checkCuda(cudaGetLastError(), "f32->fp16 (INT4)");
        in_fp16 = buf;
    }

    if (M == 1) {
        // ── Custom GEMV: 直接读 INT4 packed 权重, on-the-fly dequant (4x 带宽节省) ──
        int grid = ((int)N + GEMV_WARPS_PER_BLOCK - 1) / GEMV_WARPS_PER_BLOCK;
        size_t smem = K_orig * sizeof(__half);
        bool out_is_fp16 = (out_dtype == LLAISYS_DTYPE_F16);
        int4_gemv_kernel<<<grid, GEMV_BLOCK_DIM, smem, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
            out, in_fp16, weight_packed, scales,
            (int)N, (int)K_orig, (int)num_groups, (int)group_size, out_is_fp16);
        checkCuda(cudaGetLastError(), "int4_gemv_kernel");
    } else {
        // ── FP16 Tensor Core 路径 (M>1 prefill) ──
        __half* w_fp16 = get_or_create_fp16_weight_int4(
            weight_packed, scales, N, K_packed, num_groups, group_size);
        cudaDataType_t c_type = (out_dtype == LLAISYS_DTYPE_F16) ? CUDA_R_16F : CUDA_R_32F;
        const float alpha = 1.0f, beta = 0.0f;
        checkCublas(
            cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)N, (int)M, (int)K_orig,
                &alpha,
                w_fp16, CUDA_R_16F, (int)K_orig,
                in_fp16, CUDA_R_16F, (int)K_orig,
                &beta,
                out, c_type, (int)N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "cublasGemmEx FP16 TC (INT4)");
    }

    if (bias) {
        constexpr int T = 256;
        size_t total = M * N;
        int blocks = (int)((total + T - 1) / T);
        if (out_dtype == LLAISYS_DTYPE_F16) {
            if (bias_dtype == LLAISYS_DTYPE_F16) {
                add_bias_fp16<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                    reinterpret_cast<__half*>(out),
                    reinterpret_cast<const __half*>(bias), M, N);
            } else {
                add_bias_fp16_from_f32<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                    reinterpret_cast<__half*>(out),
                    reinterpret_cast<const float*>(bias), M, N);
            }
        } else {
            add_bias_f32<<<blocks, T, 0, (cudaStream_t)llaisys::core::context().runtime().stream()>>>(
                reinterpret_cast<float*>(out),
                reinterpret_cast<const float*>(bias), M, N);
        }
        checkCuda(cudaGetLastError(), "bias add (INT4)");
    }
}

} // anonymous namespace

// ============================================================
//  权重缓存清理 (释放所有 GPU 显存)
// ============================================================
namespace llaisys::ops::nvidia {
 
void cleanup_quantized_weight_cache() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for (auto& [key, ptr] : g_weight_fp16_cache) {
        if (ptr) cudaFree(ptr);
    }
    g_weight_fp16_cache.clear();
    for (auto& [key, ptr] : g_weight_int4_fp16_cache) {
        if (ptr) cudaFree(ptr);
    }
    g_weight_int4_fp16_cache.clear();
}

} // namespace llaisys::ops::nvidia

// ============================================================
//  对外接口
// ============================================================
namespace llaisys::ops::nvidia {

void linear_quantized(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_int8,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
) {
    linear_quantized_impl(
        out, in,
        reinterpret_cast<const int8_t*>(weight_int8),
        reinterpret_cast<const float*>(scales),
        bias,
        in_features, out_features, rows,
        in_dtype, out_dtype, bias_dtype);
}


void linear_quantized_int4(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight_packed,
    const std::byte* scales,
    const std::byte* bias,
    size_t in_features,
    size_t out_features,
    size_t rows,
    size_t num_groups,
    size_t group_size,
    llaisysDataType_t in_dtype,
    llaisysDataType_t out_dtype,
    llaisysDataType_t bias_dtype
) {
    linear_quantized_int4_impl(
        out, in,
        reinterpret_cast<const uint8_t*>(weight_packed),
        reinterpret_cast<const __half*>(scales),
        bias,
        in_features, out_features, rows,
        num_groups, group_size,
        in_dtype, out_dtype, bias_dtype);
}

} // namespace llaisys::ops::nvidia
