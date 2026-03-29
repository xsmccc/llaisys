# LLAISYS 技术报告

> NVIDIA RTX 4060 Ti 8GB (sm_89, Ada Lovelace) | CUDA 12.6 | xmake 构建

---

## 目录

1. [系统架构](#1-系统架构)
2. [FlashAttention v2 实现](#2-flashattention-v2-实现)
3. [INT8/INT4 量化优化](#3-int8int4-量化优化)
4. [KV Cache INT8 量化](#4-kv-cache-int8-量化)
5. [Prefill/Decode 分离](#5-prefill-decode-分离)
6. [算子融合与内存管理](#6-算子融合与内存管理)
7. [模型适配](#7-模型适配)
8. [性能汇总](#8-性能汇总)
9. [Roofline 分析与性能剖析](#9-roofline-分析与性能剖析)
10. [API 参考](#10-api-参考)
11. [复现指南](#11-复现指南)

---

## 1. 系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────┐
│              Python Frontend (ctypes FFI)         │
│  ├─ models/ (Qwen2, Llama3)                      │
│  ├─ server/ (FastAPI + SSE + Web UI)             │
│  └─ ops.py / tensor.py                           │
├──────────────────────────────────────────────────┤
│              C API Layer (include/llaisys/)       │
├──────────────────────────────────────────────────┤
│  C++ Core                                         │
│  ├─ Context  (thread_local 设备上下文管理)        │
│  ├─ Runtime  (流管理, 设备抽象, 惰性初始化)       │
│  └─ CachingAllocator (best-fit 显存池)           │
├──────────────────────────────────────────────────┤
│  CUDA Operators (4,597 行)                        │
│  ├─ self_attention   (815L)  FlashAttention v2    │
│  ├─ linear_quantized (714L)  INT8/INT4 持久缓存   │
│  ├─ kv_cache_quant          KV Cache INT8         │
│  ├─ fused_add_rmsnorm       残差+归一化融合        │
│  ├─ rms_norm / argmax       warp reduce           │
│  └─ rope / swiglu / embedding / add               │
├──────────────────────────────────────────────────┤
│  Device Backend: NVIDIA CUDA 12 / cuBLAS TC      │
└──────────────────────────────────────────────────┘
```

### 1.2 多设备抽象 — 函数指针策略模式

采用 C 函数指针表替代 C++ 虚函数，保证 C ABI 兼容性：

```cpp
struct LlaisysRuntimeAPI {
    void *(*malloc_device)(size_t);
    void  (*free_device)(void*);
    void  (*memcpy_sync)(void*, const void*, size_t, llaisysMemcpyKind_t);
    void  (*memcpy_async)(void*, const void*, size_t, llaisysMemcpyKind_t, llaisysStream_t);
    llaisysStream_t (*create_stream)();
    void (*sync_stream)(llaisysStream_t);
    // ...
};
```

`Context` 通过 `thread_local` 管理当前设备上下文，`Runtime` 按设备类型惰性初始化。新增硬件只需实现 RuntimeAPI + 算子 kernel。

### 1.3 推理数据流

```
generate(prompt_tokens, max_tokens):
  ┌─ PREFILL (prompt_len > 1) ──────────────────────┐
  │ embedding → 28× [rms_norm → fused_qkv → rope    │
  │   → self_attention → residual_add                 │
  │   → fused_add_rmsnorm → gate+up → swiglu → down  │
  │   → residual_add] → rms_norm → lm_head → argmax  │
  │ KV Cache 填充完毕, current_pos += prompt_len      │
  └──────────────────────────────────────────────────┘
  ┌─ DECODE LOOP (逐 token) ────────────────────────┐
  │ embedding(1) → 28× [norm → q/k/v → rope          │
  │   → KV append → attention → residual              │
  │   → norm → mlp → residual] → norm → lm_head      │
  │   → argmax → yield token                          │
  │ current_pos += 1, 重复直到 EOS/max_tokens         │
  └──────────────────────────────────────────────────┘
```

---

## 2. FlashAttention v2 实现

**实现文件**: `src/ops/self_attention/nvidia/self_attention_nvidia.cu` (815 行)

### 2.1 双路径策略

| 路径 | 条件 | 优势 | 适用场景 |
|------|------|------|---------|
| **Naive Fused** | scores 全放 SMEM (≤48KB) | 零 HBM 读写 scores | decode (M=1), 短序列 |
| **Flash Tiled** | 任意长度 | 在线 softmax, O(√N) SMEM | prefill, 长序列 ≥12K |

### 2.2 在线 Softmax 算法

标准 softmax 需要两次全序列遍历 (求 max → 求 exp/sum)，FlashAttention 通过在线修正实现单次遍历：

```
初始化: m = -inf, l = 0, O_acc = 0

For each KV tile j ∈ [0, ⌈T/Bc⌉):
    // 1. 计算当前 tile 的 Q·K^T
    S_tile[Bc] = Q · K[j·Bc : (j+1)·Bc]^T × scale

    // 2. Causal mask
    for t in tile: if pos_t > pos_q: S_tile[t] = -inf

    // 3. 在线 softmax 更新
    tile_max = warp_max(S_tile)
    m_new = max(m, tile_max)
    rescale = exp(m - m_new)              // 历史修正因子

    // 4. 累积更新
    l = l × rescale + Σ exp(S_tile - m_new)
    O_acc = O_acc × rescale + Σ exp(S_tile - m_new) · V_tile
    m = m_new

Output = O_acc / l
```

### 2.3 实现要点

- **128 threads / 4 warps per block**, 每 block 处理 1 个 Q head
- **Block size**: Bc=32 (F32) / Bc=64 (FP16/BF16)
- **GQA**: `kv_head = q_head % n_kv_heads` (支持 Qwen2 12:2, LLaMA 32:8)
- **Warp 归约**: `__shfl_xor_sync(0xffffffff, val, off)` + `__shfl_down_sync` 实现 max/sum
- **Bank conflict 回避**: scores 存放 shared memory, 128-dim head broadcast
- **全精度**: F32 / F16 / BF16, 编译期模板特化

### 2.4 正确性验证

PyTorch `torch.nn.functional.scaled_dot_product_attention` 对照：
- F32: atol=1e-5, rtol=1e-5
- F16/BF16: atol=5e-3, rtol=5e-3
- 测试覆盖: causal mask, GQA, 序列长度 1~32K

---

## 3. INT8/INT4 量化优化

**实现文件**: `src/ops/linear_quantized/nvidia/linear_quantized_nvidia.cu` (714 行)

### 3.1 性能问题根因分析

原始 INT8 实现仅 **0.4 tok/s** (FP32 基线 34 tok/s)。通过系统性分析定位根因：

| 层次 | 问题 | 影响 |
|------|------|------|
| **L1: 显存分配** | 每次 GEMM 调用 `cudaMalloc/cudaFree` | ~50μs 同步开销 × 196 次/token |
| **L2: 冗余计算** | 每次 forward 对 196 个权重矩阵重复 dequant | 全部计算时间花在 dequant |
| **L3: 计算路径** | INT8→FP32→cublasSgemm (标量计算) | 未利用 FP16 Tensor Core (8× 理论差) |

### 3.2 解决方案 — FP16 持久权重缓存

```
原始流程 (每次 decode):
  INT8 权重 → cudaMalloc → dequant→FP32 → cublasSgemm → cudaFree

优化流程:
  首次: INT8 权重 → dequant→FP16 → 持久缓存 (static unordered_map<void*, half*>)
  后续: hash 查找 → cublasGemmEx FP16 TC (零额外开销)
```

**Vec4 Dequant Kernel**:
```cuda
__global__ void dequant_int8_to_fp16_vec4(const int8_t* in, half* out,
                                          const half* scales, int N, int K) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= N * K) return;
    uint32_t packed = *reinterpret_cast<const uint32_t*>(in + idx);
    int8_t v0 = packed & 0xFF;
    int8_t v1 = (packed >> 8) & 0xFF;
    int8_t v2 = (packed >> 16) & 0xFF;
    int8_t v3 = (packed >> 24) & 0xFF;
    half s = scales[idx / K];
    float sf = __half2float(s);
    out[idx]   = __float2half(v0 * sf);
    out[idx+1] = __float2half(v1 * sf);
    out[idx+2] = __float2half(v2 * sf);
    out[idx+3] = __float2half(v3 * sf);
}
```

### 3.3 优化历程

| 阶段 | tok/s | vs FP32 | 措施 |
|------|------:|------:|------|
| FP32 基线 | 14 | 1.0× | FP32 权重 + cublasSgemm |
| 原始 INT8 | 0.4 | 0.03× (bug) | 逐次 malloc + dequant → FP32 → cublasSgemm |
| +CachingAllocator | 2.6 | 0.19× | 消除 cudaMalloc 同步开销 |
| +FP16 Tensor Core | 3.2 | 0.23× | cublasGemmEx + CUBLAS_COMPUTE_32F |
| +持久权重缓存 | **57.5** | **4.1×** | dequant once → hash 查找复用 ← 根因修复 |
| +FP16 全计算管线 | 76 | 5.4× | embedding/norm/activation 全链路 FP16 |
| +KV Cache INT8 | **90** | **6.4×** | 正交量化叠加 |

### 3.4 INT4 Group Quantization

在 INT8 基础上进一步压缩: 2 个 INT4 打包为 1 个 uint8:

- **Group size** = 128 (每 128 元素共享 1 个 FP16 scale)
- **对称 absmax**: `scale = max(|values|) / 7`
- **压缩比**: 6.62 GB → 1.61 GB (**4.1×**)
- **推理路径**: 与 INT8 相同 (dequant→FP16 缓存→cublasGemmEx TC)

### 3.5 精度控制

- cuBLAS 使用 `CUBLAS_COMPUTE_32F` (非 FAST_16F/TF32 变体)
- FP32 累加保证精度 (atol=1e-5)
- `CUBLAS_COMPUTE_32F_FAST_16F` 在 4096×4096 矩阵上会出现 ~1e-4 误差

---

## 4. KV Cache INT8 量化

**实现文件**: `src/ops/kv_cache_quant/nvidia/kv_cache_quant_nvidia.cu`

### 4.1 动机

KV Cache 在长序列推理中占据大量显存:
- Qwen2-1.5B: 28 层 × 2 KV × 2 头 × 128 维 × 8192 长度 × 4B = **224 MB** (FP32)
- INT8 量化后: **58 MB** (75% 节省)

### 4.2 设计: Per-token Per-head 对称量化

```
写入时量化 (attention 计算新 KV 后):
  对每个 head (128 维):
    scale = max(|values|) / 127.0
    quantized[i] = round(values[i] / scale)  → int8
    存储: int8 values + fp32 scale per head

读取时反量化 (self_attention 前):
  output[i] = int8_values[i] * scale  → FP32/FP16/BF16
```

### 4.3 CUDA Kernel 设计

**量化 kernel** (`quantize_kv_cache_kernel`):
- 每 block 处理 1 个 head (128 元素)
- Pass 1: warp reduce (`__shfl_xor_sync`) 求 absmax → scale
- Pass 2: 量化写入 INT8

**反量化 kernel** (`dequantize_kv_cache_kernel`):
- 每线程 1 元素: `output[i] = int8_val[i] * scale`
- 支持输出到 F32/F16/BF16

### 4.4 与权重量化的正交性

| 维度 | 权重量化 (INT8/INT4) | KV Cache INT8 |
|------|---------------------|---------------|
| **开关** | 模型格式 (quantized=True) | 环境变量 `LLAISYS_KV_CACHE_INT8=1` |
| **数据** | 权重矩阵 → FP16 持久缓存 | KV 向量 → INT8 临时缓存 |
| **时机** | 模型加载时 dequant once | 每 token 实时量化/反量化 |
| **正交** | 零条件依赖，独立开关 | 零条件依赖，独立开关 |

### 4.5 性能与正确性

| 配置 | tok/s | vs INT8 only |
|------|------:|:-----------:|
| INT8 (W8A16) | ~76 | — |
| INT8 + KV INT8 | **~90** | **+19%** |

> FP32 + KV INT8 在 8GB 卡上因 VRAM 压力反而变慢 (14→8 tok/s)，KV INT8 收益在权重已量化后才充分释放。

**正确性**: INT8 + KV INT8 输出与 INT8-only 逐字符 **bit-identical** (greedy decode 验证)。

---

## 5. Prefill/Decode 分离

### 5.1 计算特征差异

| 特征 | Prefill (M>1) | Decode (M=1) |
|------|:------------:|:------------:|
| **瓶颈** | Compute-bound | Memory-bound |
| **GEMM 形状** | [M, K] × [K, N] (M≥100) | [1, K] × [K, N] (GEMV) |
| **Arithmetic Intensity** | 高 (可饱和 TC) | 低 (受限于 HBM 带宽) |
| **KV Cache** | 批量写入 | 单 token 追加 |

### 5.2 实现: 自动路由

```cpp
// src/models/qwen2/attention.hpp
tensor_t forward(tensor_t x, size_t start_pos, tensor_t pos_tensor) {
    size_t seq_len = x->shape()[0];
    if (seq_len > 1) {
        // PREFILL: Fused QKV projection → batch attention
        qkv_proj_.forward(ws_qkv_, x);   // [M, 3·head·dim] 单次 GEMM
        split_qkv_batch(ws_qkv_, q, k, v, seq_len);
    } else {
        // DECODE: Separate Q/K/V → GEMV (latency optimized)
        q_proj_.forward(ws_q_2d_, x);     // 3× 小 GEMV
        k_proj_.forward(ws_k_2d_, x);
        v_proj_.forward(ws_v_2d_, x);
    }
    // ... RoPE → cache append → attention → output projection
}
```

**Prefill 优化**: Fused QKV 将 3 个 GEMM 合并为 1 个，减少 kernel launch 和 HBM 读取。
**Decode 优化**: 独立投影避免 fused 路径的 split 开销，直接进入 GEMV。

---

## 6. 算子融合与内存管理

### 6.1 Fused Add + RMSNorm

`src/ops/fused_add_rmsnorm/nvidia/fused_add_rmsnorm_nvidia.cu`

将残差加法和 RMSNorm 融合为单 kernel:
- **Before**: `add(res, x, attn_out)` + `rms_norm(out, res, weight, eps)` = 3R+3W
- **After**: 单 kernel 内完成加法+归一化 = **2R+2W** (减少 33% HBM 流量)
- float4 向量化加载, L1 cache 局部性

### 6.2 CachingAllocator

`src/core/allocator/caching_allocator.cpp`

```cpp
class CachingAllocator {
    std::multimap<size_t, std::byte*> free_blocks_;  // size → ptr
    std::map<std::byte*, size_t> allocated_sizes_;    // ptr → size

    void* alloc(size_t size) {
        auto it = free_blocks_.lower_bound(size);     // Best-fit
        if (it != free_blocks_.end() && it->first <= 2 * size) {
            // 复用: 取出空闲块
        } else {
            // 新分配: cudaMalloc
        }
    }
    void dealloc(void* ptr) {
        // 不 cudaFree, 放回 free_blocks_ 等待复用
    }
};
```

- **Best-fit 策略**: `lower_bound(size)` 找最小满足块, `≤2×size` 防碎片
- **RAII 安全**: Storage + shared_ptr 自动释放
- **效果**: warmup 后 cache hit 率 100%, 消除推理期间所有 cudaMalloc/cudaFree

### 6.3 RMSNorm — Warp Shuffle Reduce

```cuda
// 两阶段归约: warp reduce → block reduce
float sum_sq = 0.0f;
for (int i = tid; i < hidden; i += blockDim.x)
    sum_sq += val[i] * val[i];

// Warp-level reduction
for (int off = 16; off > 0; off >>= 1)
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);

// Shared memory broadcast
if (lane_id == 0) smem[warp_id] = sum_sq;
__syncthreads();

// Block-level merge
if (warp_id == 0) {
    sum_sq = (lane_id < num_warps) ? smem[lane_id] : 0;
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
}
```

### 6.4 Argmax — 二阶段 Block Reduce

- **Phase 1**: 每 block 在 shared memory 内 warp reduce 求局部最大值 → 写入 temp buffer
- **Phase 2**: 单 block merge 所有局部结果 → 全局 argmax
- 避免 atomic 操作, 确定性结果

---

## 7. 模型适配

### 7.1 DeepSeek-R1-Distill-Qwen-1.5B

| 参数 | 值 |
|------|---|
| 层数 | 28 |
| Hidden Size | 1536 |
| Attention Heads | 12 (GQA 12:2) |
| Head Dim | 128 |
| MLP Intermediate | 8960 |
| RoPE Theta | 1,000,000 |

### 7.2 LLaMA-3.2-1B 架构适配

| 差异点 | Qwen2 | LLaMA 3.2 | 适配方式 |
|--------|-------|-----------|---------|
| GQA 比例 | 12:2 | 32:8 (4:1) | 通用 GQA `q_head % n_kv` |
| Attention Bias | 有 | 无 | 条件 bias 加载 |
| Tie Embeddings | 独立 lm_head | 共享 embed_tokens | 指针复用 |
| 层数 | 28 | 16 | Config 驱动 |

---

## 8. 性能汇总

### 推理速度 (RTX 4060 Ti 8GB)

| 模型 | 配置 | tok/s | 加速比 |
|------|------|------:|------:|
| Qwen2-1.5B | FP32 | ~14 | 1.0× |
| Qwen2-1.5B | INT8 (W8A16 + FP16 管线) | ~76 | 5.6× |
| Qwen2-1.5B | **INT8 + KV INT8** | **~90** | **6.6×** |
| Qwen2-1.5B | INT4 (W4A16) | ~33 | 2.4× |
| LLaMA-3.2-1B | FP32 | ~40 | — |
| HuggingFace (参考) | FP32/BF16 | ~32 | 0.35× |

### 正确性

- **8/8 算子测试通过** (F32/F16/BF16, PyTorch 对照)
- **端到端推理** 与 HuggingFace 输出一致 (greedy decode 比对)
- **KV Cache INT8**: INT8+KV INT8 与 INT8-only 输出 **bit-identical**
- **cuBLAS 精度**: `CUBLAS_COMPUTE_32F` 保证 atol=1e-5

---

## 9. Roofline 分析与性能剖析

### 9.1 理论 Roofline 分析

**硬件参数 (RTX 4060 Ti 8GB, Ada Lovelace)**:
- FP32 峰值算力: 22.06 TFLOPS
- FP16 Tensor Core 峰值: 176.5 TFLOPS
- HBM 带宽: 288.0 GB/s
- SM 数量: 34
- L2 Cache: 32 MB

**Decode 阶段 (M=1) 逐算子分析**:

| 算子 | 数据规模 | HBM 读写 (MB) | FLOPs | 算术强度 (FLOP/Byte) | 瓶颈类型 |
|------|----------|-------------:|------:|---------:|----------|
| Linear Q-proj | 1×1536 @ 1536×1536 | 9.0 | 4.7M | 0.50 | **Memory-bound** |
| Linear gate_proj | 1×1536 @ 1536×4096 | 24.0 | 12.6M | 0.50 | **Memory-bound** |
| Linear lm_head | 1×1536 @ 1536×151936 | 886.3 | 466.6M | 0.50 | **Memory-bound** |
| SelfAttention | M=1, T=100 | 3.0 | 0.9M | 0.29 | **Memory-bound** |
| RMSNorm | 1×1536 | 0.018 | 4.6K | 0.25 | **Memory-bound** |
| SwiGLU | 1×4096 | 0.048 | 4.1K | 0.08 | **Memory-bound** |

> FP32 Ridge Point = 76.6 FLOP/Byte, FP16 Ridge Point = 612.8 FLOP/Byte

**关键结论**: Decode 阶段所有算子的算术强度均远低于 Ridge Point，全部处于 **Memory-bound** 区域。这意味着：

1. **Decode 吞吐量的理论上限由 HBM 带宽决定**：每个 token 需读取 ~3,154 MB 权重数据
2. **理论极限**: 288 GB/s ÷ 3154 MB/token = **91.3 tok/s**
3. **实测 90 tok/s = 98.6% 理论带宽利用率** (L2 Cache 命中进一步提升小矩阵性能)

> **计算方法说明**: 3,154 MB/token 为 INT8 模式下所有 Linear 层 FP16 持久缓存权重的逐 token 读取总量 (INT8 权重在首次使用时 dequant 为 FP16 并永久缓存)。理论极限 = HBM 带宽 (288 GB/s) ÷ 每 token 总读取量。98.6% 表示实测吞吐与带宽理论极限之比，L2 Cache 对小矩阵 (RMSNorm 18KB, SwiGLU 48KB) 的命中率使实际 HBM 压力略低于理论值。

```
                   Roofline Model (RTX 4060 Ti)
    ┌─────────────────────────────────────────────────┐
    │              FP16 TC: 176.5 TFLOPS              │
    │                        ┌────────────────────────│
    │                       /                         │
    │ FP32: 22.06 T  ┌────/──────────────────────────│
    │               /│   /                            │
    │              / │  /    ← Ridge Point            │
    │    TFLOPS   /  │ /       FP32: 76.6             │
    │            /   │/        FP16: 612.8            │
    │           /    /                                │
    │          /    /│                                 │
    │         /   / │                                 │
    │   ★   /  /   │   ★ = All decode ops            │
    │      / /     │                                  │
    │     //       │                                  │
    │    /         │                                  │
    └─────────────────────────────────────────────────┘
              Arithmetic Intensity (FLOP/Byte)
```

### 9.2 Nsight Compute 实测 — RMSNorm 内核剖析

使用 `ncu` 对 Decode 阶段的 RMSNorm 内核进行实测 (M=1, hidden_size=1536, FP32):

```bash
ncu --target-processes all --launch-count 5 \
    -o report/ncu_ops python3 -c "import llaisys; ..."
```

**实测数据 (5 次运行平均)**:

| 指标 | 值 | 分析 |
|------|------|------|
| Block Size | 256 (8 warps) | — |
| Grid Size | **1** | Decode M=1 → 仅 1 行 → 1 block |
| Registers/Thread | 26 | 理论占用率 100% |
| **实际占用率** | **15.4%** | 8 warps / 48 max (1 block on 34 SMs) |
| **DRAM 吞吐量** | **~5% peak** | 数据量太小 (18 KB) |
| **SM 计算吞吐量** | **0.12% peak** | 计算量极少 → 确认 Memory-bound |
| Estimated Speedup | 97.06% | ncu: "Grid too small to fill GPU" |

**ncu 诊断原文**:
> *"This kernel grid is configured to execute only 1 block, which is less than the GPU's 34 multiprocessors. This can underutilize some multiprocessors."*

#### 9.2.1 分析

RMSNorm 在 Decode 阶段 (M=1) 的 Grid Size=1 是 **正确且不可避免** 的设计：
- 每行一个 block，M=1 就只有 1 个 block
- ncu 报告的 97% 优化空间是 **误导性的** — 这不是内核实现的问题，而是 Decode 工作负载的固有特征
- RMSNorm 每次调用仅处理 18 KB 数据 (1536×4 输入 + 1536×4 权重 + 1536×4 输出)
- 每 token 的 RMSNorm 总耗时: ~3.2μs × 56 次 = **0.18 ms** (仅占总推理时间的 1.6%)
- **真正的时间消耗在 Linear 算子**: 196 次 cuBLAS GEMV 占据 ~98% 推理时间

### 9.3 CUDA Graph 适用性分析

CUDA Graph 通过捕获 kernel launch 序列来消除逐次 launch 的 CPU 开销。
对本项目的适用性分析：

**理论收益估算**:
```
每 token kernel 启动数: ~308 次 (28 层 × 11 算子/层)
单次 launch 开销: 5-7 μs
总 launch 开销: ~1.5-2.0 ms / token
当前 token 时间: ~11.1 ms (90 tok/s)
预期加速: 13-18% → 102-106 tok/s
```

**实施约束**:

| 约束 | 说明 | 影响 |
|------|------|------|
| KV Cache 动态写入 | `update_cache()` 写入位置 = `start_pos`，每步递增 | 需要 kernel 间接寻址重写 |
| cuBLAS 兼容性 | cuBLAS 调用需要 `CUBLAS_WORKSPACE_CAPTURED` 模式 | 配置改动 |
| INT8 反量化缓存 | 首次推理时 dequant → 后续复用 FP16 缓存 | Graph 捕获时需确保缓存已填充 |
| KV Cache INT8 | quantize/dequantize 写入的位置同样动态 | 需配合位置 buffer |

**设计方案** (已论证，未实施):

```
// 方案: GPU-side 位置 buffer + 间接寻址
int* d_pos_buffer;  // GPU 上的位置 buffer
cudaMemcpyAsync(d_pos_buffer, &current_pos, sizeof(int), H2D);

// CUDA Graph 捕获
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  forward_decode(input, d_pos_buffer);  // 所有 kernel 读取 d_pos_buffer
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, 0);

// 推理循环
for (int step = 0; step < max_len; step++) {
    cudaMemcpyAsync(d_pos_buffer, &step, sizeof(int), H2D);
    cudaGraphLaunch(instance, stream);
}
```

**不实施的理由**:
1. **收益有限**: 已达 98.6% HBM 带宽利用率，加速主要来自消除 CPU launch gap
2. **实施复杂度高**: 需重写 KV Cache update、INT8 quantize/dequantize 的地址计算
3. **调试困难**: CUDA Graph 的错误信息不如普通 kernel 清晰
4. **ROI 不高**: 预期 13-18% 提升，但需 ~2000 行代码改动

### 9.4 进一步优化方向

基于 Roofline 分析，本项目 Decode 阶段已 **触及 HBM 带宽理论极限**。继续提升单请求 tok/s 的唯一路径：

| 方向 | 预期收益 | 原理 |
|------|----------|------|
| **FP8 量化 (Hopper+)** | ~2× | 权重体积减半 → HBM 读取减半 |
| **INT4 W4A16** | ~1.8× | 已实现但 dequant 开销大 (33 tok/s) |
| **Batch 推理** | N× | M=N 时算术强度线性增长，进入 Compute-bound |
| **Continuous Batching** | ∞ 吞吐 | 多请求共享 GPU → 利用全部 SM |
| **Speculative Decoding** | 2-3× latency | 小模型预测+大模型验证 |

> 参见 `scripts/profile_roofline.py` — 完整 Roofline 计算脚本

---

## 10. API 参考

### 9.1 C API

**Tensor**:
| 函数 | 说明 |
|------|------|
| `tensorCreate(shape, ndim, dtype, device, id)` | 创建张量 |
| `tensorLoad(tensor, data_ptr)` | 从 Host 加载数据 |
| `tensorView / tensorSlice / tensorPermute` | 零拷贝视图操作 |

**算子**:
| 函数 | 说明 |
|------|------|
| `llaisysLinear(out, in, weight, bias)` | cuBLAS Tensor Core GEMM |
| `llaisysLinearQuantized(out, in, weight, scales, bias)` | INT8 量化线性层 |
| `llaisysSelfAttention(attn_val, q, k, v, scale)` | FlashAttention v2 |
| `llaisysRmsNorm(out, in, weight, eps)` | RMS 归一化 |
| `llaisysROPE / llaisysEmbedding / llaisysSwiGLU / llaisysAdd / llaisysArgmax` | 其他算子 |

**运行时**:
```c
const LlaisysRuntimeAPI* llaisysGetRuntimeAPI(llaisysDeviceType_t);
void llaisysSetContextRuntime(llaisysDeviceType_t, int device_id);
```

### 9.2 Python API

```python
from llaisys.models import Qwen2
from llaisys.runtime import DeviceType

model = Qwen2("models/DeepSeek-R1-Distill-Qwen-1.5B", DeviceType.NVIDIA)
response = model.generate("你好", max_new_tokens=100)
for token in model.generate_stream("你好"):
    print(token, end="", flush=True)
model.reset()
```

### 9.3 Chat Server (OpenAI 兼容)

```bash
PYTHONPATH=python python -m llaisys.server.app --model <path> --device nvidia
```

| 端点 | 说明 |
|------|------|
| `POST /v1/chat/completions` | 聊天补全 (支持 stream: true) |
| `GET /v1/models` | 模型列表 |
| `GET /` | Web Chat UI |

---

## 11. 复现指南

### 10.1 环境要求

| 依赖 | 版本 |
|------|------|
| NVIDIA GPU | sm≥60, ≥6GB VRAM |
| CUDA Toolkit | ≥12.0 |
| xmake | ≥2.8 |
| Python | ≥3.9 |
| GCC | ≥9 (C++17) |

### 10.2 编译与安装

```bash
git clone https://github.com/xsmccc/llaisys.git && cd llaisys
xmake f --nv-gpu=y -cv && xmake build
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
pip install -e python/
```

### 10.3 模型下载与量化

```bash
# 下载
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B

# INT8 量化
PYTHONPATH=python python3 scripts/quantize_model.py \
    models/DeepSeek-R1-Distill-Qwen-1.5B \
    models/DeepSeek-R1-Distill-Qwen-1.5B-INT8

# INT4 量化
PYTHONPATH=python python3 scripts/quantize_model_int4.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 --group-size 128
```

### 10.4 运行测试

```bash
# 算子正确性 (8/8)
for test in test/ops/*.py; do PYTHONPATH=python:test python3 "$test" --device nvidia; done

# 端到端推理
PYTHONPATH=python python3 test/test_infer.py --device nvidia \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --test

# 最优配置推理
LLAISYS_KV_CACHE_INT8=1 PYTHONPATH=python python3 -c "
from llaisys.models import Qwen2; from llaisys import DeviceType
model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT8', DeviceType.NVIDIA, quantized=True)
for t in model.generate_stream('What is GPU computing?', max_new_tokens=64):
    print(t, end='', flush=True)
"
```

### 10.5 常见问题

| 问题 | 解决 |
|------|------|
| 推理速度异常慢 | 检查 .so 已同步到 `python/llaisys/libllaisys/` |
| INT8 首 token 延迟 | 正常: 首次 dequant 196 个权重到 FP16 缓存 (~200ms) |
| VRAM 不足 | 缩小 max_seq_len (4096→2048) |
| 算子精度不过 | 确认 CUBLAS_COMPUTE_32F (非 FAST 变体) |
