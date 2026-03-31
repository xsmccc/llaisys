# LLAISYS — C++/CUDA LLM Inference Engine

从零构建的高性能 LLM 推理引擎，全部 CUDA kernel 自行实现，不依赖 vLLM / TensorRT-LLM 等推理框架。

当前支持 DeepSeek-R1-Distill-Qwen-1.5B 与 LLaMA-3.2-1B，在 RTX 4060 Ti 8GB 上 INT8 量化 + KV Cache INT8 + CUDA Graph 静态捕获达到 **132 tok/s**，相比 FP32 基线 30 tok/s 加速 **4.4×**，Graph 消除 91% kernel launch 开销（308 launches/token → 1 次 cudaGraphLaunch）。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 优化路径

从 FP32 基线到最优配置的完整优化链路：

```
FP32 Baseline (~30 tok/s)
  │
  ├─ INT8 权重量化 (W8A16)
  │   ├─ 初版实现:           0.4 tok/s  (每次 forward 196× 冗余 dequant + cudaMalloc)
  │   ├─ + CachingAllocator: 2.6 tok/s  (消除 cudaMalloc 同步开销)
  │   ├─ + FP16 Tensor Core: 3.2 tok/s  (cublasGemmEx FP16 计算)
  │   ├─ + 持久 FP16 权重缓存: 57 tok/s ← 根因修复 (dequant 一次，永久复用)
  │   └─ + FP16 全计算管线:   76 tok/s  (embedding/norm/activation 全链路 FP16)
  │
  ├─ + KV Cache INT8:        118 tok/s (正交叠加，KV 显存压缩 75%)
  └─ + CUDA Graph 静态捕获: 132 tok/s  (消除 91% kernel launch overhead, +12.2%)
```

### 主要优化发现

- **INT8 初版性能倒退 85×**：通过 profiling 定位到根因——每次 decode forward 调用 196 次 GEMM（28 层 × 7 矩阵），每次均执行 `cudaMalloc → dequant → cublasSgemm → cudaFree`。问题不在量化本身，而在冗余的 dequant 与显存分配。引入 persistent FP16 weight cache（static hashmap，启动时一次性 dequant）后直接从 0.4 提升至 57 tok/s。

- **KV Cache INT8 在 FP32 模式下反而退化**：8GB 显存中 FP32 权重已占用 6.6GB，KV INT8 的 dequant 临时缓冲区加剧显存压力，导致 14→8 tok/s。KV 量化的收益需要在权重已量化、显存充裕的条件下才能释放——INT8 权重 + KV INT8 = 118 tok/s。

- **Decode 阶段 Tensor Core 收益有限**：尝试 `cublasGemmEx + CUBLAS_COMPUTE_32F_FAST_TF32`，但 Decode 为 GEMV（M=1），完全 memory-bound，TC 仅带来约 2% 提升。

- **CUDA Graph 静态捕获 +12.2%**：每 token 约 308 次 kernel launch（28 层 × ~11 算子），每次 5-7μs CPU 开销合计约 1.8ms/token。三步解决动态参数：① 设备侧指针间接寻址（`*d_start_pos`）— Graph 录内存地址而非值；② `d_total_len` 传递给 FlashDecoding + idle block 早退（固定最大 Grid）；③ Pinned Memory + `cudaMemcpyAsync` 消除 default-stream 序列化（修复 FP32 -12% 回归）。实测 overhead：~131μs/token vs 原始 ~1850μs，节省约 93%。

---

## 核心实现

### FlashAttention v2

完整手写的 CUDA kernel（1150 行），实现 tiled online softmax，非调用第三方库。

采用双路径策略：
- **Naive fused path**：scores 全部驻留 shared memory，KV ≤ 12K tokens 时使用，decode（M=1）下为最优路径
- **Flash tiled path**：Bc 分块 + 在线 softmax 修正，支持 32K+ 任意长度序列

在线 softmax 的核心：每处理一个 KV tile 即更新全局 max 与 sum，通过 rescale factor `exp(m_old - m_new)` 修正已累积的 O 和 l，无需两次遍历。

实现细节：128 threads / 4 warps 处理一个 Q head，Bc=32（F32）/ 64（FP16/BF16），warp shuffle reduce（`__shfl_xor_sync` / `__shfl_down_sync`），causal mask，GQA 支持。

### INT8/INT4 量化

**INT8（W8A16）**：per-channel absmax 量化，dequant kernel 用 `uint32_t` 打包 4 个 INT8 一次读取，配合 FP16 persistent weight cache + cublasGemmEx Tensor Core 路径。

**INT4（W4A16）**：group quantization（group_size=128），每 byte 打包 2 个 4-bit 权重，模型从 6.62GB 压缩至 1.61GB（4.1×），共用 FP16 持久缓存路径，decode ~33 tok/s。

**KV Cache INT8**：per-token per-head 对称量化，与权重量化完全正交——独立开关，独立数据通路。量化使用 warp reduce 求 absmax，反量化逐元素乘 scale。

### CUDA Graph 静态捕获

完整实现 `CudaGraphManager`，Decode 阶段"捕获一次，永久重放"：

- **静态参数设计**：token id、position、start_pos、total_len 改为 GPU 侧设备内存间接寻址；每 token 仅需 4 次 `cudaMemcpyAsync`（pinned ~35μs）+ 1 次 `cudaGraphLaunch`（~130μs）
- **FlashDecoding 适配**：固定最大 Grid = (max_splits=32, nhead=12)，`d_total_len` 参数 + idle block `kv_len<=0` 早退
- **FP32 回归修复**：`cudaMallocHost` page-locked buffer + compute stream async H2D，消除原 `cudaMemcpy` 同步造成的 -12% 回归

实测：capture 仅一次 4012μs，后续 avg_launch=103μs。INT8+KV8: 117.6 → **132.0 tok/s (+12.2%)**。

### Prefill / Decode 分离

两个阶段计算特性不同，分别优化：
- **Prefill**（seq_len > 1）：compute-bound，采用 fused QKV 投影 + 批量 attention
- **Decode**（seq_len = 1）：memory-bound GEMV，拆分为独立 Q/K/V 投影，单 token 追加 KV cache

### 算子融合

`fused_add_rmsnorm` 将残差加法与 RMSNorm 融合为单 kernel，HBM 读写从 3R+3W 降至 2R+2W，减少 33% 显存带宽消耗。

---

## 性能分析

通过 Roofline Model 理论分析 + Nsight Compute 实测:

Decode 阶段所有算子的算术强度均低于 ridge point，全部处于 memory-bound 区域。HBM 带宽理论极限 288 GB/s ÷ 3154 MB/token ≈ **91.3 tok/s**；加入 CUDA Graph 后实测 **132 tok/s**，超出 HBM 理论极限——因 L2 Cache 对小矩阵（18-48KB，完整缓存于 32MB L2）的命中红利 + Graph 消除 CPU launch gap，GPU 持续满载。

> 带宽计算基于 INT8 + FP16 persistent cache 下的实际权重读取量。RMSNorm/RoPE 等小算子几乎全命中 L2 Cache，有效带宽超过 HBM 峰值带宽。No-Graph 约 118 tok/s，CUDA Graph 进一步减少 launch gap（~1.85ms/token → ~0.13ms/token），释放完整 GPU 算力达 132 tok/s。

详见 [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## 系统架构

```
Python Frontend (ctypes FFI)
  ├─ models/ (Qwen2, LLaMA3)
  ├─ server/ (FastAPI + SSE + Web UI)
  └─ ops.py / tensor.py
        │
    C API Layer (include/llaisys/)
        │
C++ Core
  ├─ Context            thread_local 设备上下文
  ├─ Runtime            流管理, 设备抽象
  └─ CachingAllocator   best-fit 显存池 (multimap<size_t, byte*>)
        │
CUDA Kernels
  ├─ self_attention      FlashAttn v2 (1150L)
  ├─ linear_quantized    INT8/INT4 + FP16 cache (714L)
  ├─ kv_cache_quant      KV Cache INT8 量化/反量化
  ├─ fused_add_rmsnorm   残差 + 归一化融合
  ├─ rms_norm / argmax   warp reduce
  └─ rope / swiglu / embedding / add
```

**内存管理**：CachingAllocator 使用 best-fit 策略，warmup 后 cache hit 率 100%，消除推理期间所有 `cudaMalloc/cudaFree` 开销。

---

## Quick Start

```bash
# 编译
git clone https://github.com/xsmccc/llaisys.git && cd llaisys
xmake f --nv-gpu=y -cv && xmake build
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/
pip install -e python/

# 下载模型 + INT8 量化
pip install torch transformers safetensors numpy
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B
PYTHONPATH=python python3 scripts/quantize_model.py \
    models/DeepSeek-R1-Distill-Qwen-1.5B \
    models/DeepSeek-R1-Distill-Qwen-1.5B-INT8

# 最优配置: INT8 + KV Cache INT8 + CUDA Graph (132 tok/s on RTX 4060 Ti)
LLAISYS_KV_CACHE_INT8=1 PYTHONPATH=python python3 -c "
from llaisys.models import Qwen2
from llaisys import DeviceType
model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT8',
              device=DeviceType.NVIDIA, quantized=True)
for token in model.generate_stream('What is GPU computing?', max_new_tokens=100):
    print(token, end='', flush=True)
"

# Chat Server (OpenAI 兼容 API)
pip install fastapi uvicorn
PYTHONPATH=python python3 -m llaisys.server.app \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia
# → http://localhost:8000 (Web UI) | POST /v1/chat/completions (API)
```

### 算子正确性验证

全部算子通过 PyTorch 对照测试（F32/F16/BF16）：

```bash
for test in test/ops/*.py; do PYTHONPATH=python:test python3 "$test" --device nvidia; done
# 8/8 PASSED — add, argmax, embedding, linear, linear_quantized, rms_norm, rope, self_attention
```

---

## 项目结构

```
src/
├── core/                    # Context, Runtime, CachingAllocator
├── ops/                     # CUDA 算子实现
│   ├── self_attention/      # FlashAttention v2 (1150L)
│   ├── linear_quantized/    # INT8/INT4 + FP16 持久缓存 (714L)
│   ├── kv_cache_quant/      # KV Cache INT8
│   ├── fused_add_rmsnorm/   # 残差 + 归一化融合
│   └── ...                  # rms_norm, argmax, linear, rope, swiglu, embedding, add
├── models/                  # Qwen2, LLaMA3 推理实现
└── tensor/                  # 多精度 Tensor (view/slice/permute, 引用计数)

python/llaisys/
├── models/                  # Python 模型封装
├── server/                  # FastAPI Chat Server + Web UI
└── libllaisys/              # ctypes FFI 绑定

test/ops/                    # 算子 PyTorch 对照测试
scripts/                     # 量化工具, 性能分析脚本
```

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA RTX 4060 Ti 8GB (sm_89, Ada Lovelace) |
| CUDA | 12.6 + cuBLAS |
| OS | Ubuntu 22.04 (WSL2) |
| 构建 | xmake + NVCC + GCC 11 (C++17) |

## License

[MIT](LICENSE)
