# LLAISYS 项目报告

> **LLAISYS** — 从零构建的 C++/CUDA 大模型推理框架
>
> 硬件：NVIDIA RTX 4060 Ti 8GB (sm_89) | CUDA 12.6 | xmake 构建（A100服务器一直刷新也进不去）
>
> 代码量：~16,700 行 (C++/CUDA/Python)

---

## 目录

[TOC]



---

## 1. 项目概述与完成情况

| 项目 | 状态 | 核心内容 |
|------|:----:|----------|
| **#1 CPU 推理优化** | ✅ | OpenBLAS `cblas_sgemm` 替代朴素矩阵乘法 |
| **#2 CUDA 集成** | ✅ | 10 个 CUDA 算子 + NVIDIA Runtime API + cuBLAS Tensor Core |
| **#3 AI 聊天机器人** | ✅ | FastAPI 服务器 + SSE 流式 + Web UI + Temperature/Top-K/Top-P 采样 |
| **#4 多用户推理** | — | 未实现 |
| **#5 分布式推理** | — | 未实现 |
| **#6 新模型支持** | ✅ | 新增 Llama-3.2-1B 架构支持 |

### 支持矩阵

| 模型 | FP32 | INT8 | INT4 |
|------|:----:|:----:|:----:|
| DeepSeek-R1-Distill-Qwen-1.5B | ✅ 34 tok/s | ✅ 57–63 tok/s | ✅ 55–61 tok/s |
| Llama-3.2-1B | ✅ 40 tok/s | — | — |

> 测试环境：NVIDIA RTX 4060 Ti 8GB, CUDA 12.6, Ubuntu (WSL2)
>
> 沐曦 (MetaX) 显卡平台上能够正确实现推理

---

## 2. 框架架构

```
┌─────────────────────────────────────────────┐
│           Python Frontend (ctypes)          │
│  ├─ models/ (Qwen2, Llama3)                │
│  ├─ ops.py / tensor.py                     │
│  └─ server/ (FastAPI + OpenAI 兼容 API)    │
├─────────────────────────────────────────────┤
│           C API Layer (include/llaisys/)    │
├─────────────────────────────────────────────┤
│  C++ Core                                   │
│  ├─ Context (线程局部，多设备切换)          │
│  ├─ Runtime (流管理，内存分配，设备管理)     │
│  └─ CachingAllocator (GPU 显存池)           │
├─────────────────────────────────────────────┤
│  Operators           │  Device Backends     │
│  ├─ linear (cuBLAS)  │  ├─ NVIDIA (CUDA)    │
│  ├─ linear_quantized │  ├─ CPU (OpenBLAS)   │
│  ├─ self_attention   │  ├─ MetaX     │
│  ├─ rope / rms_norm  │  └─ Tianshu   │
│  ├─ embedding        │                      │
│  ├─ swiglu / add     │                      │
│  └─ argmax           │                      │
└─────────────────────────────────────────────┘
```

### 多设备策略模式

- `LlaisysRuntimeAPI` 定义统一接口（malloc, memcpy, stream, sync）
- 每种设备实现 `getRuntimeAPI()` 返回函数指针表
- `Context` 线程局部管理设备，`Runtime` 按设备类型惰性初始化
- 新增硬件只需实现 RuntimeAPI + 算子 kernel，无需改上层

---

## 3. 项目#1：CPU 推理优化

### 优化方案

CPU `linear` 算子使用 **OpenBLAS** 的 `cblas_sgemm` 替代朴素三重循环：

```cpp
// src/ops/linear/cpu/linear_cpu.cpp
// F32 路径：直接调用 BLAS
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    M, N, K,          // M=seq_len, N=out_features, K=in_features
    1.0f, input, K,   // A = input [M, K]
    weight, K,        // B = weight [N, K] (转置)
    0.0f, output, N); // C = output [M, N]
```

- **F16/BF16**：先转换为 F32 临时缓冲区 → `cblas_sgemm` → 转回目标类型
- xmake 中通过 `add_packages("openblas")` 链接

### 性能

OpenBLAS 会自动利用多线程 + CPU SIMD 指令（AVX2/AVX-512），相比朴素循环加速 **20–80×**（视核数和 SIMD 支持）。

---

## 4. 项目#2：CUDA 集成与算子实现

### 4.1 NVIDIA Runtime API

实现文件：`src/device/nvidia/nvidia_runtime_api.cu`

实现了完整的 CUDA Runtime API 适配：
- `malloc_host` / `malloc_device` — `cudaMallocHost` / `cudaMalloc`
- `free_host` / `free_device` — `cudaFreeHost` / `cudaFree`
- `memcpy_sync` / `memcpy_async` — `cudaMemcpy` / `cudaMemcpyAsync`
- `create_stream` / `destroy_stream` / `sync_stream` — CUDA 流管理
- `device_sync` — `cudaDeviceSynchronize`

通过 `ENABLE_NVIDIA_API` 宏和 `xmake f --nv-gpu=y` 开关控制编译。

### 4.2 CachingAllocator（GPU 显存池）

解决频繁 `cudaMalloc/cudaFree` 的 ~50μs/次同步开销：

```
allocate(size):
    在 free_blocks_ (multimap<size_t, byte*>) 中
    查找 [size, 2×size] 范围的空闲块
    → 命中: 复用旧块
    → 未命中: cudaMalloc 分配新块

release(ptr):
    不调用 cudaFree，放回 free_blocks_ 等待复用
```

效果：warmup 后 cache hit 率极高（822 blocks cached, 0 misses）。

### 4.3 CUDA 算子清单

共实现 **10 个 CUDA 算子**（3,591 行 CUDA 代码）：

| 算子 | 行数 | 数据类型 | 实现技术 |
|------|-----:|---------|----------|
| **linear** | 479 | F32/F16/BF16 | cuBLAS `cublasGemmEx` + Tensor Core + FP32 累加 |
| **linear_quantized** | 451 | INT8→FP16, INT4→FP16 | 持久 FP16 缓存 + vec4 dequant kernel |
| **self_attention** | 555 | F32/F16/BF16 | FlashAttention v2（在线 softmax + KV tiling）|
| **swiglu** | 648 | F32/F16/BF16 | fused gate × silu(up) kernel |
| **add** | 379 | F32/F16/BF16 | vectorized element-wise |
| **argmax** | 395 | F32 | 2-phase warp reduce |
| **rms_norm** | 260 | F32/F16/BF16 | fused parallel reduction |
| **embedding** | 232 | F32/F16/BF16 | index-based lookup kernel |
| **rope** | 192 | F32/F16/BF16 | cos/sin 旋转位置编码 |
| **rearrange** | — | all | stride-based memcpy |

### 4.4 FlashAttention v2 实现

`src/ops/self_attention/nvidia/self_attention_nvidia.cu`

**双路径策略**：
- **Naive 路径**：scores 数组 ≤ 48KB shared memory 时使用，适合 decode（M=1）
- **Flash 路径**：KV 分块处理，使用在线 softmax，支持 32K+ token

Flash 核心算法：
```
初始化: m = -inf, l = 0, O_acc = 0
对每个 KV tile [j*Bc, (j+1)*Bc):
    S_tile = Q · K_tile^T × scale
    causal mask: S[t] = -inf if pos_t > pos_q
    tile_max = block_max(S_tile)
    S_tile = exp(S_tile - tile_max), tile_sum = sum(S_tile)

    在线更新:
    m_new = max(m, tile_max)
    l = l × exp(m - m_new) + tile_sum × exp(tile_max - m_new)
    O_acc = O_acc × exp(m - m_new) + (S_tile · V_tile) × exp(tile_max - m_new)
    m = m_new
最终: O = O_acc / l
```

技术要点：128 threads/4 warps 每 block 处理 1 个 Q head；Block size Bc=32(F32)/64(FP16)；GQA 通过 `kv_head = q_head % n_kv_heads` 实现。

### 4.5 cuBLAS Tensor Core Linear

- 使用 `cublasGemmEx` + `CUBLAS_COMPUTE_32F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
- sm≥70 自动启用 TF32 TC，sm≥80 支持 FP16/BF16 TC
- Bias 通过 `float4` 向量化 kernel 高效处理
- 采用 `CUBLAS_COMPUTE_32F`（非 FAST 变体）确保精度满足 atol=1e-5

---

## 5. 项目#3：AI 聊天机器人

### 5.1 随机采样算子

在 `src/models/qwen2/qwen2.cpp` 中实现完整采样流水线：

```
Temperature 缩放 → Softmax → Top-K 过滤 → Top-P 过滤 → 多项式采样
```

- **Temperature**：`logits *= 1/T`，T>1 更随机，T<1 更确定
- **Top-K**：保留概率最高的 K 个 token
- **Top-P (Nucleus)**：保留累积概率 ≤ P 的最小 token 集
- **Multinomial**：从过滤后的分布中采样，支持可复现随机种子
- 当 `top_k=1` 自动退化为 argmax 贪心解码

### 5.2 Chat Server — OpenAI 兼容 API

```
python -m llaisys.server --model ./models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia
```

| 端点 | 说明 |
|------|------|
| `POST /v1/chat/completions` | 聊天补全（支持 streaming / non-streaming） |
| `GET /v1/models` | 模型列表 |
| `GET /` | Web 聊天 UI |

技术栈：**FastAPI** + **SSE (Server-Sent Events)** 流式响应 + **uvicorn** ASGI 服务器

安全措施：
- `MAX_INPUT_TOKENS = 4096` 防止过长输入 (DoS)
- `GENERATION_TIMEOUT_S = 300` 单次生成超时保护
- KV cache 溢出保护：自动截断输入并 clamp `max_tokens`
- `threading.Lock` 序列化模型访问

### 5.3 Web UI

内置 HTML/JS 单文件聊天界面（814 行），特性：
- **多会话管理**：新建 / 切换 / 删除对话，localStorage 持久化
- **Markdown 渲染**：集成 marked.js + highlight.js 代码高亮
- **DeepSeek 思维链**：自动折叠 `<think>...</think>` 推理过程
- **实时参数调节**：Temperature / Top-K / Top-P / Max Tokens 滑块
- **流式输出**：逐 token 实时显示，支持中途停止
- **代码复制**：一键复制代码块

---

## 6. 项目#6：新模型支持 — LLaMA 3.2-1B

### 架构差异对比

| 特性 | Qwen2-1.5B | LLaMA 3.2-1B |
|------|-----------|---------------|
| 层数 | 28 | 16 |
| Hidden Size | 1536 | 2048 |
| Attention Heads | 12 | 32 |
| KV Heads (GQA) | 2 | 8 |
| Head Dim | 128 | 64 |
| MLP Intermediate | 8960 | 8192 |
| RoPE Theta | 1,000,000 | 500,000 |
| Attention Bias | QKV 有 bias | 无 bias |
| Tie Embeddings | 无 | lm_head 共享 embed_tokens |

### 关键实现点

1. **`tie_word_embeddings`**：无独立 `lm_head.weight`，推理时直接复用 `embed_tokens` 指针
2. **无 attention bias**：简化权重映射和前向计算
3. **GQA 32:8 (4:1)**：每个 KV head 对应 4 个 Q head

### 新增文件

```
include/llaisys/models/llama3.h         # C API 头文件
src/models/llama3/
├── llama3_impl.hpp                     # Llama3Config
├── attention.hpp                       # Llama3Attention (无bias GQA)
├── mlp.hpp                             # Llama3MLP (无bias SwiGLU)
├── layer.hpp                           # Llama3DecoderLayer
└── llama3.cpp                          # 模型实现 + C 接口
python/llaisys/libllaisys/llama3.py     # ctypes 绑定
python/llaisys/models/llama3.py         # Python 模型类
```

---

## 7. INT8/INT4 量化推理

### 7.1 INT8 量化 (W8A32)

#### 问题

原始 INT8 实现仅 **0.4 tok/s**（FP32 基线 34 tok/s）。

#### 根因

1. 每次 forward 对 196 个权重矩阵重复 dequant
2. 每次 GEMM 调用 `cudaMalloc/cudaFree`
3. 未使用 Tensor Core（FP32 标量 22 TFLOPS vs FP16 TC 176.5 TFLOPS = 8× 差距）

#### 解决方案 — FP16 持久权重缓存

```
原始 (每次 decode):
  INT8 权重 → cudaMalloc → dequant→FP32 → cublasSgemm → cudaFree

优化 (首次 + 后续):
  首次: INT8 权重 → dequant→FP16 → 持久缓存 (static unordered_map)
  后续: 缓存命中 → cublasGemmEx FP16 TC (零额外开销)
```

核心 kernel：`dequant_int8_to_fp16_vec4` — 一次加载 4 个 INT8 (`uint32_t`)，向量化解包乘 scale。

#### 优化历程

| 阶段 | tok/s | 加速比 |
|------|------:|-------:|
| 原始 INT8 | 0.4 | 1× |
| + CachingAllocator | 2.6 | 6.5× |
| + FP16 Tensor Core | 3.2 | 8× |
| + 持久权重缓存 | **57.5** | **143×** |

### 7.2 INT4 量化 (W4A32)

在 INT8 基础上进一步压缩：2 个 INT4 打包为 1 个 `uint8`。

- Group size = 128（每 128 元素共用 1 个 FP16 scale）
- 对称 absmax 量化
- 压缩比：6.62 GB → 1.61 GB (**4.11×**)

Dequant kernel：
```cuda
uint8_t byte = packed[idx];
int4_lo = (byte & 0x0F) - 8;   // 低 nibble [-8, 7]
int4_hi = (byte >> 4) - 8;     // 高 nibble [-8, 7]
out[2k]   = __float2half(int4_lo * scale);
out[2k+1] = __float2half(int4_hi * scale);
```

推理路径与 INT8 相同：首次 dequant 到 FP16 持久缓存，后续直接用 `cublasGemmEx` FP16 TC。

---

## 8. 性能汇总

### 推理速度（RTX 4060 Ti 8GB）

| 模型 | 精度 | tok/s | VRAM（推理后）|
|------|------|------:|-------------:|
| DeepSeek-R1-Distill-Qwen-1.5B | FP32 | 34 | ~6.3 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | INT8 | 57–63 | ~6.3 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | INT4 | 55–61 | ~7.2 GB |
| Llama-3.2-1B | FP32 | 40 | ~2.5 GB |

> INT4 VRAM 高于 INT8 的原因：
> INT4 原始 (1.61 GB) + FP16 缓存 (3.31 GB) + KV = 7.2 GB；
> INT8 原始 (3.22 GB) + FP16 缓存 (3.31 GB) ≈ 6.5 GB。

### 正确性

- **8/8 算子测试通过**（F32/F16/BF16，NVIDIA + CPU）
- **端到端推理**与 HuggingFace 输出一致（greedy decode 比对）

---

## 9. 复现流程

### 9.1 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| OS | Linux / WSL2 (Ubuntu 20.04+) | |
| GPU | NVIDIA sm≥60 | 推荐 ≥6GB VRAM |
| CUDA Toolkit | ≥12.0 | 含 cuBLAS |
| xmake | ≥2.8 | 构建工具 |
| Python | ≥3.9 | 推荐 3.12 |
| GCC | ≥9 | C++17 |

### 9.2 编译

```bash
git clone https://github.com/xsmccc/llaisys.git && cd llaisys

# 配置 + 编译（启用 NVIDIA GPU）
xmake f --nv-gpu=y -cv
xmake build

# 同步 .so 到 Python 包（必须在 pip install 之前）
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/

# 安装 Python 包
pip install -e python/
```

### 9.3 Python 环境

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers accelerate safetensors numpy
```

### 9.4 下载模型

```bash
# 法一：HuggingFace
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B

# 法二：国内
pip install modelscope
modelscope download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local_dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 9.5 运行测试

```bash
# 算子正确性测试（全部 8 个）
for test in test/ops/*.py; do
    python3 "$test" --device nvidia
done

# 端到端推理正确性测试
python3 test/test_infer.py \
    --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --test
```

### 9.6 推理演示

```bash
# FP32 推理
python3 -c "
from llaisys.models import Qwen2
from llaisys import DeviceType
from transformers import AutoTokenizer

model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B', DeviceType.NVIDIA)
tok = AutoTokenizer.from_pretrained('models/DeepSeek-R1-Distill-Qwen-1.5B')
messages = [{'role': 'user', 'content': 'What is GPU computing?'}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
out = model.generate(ids, max_new_tokens=64, temperature=0.7)
print(tok.decode(out[len(ids):], skip_special_tokens=True))
"
```

### 9.7 INT8 量化

```bash
# 量化
python3 scripts/quantize_model.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT8

# 推理
python3 -c "
from llaisys.models import Qwen2; from llaisys import DeviceType
model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT8', DeviceType.NVIDIA, quantized=True)
"
```

### 9.8 INT4 量化

```bash
# 量化（约 10 分钟）
python3 scripts/quantize_model_int4.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 \
    --group-size 128

# 推理
python3 -c "
from llaisys.models import Qwen2; from llaisys import DeviceType
model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT4', DeviceType.NVIDIA, int4=True)
"
```

### 9.9 LLaMA 3.2-1B

```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B

python3 -c "
from llaisys.models import Llama3; from llaisys import DeviceType
from transformers import AutoTokenizer

model = Llama3('models/Llama-3.2-1B', device=DeviceType.NVIDIA, max_seq_len=4096)
tok = AutoTokenizer.from_pretrained('models/Llama-3.2-1B')
ids = tok.encode('Once upon a time')
out = model.generate(ids, max_new_tokens=100, temperature=0.7)
print(tok.decode(out, skip_special_tokens=True))
"
```

### 9.10 启动 Chat Server

```bash
python3 -m llaisys.server \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia

# 浏览器访问 http://localhost:8000
# 或 curl：
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 9.11 可能的问题

| 问题 | 解决 |
|------|------|
| 推理速度异常慢 | 检查 .so 是否已同步到 `python/llaisys/libllaisys/`；确认 `DeviceType.NVIDIA` |
| INT8 首个 token 很慢 | 正常。首次需将 196 个权重 dequant 到 FP16 缓存 (~200ms warmup) |
| VRAM 不足 | 缩小 `max_seq_len` 参数（4096→2048）降低 KV Cache 占用 |
| 算子精度不通过 | 确保编译使用 `CUBLAS_COMPUTE_32F`（非 FAST 变体） |

---

## 项目结构

```
llaisys/
├── include/llaisys/           # C API 头文件
│   └── models/                # 模型 C 接口 (qwen2.h, llama3.h)
├── src/
│   ├── core/                  # Context, Runtime, CachingAllocator, Tensor
│   ├── device/                # 设备 Runtime API (nvidia/, cpu/)
│   ├── ops/                   # 10 个算子 (每个含 cpu/ 和 nvidia/ 子目录)
│   ├── models/                # 模型 C++ 实现 (qwen2/, llama3/)
│   └── llaisys/               # C API 包装层
├── python/llaisys/
│   ├── libllaisys/            # ctypes 底层绑定
│   ├── models/                # 高级模型封装
│   ├── server/                # Chat Server (FastAPI + Web UI)
│   └── tensor.py / ops.py
├── test/                      # 算子 + 推理测试
├── scripts/                   # 量化工具 + 基准测试
├── xmake.lua                  # 构建配置
└── xmake/                     # 设备编译规则 (nvidia.lua, cpu.lua)
```
