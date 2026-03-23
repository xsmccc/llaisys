# LLAISYS — C++/CUDA LLM Inference Engine

> **Let's Learn AI SYStem**: 从零构建的高性能大语言模型推理框架

[![Build](https://img.shields.io/badge/build-xmake-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen)](LICENSE)

## Highlights

- **纯 C++17 / CUDA 实现** — 不依赖 PyTorch、TensorRT 等推理框架
- **10 个手写 CUDA 算子** — 含 FlashAttention v2、cuBLAS Tensor Core GEMM
- **INT8 / INT4 量化推理** — W8A32 通过 FP16 权重缓存实现 **143× 加速**（0.4 → 57 tok/s）
- **多设备抽象** — 函数指针表 (`LlaisysRuntimeAPI`) 支持 NVIDIA / CPU / MetaX / Tianshu
- **CachingAllocator** — GPU 显存池化，best-fit 分配策略，避免 cudaMalloc 同步开销
- **OpenAI 兼容 API** — FastAPI + SSE 流式响应 + Web UI

## Benchmark (RTX 4060 Ti 8GB, sm_89)

| 配置 | Decode TPS | TTFT (3 tok) | TTFT (9 tok) | VRAM 占用 |
|------|-----------|-------------|-------------|-----------|
| FP32 | 24.65 | 99.8 ms | 274.3 ms | 7.7 GB |
| INT8 | 35.07 | 63.4 ms | 159.9 ms | 3.2 GB (load) → 6.4 GB (peak) |
| INT4 | 32.90 | 70.1 ms | 167.8 ms | 2.5 GB (load) → 5.7 GB (peak) |

> FP32→INT8: Decode **1.42×** 加速, VRAM 加载体积 **2.4×** 压缩, TTFT **1.57×** 加速

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python Frontend                                        │
│  ├─ models/ (Qwen2-1.5B, Llama3-1B)                    │
│  ├─ ops.py / tensor.py (ctypes FFI bindings)            │
│  └─ server/ (FastAPI + OpenAI-compat chat API)          │
├─────────────────────────────────────────────────────────┤
│  C API Layer (include/llaisys/)                         │
│  └─ __export 函数 → 编译为 libllaisys.so               │
├─────────────────────────────────────────────────────────┤
│  C++ Core                                               │
│  ├─ Context   — thread_local 设备上下文                 │
│  ├─ Runtime   — stream, allocator, device 管理          │
│  └─ CachingAllocator — GPU 显存池, best-fit, 2× 复用   │
├─────────────────────────────────────────────────────────┤
│  CUDA Operators                │  Device Runtime        │
│  ├─ linear (cuBLAS TC)         │  ├─ NVIDIA (CUDA 12)   │
│  ├─ linear_quantized (INT8/4)  │  ├─ CPU (stdlib)       │
│  ├─ self_attention (Flash v2)  │  ├─ MetaX              │
│  ├─ rope / rms_norm            │  └─ Tianshu            │
│  ├─ embedding / swiglu / add   │                        │
│  └─ argmax (2-phase reduce)    │                        │
└─────────────────────────────────────────────────────────┘
```

## Supported Models

| Model | Architecture | FP32 | INT8 | INT4 |
|-------|-------------|:----:|:----:|:----:|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | ✅ | ✅ | ✅ |
| Llama-3.2-1B | Llama3 | ✅ | — | — |

## Quick Start

```bash
# 1. 编译
xmake f --nv-gpu=y -cv
xmake build

# 2. 部署共享库
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/

# 3. Python 环境
python -m venv venv && source venv/bin/activate
pip install torch transformers safetensors numpy fastapi uvicorn

# 4. 运行推理
PYTHONPATH=python python -c "
from llaisys.models import Qwen2
from llaisys.runtime import DeviceType
model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B', DeviceType.NVIDIA)
for token in model.generate_stream('Hello!', max_new_tokens=50):
    print(token, end='', flush=True)
"

# 5. 启动 Chat Server
PYTHONPATH=python python -m llaisys.server.app \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia
# 访问 http://localhost:8000/docs 查看 OpenAI-compat API
```

## Key Technical Details

### FlashAttention v2 (自实现)
- **双路径策略**: Naive (KV < 12K, 共享内存放完整 scores) + Flash (任意长度, Bc=32/64 tiled online softmax)
- 支持 F32 / F16 / BF16, MHA + GQA, 单 kernel 前向
- Decode 场景 (M=1) 走 Naive 路径更快；Prefill / 长序列走 Flash 路径

### INT8 量化优化
- **问题**: 原始逐次 dequant 整个权重矩阵 → 0.4 tok/s
- **方案**: 持久化 FP16 权重缓存 (首次 dequant, 后续直接 cublasGemmEx FP16 Tensor Core)
- **结果**: 143× 加速 (0.4 → 57.5 tok/s), 甚至 1.7× 超越 FP32

### CachingAllocator
- `std::multimap<size_t, byte*>` best-fit 空闲链
- 2× 大小容忍度复用、4GB 上限、自动清理
- 避免 cudaMalloc 同步开销 (~1ms/call → amortized ~0)

### 多设备抽象
- `LlaisysRuntimeAPI` 函数指针表替代虚函数 (C ABI 兼容, 无 vtable 开销)
- 编译期 xmake target 选择, 运行期 `Context::setDevice()` 切换

## Project Structure

```
include/          C API 头文件 (__export 函数声明)
src/
  ├─ core/        Context, Runtime, CachingAllocator, Storage
  ├─ tensor/      Tensor 类 (meta + storage + offset)
  ├─ ops/         10 个算子 (每算子 cpu/ + nvidia/ 子目录)
  ├─ models/      Qwen2/Llama3 C++ 模型实现
  └─ device/      设备特定资源管理
python/
  ├─ llaisys/     Python 包 (ctypes bindings + models + server)
  └─ libllaisys/  编译好的 .so 放置目录
test/             Python 测试 (算子 + 推理 + tensor)
scripts/          Benchmark, profiling, quantization 脚本
docs/             技术报告, 复现指南, 原始作业文档
```

## Documentation

| 文档 | 说明 |
|------|------|
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | 完整技术报告 (核心实现 + 优化历程 + 性能数据) |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构图 (Mermaid: 数据流 + 内存管理 + 量化路径) |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | C / Python / HTTP 全量 API 参考 |
| [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) | 环境搭建 + 编译 + 测试 + 部署完整流程 |
| [ASSIGNMENTS.md](docs/ASSIGNMENTS.md) | 原始课程作业说明文档 |

## Tech Stack

| 类别 | 技术 |
|------|------|
| 语言 | C++17, CUDA C++, Python 3 |
| 构建 | xmake |
| GPU | CUDA 12.6, cuBLAS, Tensor Core (sm_89) |
| 量化 | W8A32 (INT8), W4A32 (INT4), 自实现 dequant kernel |
| 前端 | Python ctypes FFI, FastAPI, SSE |
| 测试 | PyTorch 对照验证, numpy, atol/rtol 精度门控 |

## License

[Apache 2.0](LICENSE)
