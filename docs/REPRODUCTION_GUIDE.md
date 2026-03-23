# LLAISYS 复现指南

---

## 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| OS | Linux / WSL2 (Ubuntu 20.04+) | |
| GPU | NVIDIA sm≥60 (推荐 sm≥70) | 需 ≥6GB VRAM |
| CUDA Toolkit | ≥12.0 | 含 cuBLAS |
| xmake | ≥2.8 | 构建工具 |
| Python | ≥3.9 | 推荐 3.12 |
| GCC | ≥9 | C++17 支持 |

---

## 2. 编译与安装

```bash
git clone https://github.com/xsmccc/llaisys.git
cd llaisys

# 启用 NVIDIA GPU 支持并编译
xmake f --nv-gpu=y -cv
xmake build

# 同步 .so 到 Python 包（必须在 pip install 之前）
cp build/linux/x86_64/release/libllaisys.so python/llaisys/libllaisys/

# 安装 Python 包（editable 模式，改代码后无需重装）
pip install -e python/
```

> **提示**: 安装后即可直接 `import llaisys`，后续所有命令均不需要设置 `PYTHONPATH`。
> 如果不想安装，也可以在每条命令前加 `PYTHONPATH=python:test`。

---

## 3. Python 依赖

```bash
pip install torch transformers accelerate safetensors numpy
```

---

## 4. 下载模型

```bash
pip install huggingface_hub

# Qwen2 (主要测试模型)
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B

# LLaMA 3.2 (可选)
huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir models/Llama-3.2-1B
```

---

## 5. 算子正确性测试

```bash
# 全部 8 个算子 (F32/F16/BF16, CPU + NVIDIA)
for test in test/ops/*.py; do
    python3 "$test" --device nvidia
done
# 预期输出: 全部 "Test passed!"
```

---

## 6. 端到端推理测试

```bash
# 与 HuggingFace 对比 (greedy decode)
python3 test/test_infer.py \
    --device nvidia --model models/DeepSeek-R1-Distill-Qwen-1.5B --test
```

---

## 7. FP32 推理

```bash
python3 -c "
from llaisys.models import Qwen2
from llaisys import DeviceType
from transformers import AutoTokenizer

model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B', DeviceType.NVIDIA)
tok = AutoTokenizer.from_pretrained('models/DeepSeek-R1-Distill-Qwen-1.5B')
ids = tok.encode('What is GPU computing?')
out = model.generate(ids, max_new_tokens=64, temperature=0.7)
print(tok.decode(out))
"
# 预期: ~34 tok/s
```

---

## 8. INT8 量化推理

```bash
# 步骤 1: 量化模型
python3 scripts/quantize_model.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT8

# 步骤 2: 推理
python3 -c "
from llaisys.models import Qwen2
from llaisys import DeviceType
from transformers import AutoTokenizer

model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT8',
              DeviceType.NVIDIA, quantized=True)
tok = AutoTokenizer.from_pretrained('models/DeepSeek-R1-Distill-Qwen-1.5B')
ids = tok.encode('What is GPU computing?')
out = model.generate(ids, max_new_tokens=64)
print(tok.decode(out))
"
# 预期: ~57-63 tok/s (首个 token 较慢，因 FP16 缓存 warmup)
```

---

## 9. INT4 量化推理

```bash
# 步骤 1: 量化 (约 10 分钟, ~8GB CPU RAM)
python3 scripts/quantize_model_int4.py \
    --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 \
    --group-size 128

# 步骤 2: 推理
python3 -c "
from llaisys.models import Qwen2
from llaisys import DeviceType
from transformers import AutoTokenizer

model = Qwen2('models/DeepSeek-R1-Distill-Qwen-1.5B-INT4',
              DeviceType.NVIDIA, int4=True)
tok = AutoTokenizer.from_pretrained('models/DeepSeek-R1-Distill-Qwen-1.5B')
ids = tok.encode('What is machine learning?')
out = model.generate(ids, max_new_tokens=100, temperature=0.7)
print(tok.decode(out))
"
# 预期: ~55-61 tok/s
```

---

## 10. LLaMA 3.2-1B 推理

```bash
python3 -c "
from llaisys.models import Llama3
from llaisys import DeviceType
from transformers import AutoTokenizer

model = Llama3('models/Llama-3.2-1B', device=DeviceType.NVIDIA, max_seq_len=4096)
tok = AutoTokenizer.from_pretrained('models/Llama-3.2-1B')
ids = tok.encode('Once upon a time')
out = model.generate(ids, max_new_tokens=100, temperature=0.7)
print(tok.decode(out, skip_special_tokens=True))
"
# 预期: ~40 tok/s
```

---

## 11. 启动 Chat Server

```bash
# 启动服务 (FP32)
python3 -m llaisys.server \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia

# 启动服务 (INT8 量化)
python3 -m llaisys.server \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B-INT8 --device nvidia --quantized
```

- **Web UI**：浏览器打开 `http://localhost:8000`
- **API 调用**：

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2","messages":[{"role":"user","content":"你好"}]}'

# Streaming (SSE)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

---

## 12. 性能基准

```bash
# 算子基准 (vs PyTorch)
python3 scripts/benchmark/benchmark_ops.py --device nvidia

# 推理基准 (FP32 / INT8 对比)
python3 scripts/benchmark/benchmark_inference.py --device nvidia
```
