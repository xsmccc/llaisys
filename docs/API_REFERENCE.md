# LLAISYS API Reference

## C API (libllaisys.so)

### 类型定义

```c
// 设备类型
typedef enum {
    LLAISYS_DEVICE_CPU    = 0,
    LLAISYS_DEVICE_NVIDIA = 1,
    LLAISYS_DEVICE_METAX  = 2,
    LLAISYS_DEVICE_TIANSHU= 3,
} llaisysDeviceType_t;

// 数据类型
typedef enum {
    LLAISYS_DTYPE_I8=3, LLAISYS_DTYPE_I64=6, LLAISYS_DTYPE_F16=12,
    LLAISYS_DTYPE_F32=13, LLAISYS_DTYPE_BF16=19, /* ... */
} llaisysDataType_t;

// 内存拷贝方向
typedef enum {
    LLAISYS_MEMCPY_H2H=0, LLAISYS_MEMCPY_H2D=1,
    LLAISYS_MEMCPY_D2H=2, LLAISYS_MEMCPY_D2D=3,
} llaisysMemcpyKind_t;
```

### Tensor 接口

| 函数 | 签名 | 说明 |
|------|------|------|
| tensorCreate | `(shape, ndim, dtype, device_type, device_id) → tensor` | 创建张量 |
| tensorDestroy | `(tensor)` | 释放张量 |
| tensorLoad | `(tensor, data_ptr)` | 从 Host 加载数据 |
| tensorView | `(tensor, shape, ndim) → tensor` | 零拷贝 reshape |
| tensorPermute | `(tensor, order) → tensor` | 维度置换（转置） |
| tensorSlice | `(tensor, dim, start, end) → tensor` | 切片视图 |
| tensorIsContiguous | `(tensor) → bool` | 连续性检查 |
| tensorGetData | `(tensor) → void*` | 获取原始数据指针 |
| tensorGetShape | `(tensor, shape_out)` | 获取形状数组 |
| tensorGetStrides | `(tensor, strides_out)` | 获取步长数组 |

### 算子接口

| 函数 | 签名 | 说明 |
|------|------|------|
| llaisysLinear | `(out, in, weight, bias)` | 全连接: Y = XW^T + b (cuBLAS Tensor Core) |
| llaisysLinearQuantized | `(out, in, weight, scales, bias)` | INT8 量化线性层 (FP16 权重缓存) |
| llaisysSelfAttention | `(attn_val, q, k, v, scale)` | 自注意力 (FlashAttention v2 / Naive 双路径) |
| llaisysRmsNorm | `(out, in, weight, eps)` | RMS 归一化 |
| llaisysROPE | `(out, in, pos_ids, theta)` | 旋转位置编码 |
| llaisysEmbedding | `(out, index, weight)` | 嵌入查表 |
| llaisysSwiGLU | `(out, gate, up)` | SwiGLU 门控激活 |
| llaisysAdd | `(c, a, b)` | 逐元素加法 |
| llaisysArgmax | `(max_idx, max_val, vals)` | 二阶段归约求最大值 |

### 运行时接口

```c
// 函数指针表 (替代虚函数, C ABI 兼容)
struct LlaisysRuntimeAPI {
    int  (*get_device_count)();
    void (*set_device)(int);
    void (*device_synchronize)();
    llaisysStream_t (*create_stream)();
    void (*destroy_stream)(llaisysStream_t);
    void (*stream_synchronize)(llaisysStream_t);
    void *(*malloc_device)(size_t);
    void  (*free_device)(void*);
    void  (*memcpy_sync)(void*, const void*, size_t, llaisysMemcpyKind_t);
    void  (*memcpy_async)(void*, const void*, size_t, llaisysMemcpyKind_t, llaisysStream_t);
};

const LlaisysRuntimeAPI* llaisysGetRuntimeAPI(llaisysDeviceType_t);
void llaisysSetContextRuntime(llaisysDeviceType_t, int device_id);
```

---

## Python API

### 快速示例

```python
from llaisys.models import Qwen2
from llaisys.runtime import DeviceType

model = Qwen2("models/DeepSeek-R1-Distill-Qwen-1.5B", DeviceType.NVIDIA)

# 非流式
response = model.generate("你好", max_new_tokens=100)

# 流式
for token in model.generate_stream("你好", max_new_tokens=100):
    print(token, end="", flush=True)

# 重置对话记录
model.reset()
```

### Tensor

```python
from llaisys import Tensor, DataType, DeviceType

# 创建 GPU 张量
t = Tensor(shape=(2, 3, 4), dtype=DataType.F32, device=DeviceType.NVIDIA)

# 视图操作 (零拷贝)
t2 = t.view(6, 4)          # reshape
t3 = t.permute(2, 0, 1)    # 转置
t4 = t.slice(dim=0, start=0, end=1)  # 切片

# 检查
t.is_contiguous()  # → bool
t.shape()          # → (2, 3, 4)
t.dtype()          # → DataType.F32
```

### Ops (全部为静态方法)

```python
from llaisys import Ops, Tensor

Ops.linear(out, inp, weight, bias=None)
Ops.linear_quantized(out, inp, weight, scales, bias=None)
Ops.self_attention(attn_val, q, k, v, scale)
Ops.rms_norm(out, inp, weight, eps)
Ops.rope(out, inp, pos_ids, theta)
Ops.embedding(out, index, weight)
Ops.swiglu(out, gate, up)
Ops.add(c, a, b)
Ops.argmax(max_idx, max_val, vals)
```

### Chat Server (OpenAI 兼容)

```bash
PYTHONPATH=python python -m llaisys.server.app \
    --model models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia
```

**端点:**

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 聊天补全 (支持 `stream: true`) |
| GET  | `/` | Web Chat UI |

**请求示例:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Distill-Qwen-1.5B",
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": true,
    "max_tokens": 100
  }'
```
