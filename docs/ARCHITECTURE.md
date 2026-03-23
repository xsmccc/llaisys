# LLAISYS 系统架构

## 整体架构图

```mermaid
graph TB
    subgraph "Python Frontend"
        PY_MODEL["models/<br/>Qwen2 / Llama3"]
        PY_OPS["ops.py<br/>tensor.py"]
        PY_SERVER["server/<br/>FastAPI + SSE"]
        PY_CTYPES["ctypes FFI<br/>libllaisys.so"]
        
        PY_MODEL --> PY_OPS
        PY_SERVER --> PY_MODEL
        PY_OPS --> PY_CTYPES
    end
    
    subgraph "C/C++ Backend"
        C_API["C API Layer<br/>include/llaisys/*.h"]
        
        subgraph "Core"
            CTX["Context<br/>thread_local<br/>设备上下文"]
            RT["Runtime<br/>stream / alloc /<br/>device mgmt"]
            CACHE["CachingAllocator<br/>best-fit 显存池"]
            STOR["Storage<br/>memory block<br/>引用计数"]
        end
        
        subgraph "Tensor"
            TENSOR["Tensor<br/>meta + storage + offset<br/>view/slice/permute"]
        end
        
        subgraph "Operators (CUDA)"
            LINEAR["linear<br/>cuBLAS TC GEMM"]
            QUANT["linear_quantized<br/>INT8/INT4 dequant<br/>+ FP16 cache"]
            ATTN["self_attention<br/>FlashAttn v2<br/>Naive + Tiled"]
            ROPE["rope<br/>旋转位置编码"]
            NORM["rms_norm<br/>layernorm variant"]
            EMB["embedding<br/>lookup table"]
            SWI["swiglu<br/>gate activation"]
            ADD["add<br/>element-wise"]
            ARG["argmax<br/>2-phase reduce"]
        end
        
        subgraph "Device Backends"
            NV["NVIDIA<br/>CUDA 12 / cuBLAS"]
            CPU["CPU<br/>stdlib / OpenBLAS"]
            MX["MetaX"]
            TS["Tianshu"]
        end
    end
    
    PY_CTYPES -->|"dlopen"| C_API
    C_API --> CTX
    CTX --> RT
    RT --> CACHE
    RT --> STOR
    CTX --> TENSOR
    C_API --> LINEAR & QUANT & ATTN & ROPE & NORM & EMB & SWI & ADD & ARG
    LINEAR & QUANT & ATTN & ROPE & NORM & EMB & SWI & ADD & ARG --> NV & CPU
    
    style NV fill:#76b900,color:#fff
    style CACHE fill:#ff6b35,color:#fff
    style ATTN fill:#0077b6,color:#fff
    style QUANT fill:#9b59b6,color:#fff
```

## 推理数据流

```mermaid
sequenceDiagram
    participant User
    participant Server as FastAPI Server
    participant Model as Qwen2 Model
    participant Ops as CUDA Operators
    participant GPU as GPU (VRAM)
    
    User->>Server: POST /v1/chat/completions
    Server->>Model: generate_stream(prompt)
    
    Note over Model: Tokenize → input_ids
    
    loop Prefill (all tokens)
        Model->>Ops: embedding(input_ids)
        Ops->>GPU: lookup weight table
        loop For each layer (28 layers)
            Model->>Ops: rms_norm → rope → self_attention
            Note over Ops: FlashAttn v2 (Naive if KV<12K)
            Model->>Ops: rms_norm → linear(gate) → swiglu → linear(down)
            Model->>Ops: add(residual)
        end
        Model->>Ops: rms_norm → linear(lm_head) → argmax
    end
    
    Note over Model: KV Cache populated
    
    loop Decode (1 token at a time)
        Model->>Ops: embedding → 28× (attn+mlp) → argmax
        Note over Ops: Decode: M=1 GEMV<br/>Memory-bound
        Model-->>Server: yield token (SSE)
        Server-->>User: data: {"token": "..."}
    end
```

## 内存管理流

```mermaid
graph LR
    subgraph "CachingAllocator"
        REQ["alloc(size)"]
        FREE_MAP["free_blocks_<br/>multimap&lt;size, ptr&gt;"]
        ACTIVE["allocated_sizes_<br/>map&lt;ptr, size&gt;"]
        
        REQ -->|"best-fit 查找<br/>(size ≤ 2× requested)"| FREE_MAP
        FREE_MAP -->|"命中 → 复用"| ACTIVE
        FREE_MAP -->|"未命中 → cudaMalloc"| ACTIVE
    end
    
    subgraph "释放流程"
        DEALLOC["dealloc(ptr)"]
        DEALLOC -->|"从 ACTIVE 移除<br/>放回 FREE_MAP"| FREE_MAP
    end
    
    subgraph "约束"
        MAX["MAX_CACHE = 4GB<br/>超限 → cudaFree"]
    end
```

## 量化推理路径

```mermaid
graph LR
    subgraph "INT8 路径 (W8A32)"
        W8["INT8 权重<br/>(npz 文件)"]
        DQ["dequant kernel<br/>INT8→FP16<br/>(首次调用)"]
        FC["FP16 Weight Cache<br/>(持久化, mutex保护)"]
        GEMM["cublasGemmEx<br/>FP16 Tensor Core"]
        
        W8 -->|"首次"| DQ --> FC
        FC -->|"后续直接复用"| GEMM
    end
    
    subgraph "FP32 路径"
        W32["FP32 权重"]
        SGEMM["cublasSgemm<br/>+ TC hint"]
        W32 --> SGEMM
    end
```

## 多设备抽象

```mermaid
classDiagram
    class LlaisysRuntimeAPI {
        <<C struct / 函数指针表>>
        +malloc_fn(size) void*
        +free_fn(ptr) void
        +memcpy_fn(dst, src, size, kind) void
        +memset_fn(dst, val, size) void
        +create_stream_fn() stream
        +sync_stream_fn(stream) void
    }
    
    class NvidiaRuntime {
        cudaMalloc / cudaFree
        cudaMemcpy
        cudaStreamCreate
    }
    
    class CpuRuntime {
        malloc / free
        memcpy / memset
        (no stream)
    }
    
    class Context {
        -thread_local device_type
        -thread_local device_id
        +setDevice(type, id)
        +getRuntime() LlaisysRuntimeAPI*
    }
    
    LlaisysRuntimeAPI <|.. NvidiaRuntime : 填充函数指针
    LlaisysRuntimeAPI <|.. CpuRuntime : 填充函数指针
    Context --> LlaisysRuntimeAPI : 持有当前设备的 API
```
