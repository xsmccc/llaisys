'''
创建C数组等，调用函数，创建各种结构体，实现模型结构相关的数据类型创造，模型权重的装载以及generate函数的实现；
随后调用/python/llaisys/libllaisys/qwen2.py中进行类型验证和数据转换，最终由C++执行推理计算

'''


from typing import Sequence # 用于函数参数类型提示
from pathlib import Path # 更方便的文件路径操作
import ctypes   # 与C库交互
import json     # 读JSON配置文件
import numpy as np  # 数组操作
import torch    # PyTorch 
from safetensors.torch import load_file # 加载权重文件

# 从LLAISYS库导入C接口
# LIB_LLAISYS：共享库对象，包含所有C函数
# DeviceType：设备类型 (CPU / NVIDIA)
# DataType：数据类型 (F32等)
from ..libllaisys import LIB_LLAISYS, DeviceType, DataType 

# 从C绑定导入结构体
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, TensorHandle

try:
    from ..tensor import Tensor
except ImportError:
    Tensor = None


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        """Initialize Qwen2 model and load weights from safetensors"""
        self.model_path = Path(model_path) # 模型文件夹路径
        self.device = device #计算设备
        self._kept_references = [] # 保持python对象引用，防止被回收

        # 读取config.json文件
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建Meta结构体
        self.meta = LlaisysQwen2Meta() # C结构体，传给C++后端
        # 填充结构体字段
        self.meta.dtype = 0  # F32
        self.meta.nlayer = config["num_hidden_layers"]  # 层数 (28)
        self.meta.hs = config["hidden_size"]            # 隐层大小 (2048)
        self.meta.nh = config["num_attention_heads"]    # 注意力头数 (12)
        self.meta.nkvh = config["num_key_value_heads"]  # KV头数 (12)
        self.meta.di = config["intermediate_size"]      # MLP中间层大小
        self.meta.maxseq = config["max_position_embeddings"]  # 最大序列长度
        self.meta.voc = config["vocab_size"]            # 词表大小
        self.meta.epsilon = config["rms_norm_eps"]      # RMSNorm epsilon
        self.meta.theta = config["rope_theta"]          # RoPE theta

        # 处理EOS (End of Sequence) 令牌ID
        eos_id = config.get("eos_token_id", 151643)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]  # 如果是列表，取第一个
        self.meta.end_token = eos_id    # 保存EOS令牌ID
        
        # 计算每个头的维度 = 隐层大小 / 头数
        self.meta.dh = self.meta.hs // self.meta.nh

        # 创建C++后端模型
        print(f"[Qwen2] Initializing C++ Backend...")
        print(f"        Layers: {self.meta.nlayer}, Hidden: {self.meta.hs}, Heads: {self.meta.nh}")

        # 调用C函数创建模型
        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),  # 指向meta结构体的指针
            device.value,             # 设备类型的值（CPU=0, NVIDIA=1）
            None,                     # 不传初始权重（后面再加载）
            0                         # 权重数量 = 0
        )

        # 检查是否创建成功
        if not self.handle:
            raise RuntimeError("Failed to create C++ model!")

        # 获取权重结构体 .content是解引用指针，得到实际的结构体
        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        # 加载权重
        print(f"[Qwen2] Loading weights from {model_path}...")
        self._load_safetensors(self.model_path)

    # 析构函数 
    def __del__(self):
        """销毁C++模型，释放资源"""
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle) #调用C函数销毁模型，释放内存

    # 从safetensors文件加载权重
    def _load_safetensors(self, path: Path):
        """Load weights from safetensors files"""
        # 找到所有.safetensors文件 如果模型很大，可能分成多个文件 (model-00001.safetensors等)
        files = sorted(path.glob("*.safetensors"))
        if not files:
            print(f"Warning: No safetensors found in {path}")
            return
        # 逐个加载每个文件
        for file_path in files:
            print(f"  Loading {file_path.name}...")
            weights_dict = load_file(str(file_path))
            # 对每个权重，调用_map_weight映射到C结构体
            for name, tensor in weights_dict.items():
                self._map_weight(name, tensor)

    # 将PyTorch张量转换并映射到C结构体
    def _map_weight(self, name: str, tensor: torch.Tensor):
        """Convert torch tensor to Tensor and map to model"""
        # 转换数据类型
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor = tensor.float()
        
        # PyTorch tensor → numpy数组
        array = tensor.numpy()

        # norm层的权重必须是1D的，但有时可能被保存为2D
        is_norm = "norm" in name and "weight" in name
        if is_norm and array.ndim > 1:
            array = array.reshape(-1)

        # 确保内存连续
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        # 如果没有导入Tensor类，直接返回
        if Tensor is None:
            return

        try:
            # 创建Tensor对象
            llaisys_tensor = Tensor(
                shape=array.shape,      # 形状
                dtype=DataType.F32,     # 数据类型
                device=DeviceType.CPU   # 放在CPU上先
            )
            # 获取numpy数组的内存地址
            data_ptr = array.ctypes.data_as(ctypes.c_void_p)
            # 将numpy数据加载到LLAISYS Tensor
            llaisys_tensor.load(data_ptr)
        except Exception as e:
            print(f"Error creating tensor for {name}: {e}")
            return

        # 保留Python引用
        self._kept_references.append(llaisys_tensor)
        
        # 获取C指针
        ptr = llaisys_tensor.lib_tensor()

        # 映射到权重结构体
        w = self.c_weights

        # 嵌入层权重
        if name == "model.embed_tokens.weight":
            w.in_embed = ptr
        elif name == "model.norm.weight":
            w.out_norm_w = ptr
        elif name == "lm_head.weight":
            w.out_embed = ptr

        # Transformer层权重
        elif name.startswith("model.layers."):
            parts = name.split('.')
            try:
                layer_idx = int(parts[2]) # 提取层号 (0-27)
                suffix = ".".join(parts[3:])  # 后缀部分
            except:
                return

            if layer_idx >= self.meta.nlayer:
                return

            # 注意力层权重 (28层 x 4个投影)
            if suffix == "self_attn.q_proj.weight":
                w.attn_q_w[layer_idx] = ptr   # Query投影
            elif suffix == "self_attn.k_proj.weight":
                w.attn_k_w[layer_idx] = ptr   # Key投影
            elif suffix == "self_attn.v_proj.weight":
                w.attn_v_w[layer_idx] = ptr   # Value投影
            elif suffix == "self_attn.o_proj.weight":
                w.attn_o_w[layer_idx] = ptr   # Output投影
            
            # 注意力层偏置
            elif suffix == "self_attn.q_proj.bias":
                w.attn_q_b[layer_idx] = ptr
            elif suffix == "self_attn.k_proj.bias":
                w.attn_k_b[layer_idx] = ptr
            elif suffix == "self_attn.v_proj.bias":
                w.attn_v_b[layer_idx] = ptr
            
            # 层归一化权重
            elif suffix == "input_layernorm.weight":
                w.attn_norm_w[layer_idx] = ptr
            elif suffix == "post_attention_layernorm.weight":
                w.mlp_norm_w[layer_idx] = ptr
            # MLP层权重 (3个投影)
            elif suffix == "mlp.gate_proj.weight":
                w.mlp_gate_w[layer_idx] = ptr   # Gate投影
            elif suffix == "mlp.up_proj.weight":
                w.mlp_up_w[layer_idx] = ptr # Up投影
            elif suffix == "mlp.down_proj.weight":
                w.mlp_down_w[layer_idx] = ptr   # Down投影

    def generate(
        self,
        inputs: Sequence[int],  # 输入token IDs，例如 [151644, 8948, 198, ...]
        max_new_tokens: int = 128,  # 最多生成128个新tokens
        top_k: int = 1, # top-k采样 (1=贪心)
        top_p: float = 0.8,  # top-p采样
        temperature: float = 0.8,   # 温度参数
    ) -> list:
        """
        生成tokens
        
        LLM的推理分为两个阶段：
        
        1. Prefill (预填充): 一次性处理所有输入tokens
           - 目的：计算输入的所有KV缓存
           - 速度快（并行处理）
        
        2. Decoding (解码): 逐个生成新tokens
           - 目的：根据KV缓存，每次生成一个token
           - 速度慢（逐个处理）
        """
        """Generate tokens using the model"""
        current_ids = list(inputs)

        # 创建C数组类型（用于ctypes与C交互）
        InArrayType = ctypes.c_int64 * 1

        # Prefill - 处理输入tokens（除最后一个）
        for i in range(len(current_ids) - 1):
            token = current_ids[i]
            in_ptr = InArrayType(token) # 包装成C数组
            
            # 调用C++推理函数
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle,    # 模型句柄
                in_ptr, # 输入token
                ctypes.c_size_t(1)  # token数量=1
            )
                #更新KV缓存

        # 处理最后一个输入token并获取第一个预测
        last_input = current_ids[-1]
        in_ptr = InArrayType(last_input)

        # 调用C函数，这次有返回值
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.handle,
            in_ptr,
            ctypes.c_size_t(1)
        )
        # 将预测的token追加到列表
        current_ids.append(next_token)

        # Decoding - 逐个生成剩余的tokens
        for _ in range(max_new_tokens - 1):
            # 用前一个token作为输入
            in_ptr = InArrayType(next_token)

            
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle,
                in_ptr,
                ctypes.c_size_t(1)
            )

            current_ids.append(next_token)

            # 提前停止条件 如果预测的token是EOS 
            if next_token == self.meta.end_token:
                break
        # 返回完整的token序列
        return current_ids
