"""
LLaMA 3 模型 Python 封装

与 Qwen2 的架构差异:
  1. Tied embeddings: in_embed == lm_head (无独立 lm_head.weight)
  2. 无 attention bias: Q/K/V/O 投影无 bias
  3. 不同的 GQA 配置: 32 heads / 8 KV heads (4:1)
  4. 不同的 RoPE theta: 500000 vs 10000
  5. head_dim=64 (vs Qwen2 128)
"""

from typing import Sequence
from pathlib import Path
import ctypes
import json
import numpy as np
import torch
from safetensors.torch import load_file

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.llama3 import LlaisysLlama3Meta, LlaisysLlama3Weights

try:
    from ..tensor import Tensor
except ImportError:
    Tensor = None


class Llama3:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU,
                 max_seq_len: int = 8192):
        self.model_path = Path(model_path)
        self.device = device
        self._kept_references = []

        # 读取 config.json
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)

        # 创建 Meta 结构体
        self.meta = LlaisysLlama3Meta()
        self.meta.dtype = 0  # F32
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config.get("num_key_value_heads", self.meta.nh)
        self.meta.di = config["intermediate_size"]
        # head_dim: LLaMA3 config 可能直接给出, 否则计算
        self.meta.dh = config.get("head_dim", self.meta.hs // self.meta.nh)

        model_max_seq = config.get("max_position_embeddings", 131072)
        actual_max_seq = min(max_seq_len, model_max_seq)
        if actual_max_seq < model_max_seq:
            print(f"[LLaMA3] KV cache limited to {actual_max_seq} tokens "
                  f"(model supports {model_max_seq})")
        self.meta.maxseq = actual_max_seq
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config.get("rms_norm_eps", 1e-5)
        self.meta.theta = config.get("rope_theta", 500000.0)

        # EOS token
        eos_id = config.get("eos_token_id", 128001)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.meta.end_token = eos_id

        # Tied embeddings
        self.meta.tie_embeddings = 1 if config.get("tie_word_embeddings", True) else 0

        print(f"[LLaMA3] Initializing C++ Backend...")
        print(f"         Layers={self.meta.nlayer}, Hidden={self.meta.hs}, "
              f"Heads={self.meta.nh}/{self.meta.nkvh}, HeadDim={self.meta.dh}")
        print(f"         Vocab={self.meta.voc}, TiedEmbed={bool(self.meta.tie_embeddings)}")

        # 创建 C++ 模型
        self.handle = LIB_LLAISYS.llaisysLlama3ModelCreate(
            ctypes.byref(self.meta),
            device.value,
            None, 0
        )
        if not self.handle:
            raise RuntimeError("Failed to create LLaMA3 C++ model!")

        self.c_weights = LIB_LLAISYS.llaisysLlama3ModelWeights(self.handle).contents

        print(f"[LLaMA3] Loading weights from {model_path}...")
        self._load_safetensors(self.model_path)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysLlama3ModelDestroy(self.handle)

    def _create_tensor_from_numpy(self, array: np.ndarray):
        if Tensor is None:
            return None
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        if array.dtype == np.float32:
            dtype = DataType.F32
        elif array.dtype == np.float16:
            dtype = DataType.F16
        elif array.dtype == np.int8:
            dtype = DataType.I8
        else:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

        llaisys_tensor = Tensor(shape=array.shape, dtype=dtype, device=self.device)
        data_ptr = array.ctypes.data_as(ctypes.c_void_p)
        llaisys_tensor.load(data_ptr)
        self._kept_references.append(llaisys_tensor)
        return llaisys_tensor.lib_tensor()

    def _load_safetensors(self, path: Path):
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        w = self.c_weights
        has_lm_head = False

        for file_path in files:
            print(f"  Loading {file_path.name}...")
            weights_dict = load_file(str(file_path))

            for name, tensor in weights_dict.items():
                # 转为 float32
                if tensor.dtype in [torch.bfloat16, torch.float16]:
                    tensor = tensor.float()
                array = tensor.numpy()

                # Norm 权重必须为 1D
                if "norm" in name and "weight" in name and array.ndim > 1:
                    array = array.reshape(-1)
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)

                ptr = self._create_tensor_from_numpy(array)
                if ptr is None:
                    continue

                # 映射到权重结构体
                if name == "model.embed_tokens.weight":
                    w.in_embed = ptr
                    # Tied embeddings: 也设为 out_embed
                    if self.meta.tie_embeddings:
                        w.out_embed = ptr
                elif name == "lm_head.weight":
                    w.out_embed = ptr
                    has_lm_head = True
                elif name == "model.norm.weight":
                    w.out_norm_w = ptr
                elif name.startswith("model.layers."):
                    parts = name.split('.')
                    try:
                        layer_idx = int(parts[2])
                        suffix = ".".join(parts[3:])
                    except Exception:
                        continue
                    if layer_idx >= self.meta.nlayer:
                        continue

                    if suffix == "self_attn.q_proj.weight":
                        w.attn_q_w[layer_idx] = ptr
                    elif suffix == "self_attn.k_proj.weight":
                        w.attn_k_w[layer_idx] = ptr
                    elif suffix == "self_attn.v_proj.weight":
                        w.attn_v_w[layer_idx] = ptr
                    elif suffix == "self_attn.o_proj.weight":
                        w.attn_o_w[layer_idx] = ptr
                    elif suffix == "input_layernorm.weight":
                        w.attn_norm_w[layer_idx] = ptr
                    elif suffix == "post_attention_layernorm.weight":
                        w.mlp_norm_w[layer_idx] = ptr
                    elif suffix == "mlp.gate_proj.weight":
                        w.mlp_gate_w[layer_idx] = ptr
                    elif suffix == "mlp.up_proj.weight":
                        w.mlp_up_w[layer_idx] = ptr
                    elif suffix == "mlp.down_proj.weight":
                        w.mlp_down_w[layer_idx] = ptr

        if self.meta.tie_embeddings and not has_lm_head:
            print(f"  Using tied embeddings (no separate lm_head.weight)")

        print(f"[LLaMA3] Weights loaded successfully")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = 0,
    ) -> list:
        current_ids = list(inputs)
        InArrayType = ctypes.c_int64 * 1
        use_sampling = not (top_k == 1)

        # Prefill
        for i in range(len(current_ids) - 1):
            in_ptr = InArrayType(current_ids[i])
            LIB_LLAISYS.llaisysLlama3ModelInfer(
                self.handle, in_ptr, ctypes.c_size_t(1)
            )

        # First prediction from last input token
        last_input = current_ids[-1]
        in_ptr = InArrayType(last_input)
        if use_sampling:
            next_token = LIB_LLAISYS.llaisysLlama3ModelInferEx(
                self.handle, in_ptr, ctypes.c_size_t(1),
                ctypes.c_float(temperature), ctypes.c_int(top_k),
                ctypes.c_float(top_p), ctypes.c_uint64(seed),
            )
        else:
            next_token = LIB_LLAISYS.llaisysLlama3ModelInfer(
                self.handle, in_ptr, ctypes.c_size_t(1)
            )
        current_ids.append(next_token)

        # Decoding
        for _ in range(max_new_tokens - 1):
            in_ptr = InArrayType(next_token)
            if use_sampling:
                next_token = LIB_LLAISYS.llaisysLlama3ModelInferEx(
                    self.handle, in_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k),
                    ctypes.c_float(top_p), ctypes.c_uint64(seed),
                )
            else:
                next_token = LIB_LLAISYS.llaisysLlama3ModelInfer(
                    self.handle, in_ptr, ctypes.c_size_t(1)
                )
            current_ids.append(next_token)
            if next_token == self.meta.end_token:
                break

        return current_ids

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 0.8,
        seed: int = 0,
    ):
        InArrayType = ctypes.c_int64 * 1
        use_sampling = not (top_k == 1)

        for i in range(len(inputs) - 1):
            in_ptr = InArrayType(inputs[i])
            LIB_LLAISYS.llaisysLlama3ModelInfer(
                self.handle, in_ptr, ctypes.c_size_t(1)
            )

        in_ptr = InArrayType(inputs[-1])
        if use_sampling:
            next_token = LIB_LLAISYS.llaisysLlama3ModelInferEx(
                self.handle, in_ptr, ctypes.c_size_t(1),
                ctypes.c_float(temperature), ctypes.c_int(top_k),
                ctypes.c_float(top_p), ctypes.c_uint64(seed),
            )
        else:
            next_token = LIB_LLAISYS.llaisysLlama3ModelInfer(
                self.handle, in_ptr, ctypes.c_size_t(1)
            )
        yield next_token

        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                return
            in_ptr = InArrayType(next_token)
            if use_sampling:
                next_token = LIB_LLAISYS.llaisysLlama3ModelInferEx(
                    self.handle, in_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k),
                    ctypes.c_float(top_p), ctypes.c_uint64(seed),
                )
            else:
                next_token = LIB_LLAISYS.llaisysLlama3ModelInfer(
                    self.handle, in_ptr, ctypes.c_size_t(1)
                )
            yield next_token

    def reset(self):
        LIB_LLAISYS.llaisysLlama3ModelReset(self.handle)
