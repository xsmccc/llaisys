from typing import Sequence
from pathlib import Path
import ctypes
import json
import numpy as np
import torch
from safetensors.torch import load_file

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, TensorHandle

try:
    from ..tensor import Tensor
except ImportError:
    Tensor = None


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        """Initialize Qwen2 model and load weights from safetensors"""
        self.model_path = Path(model_path)
        self.device = device
        self._kept_references = []

        # 1. Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # 2. Create meta structure
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = 0  # F32
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]
        self.meta.di = config["intermediate_size"]
        self.meta.maxseq = config["max_position_embeddings"]
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]
        self.meta.theta = config["rope_theta"]

        eos_id = config.get("eos_token_id", 151643)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.meta.end_token = eos_id
        self.meta.dh = self.meta.hs // self.meta.nh

        # 3. Create C++ model
        print(f"[Qwen2] Initializing C++ Backend...")
        print(f"        Layers: {self.meta.nlayer}, Hidden: {self.meta.hs}, Heads: {self.meta.nh}")

        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), device.value, None, 0
        )

        if not self.handle:
            raise RuntimeError("Failed to create C++ model!")

        # 4. Get weights structure
        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        # 5. Load weights from safetensors
        print(f"[Qwen2] Loading weights from {model_path}...")
        self._load_safetensors(self.model_path)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def _load_safetensors(self, path: Path):
        """Load weights from safetensors files"""
        files = sorted(path.glob("*.safetensors"))
        if not files:
            print(f"Warning: No safetensors found in {path}")
            return

        for file_path in files:
            print(f"  Loading {file_path.name}...")
            weights_dict = load_file(str(file_path))
            for name, tensor in weights_dict.items():
                self._map_weight(name, tensor)

    def _map_weight(self, name: str, tensor: torch.Tensor):
        """Convert torch tensor to Tensor and map to model"""
        # 1. Convert dtype
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor = tensor.float()

        array = tensor.numpy()

        # 2. Fix shape for norms (must be 1D)
        is_norm = "norm" in name and "weight" in name
        if is_norm and array.ndim > 1:
            array = array.reshape(-1)

        # 3. Ensure C-contiguous
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        if Tensor is None:
            return

        try:
            # Create tensor
            llaisys_tensor = Tensor(
                shape=array.shape,
                dtype=DataType.F32,
                device=DeviceType.CPU
            )
            data_ptr = array.ctypes.data_as(ctypes.c_void_p)
            llaisys_tensor.load(data_ptr)
        except Exception as e:
            print(f"Error creating tensor for {name}: {e}")
            return

        self._kept_references.append(llaisys_tensor)
        ptr = llaisys_tensor.lib_tensor()

        # 4. Map to weight structure
        w = self.c_weights

        if name == "model.embed_tokens.weight":
            w.in_embed = ptr
        elif name == "model.norm.weight":
            w.out_norm_w = ptr
        elif name == "lm_head.weight":
            w.out_embed = ptr
        elif name.startswith("model.layers."):
            parts = name.split('.')
            try:
                layer_idx = int(parts[2])
                suffix = ".".join(parts[3:])
            except:
                return

            if layer_idx >= self.meta.nlayer:
                return

            # Attention weights
            if suffix == "self_attn.q_proj.weight":
                w.attn_q_w[layer_idx] = ptr
            elif suffix == "self_attn.k_proj.weight":
                w.attn_k_w[layer_idx] = ptr
            elif suffix == "self_attn.v_proj.weight":
                w.attn_v_w[layer_idx] = ptr
            elif suffix == "self_attn.o_proj.weight":
                w.attn_o_w[layer_idx] = ptr
            elif suffix == "self_attn.q_proj.bias":
                w.attn_q_b[layer_idx] = ptr
            elif suffix == "self_attn.k_proj.bias":
                w.attn_k_b[layer_idx] = ptr
            elif suffix == "self_attn.v_proj.bias":
                w.attn_v_b[layer_idx] = ptr
            elif suffix == "input_layernorm.weight":
                w.attn_norm_w[layer_idx] = ptr
            elif suffix == "post_attention_layernorm.weight":
                w.mlp_norm_w[layer_idx] = ptr
            # MLP weights
            elif suffix == "mlp.gate_proj.weight":
                w.mlp_gate_w[layer_idx] = ptr
            elif suffix == "mlp.up_proj.weight":
                w.mlp_up_w[layer_idx] = ptr
            elif suffix == "mlp.down_proj.weight":
                w.mlp_down_w[layer_idx] = ptr

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> list:
        """Generate tokens using the model"""
        current_ids = list(inputs)

        # Convert to ctypes array type
        InArrayType = ctypes.c_int64 * 1

        # Phase 1: Prefill - process all input tokens except the last
        for i in range(len(current_ids) - 1):
            token = current_ids[i]
            in_ptr = InArrayType(token)

            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle,
                in_ptr,
                ctypes.c_size_t(1)
            )

        # Phase 2: First prediction - process last input token
        last_input = current_ids[-1]
        in_ptr = InArrayType(last_input)

        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.handle,
            in_ptr,
            ctypes.c_size_t(1)
        )
        current_ids.append(next_token)

        # Phase 3: Decoding loop - generate remaining tokens
        for _ in range(max_new_tokens - 1):
            in_ptr = InArrayType(next_token)

            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle,
                in_ptr,
                ctypes.c_size_t(1)
            )

            current_ids.append(next_token)

            if next_token == self.meta.end_token:
                break

        return current_ids
