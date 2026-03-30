

from typing import Sequence
from pathlib import Path
import ctypes
import json
import numpy as np
import torch    # PyTorch
from safetensors.torch import load_file

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType

from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, TensorHandle

try:
    from ..tensor import Tensor
except ImportError:
    Tensor = None


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU,
                 max_seq_len: int = 8192, quantized: bool = False, int4: bool = False):
        """Initialize Qwen2 model and load weights from safetensors or quantized npz

        Args:
            model_path: 模型文件夹路径
            device: 计算设备 (CPU / NVIDIA)
            max_seq_len: KV cache 最大序列长度（默认 4096）。
                         直接影响 GPU 显存占用：每层 KV cache = 2 × max_seq_len × nkvh × dh × 4B。
                         8GB GPU 约 224MB (4096), A100 可设 32768+。配合 FlashAttention 支持更长序列。
            quantized: 是否加载 INT8 量化权重 (quantized_weights.npz)
        """
        self.model_path = Path(model_path)
        self.device = device
        self._kept_references = []

        # Auto-detect quantization from model files
        if not quantized and not int4:
            if (Path(model_path) / 'quantized_weights_int4.npz').exists():
                int4 = True
            elif (Path(model_path) / 'quantized_weights.npz').exists():
                quantized = True
        self.quantized = quantized
        self.int4 = int4

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.meta = LlaisysQwen2Meta() # C结构体，传给C++后端
        self.meta.dtype = 0  # F32
        self.meta.nlayer = config["num_hidden_layers"]
        self.meta.hs = config["hidden_size"]
        self.meta.nh = config["num_attention_heads"]
        self.meta.nkvh = config["num_key_value_heads"]  # KV头数 (12)
        self.meta.di = config["intermediate_size"]      # MLP中间层大小
        # KV cache 大小限制：使用 min(max_seq_len, 模型最大值) 以避免 GPU 显存溢出
        # 原始模型 max_position_embeddings=131072 会导致 28层 KV cache 占用 ~7GB
        model_max_seq = config["max_position_embeddings"]
        actual_max_seq = min(max_seq_len, model_max_seq)
        if actual_max_seq < model_max_seq:
            print(f"[Qwen2] KV cache limited to {actual_max_seq} tokens "
                  f"(model supports {model_max_seq})")
        self.meta.maxseq = actual_max_seq
        self.meta.voc = config["vocab_size"]
        self.meta.epsilon = config["rms_norm_eps"]      # RMSNorm epsilon
        self.meta.theta = config["rope_theta"]          # RoPE theta

        eos_id = config.get("eos_token_id", 151643)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        self.meta.end_token = eos_id

        self.meta.dh = self.meta.hs // self.meta.nh

        print(f"[Qwen2] Initializing C++ Backend...")
        print(f"        Layers: {self.meta.nlayer}, Hidden: {self.meta.hs}, Heads: {self.meta.nh}")
        if self.int4:
            print(f"        Mode: INT4 Quantized (W4A16, group_size=128)")
        elif self.quantized:
            print(f"        Mode: INT8 Quantized (W8A16)")

        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            device.value,
            None,
            0
        )

        if not self.handle:
            raise RuntimeError("Failed to create C++ model!")

        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        print(f"[Qwen2] Loading weights from {model_path}...")
        if self.int4:
            self._load_quantized_int4(self.model_path)
        elif self.quantized:
            self._load_quantized(self.model_path)
        else:
            self._load_safetensors(self.model_path)

    def __del__(self):
        """销毁C++模型，释放资源"""
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def _load_safetensors(self, path: Path):
        """Load weights from safetensors files"""
        # 找到所有.safetensors文件 如果模型很大，可能分成多个文件 (model-00001.safetensors等)
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
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor = tensor.float()

        array = tensor.numpy()

        is_norm = "norm" in name and "weight" in name
        if is_norm and array.ndim > 1:
            array = array.reshape(-1)

        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        if Tensor is None:
            return

        try:
            llaisys_tensor = Tensor(
                shape=array.shape,
                dtype=DataType.F32,
                device=self.device
            )
            data_ptr = array.ctypes.data_as(ctypes.c_void_p)
            llaisys_tensor.load(data_ptr)
        except Exception as e:
            print(f"Error creating tensor for {name}: {e}")
            return

        self._kept_references.append(llaisys_tensor)

        ptr = llaisys_tensor.lib_tensor()

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

            if suffix == "self_attn.q_proj.weight":
                w.attn_q_w[layer_idx] = ptr   # Query投影
            elif suffix == "self_attn.k_proj.weight":
                w.attn_k_w[layer_idx] = ptr   # Key投影
            elif suffix == "self_attn.v_proj.weight":
                w.attn_v_w[layer_idx] = ptr   # Value投影
            elif suffix == "self_attn.o_proj.weight":
                w.attn_o_w[layer_idx] = ptr   # Output投影

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
            elif suffix == "mlp.gate_proj.weight":
                w.mlp_gate_w[layer_idx] = ptr   # Gate投影
            elif suffix == "mlp.up_proj.weight":
                w.mlp_up_w[layer_idx] = ptr # Up投影
            elif suffix == "mlp.down_proj.weight":
                w.mlp_down_w[layer_idx] = ptr   # Down投影

    def _create_tensor_from_numpy(self, array: np.ndarray):
        """从 numpy 数组创建 LLAISYS Tensor 并返回 C handle"""
        if Tensor is None:
            return None

        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        # 根据 dtype 选择 LLAISYS DataType
        if array.dtype == np.int8:
            dtype = DataType.I8
        elif array.dtype == np.float32:
            dtype = DataType.F32
        elif array.dtype == np.uint8:
            dtype = DataType.U8
        elif array.dtype == np.float16:
            dtype = DataType.F16
        else:
            raise ValueError(f"Unsupported numpy dtype: {array.dtype}")

        llaisys_tensor = Tensor(
            shape=array.shape,
            dtype=dtype,
            device=self.device
        )
        data_ptr = array.ctypes.data_as(ctypes.c_void_p)
        llaisys_tensor.load(data_ptr)
        self._kept_references.append(llaisys_tensor)
        return llaisys_tensor.lib_tensor()

    def _load_quantized(self, path: Path):
        """加载 INT8 量化权重 (quantized_weights.npz)"""
        npz_path = path / "quantized_weights.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Quantized weights not found at {npz_path}")

        config_path = path / "quantize_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                quant_config = json.load(f)
            quantized_names = set(quant_config.get("quantized_weights", []))
            print(f"  Quantized weights: {len(quantized_names)}")
        else:
            quantized_names = set()
            print("  Warning: quantize_config.json not found, detecting from dtypes")

        print(f"  Loading {npz_path.name}...")
        data = np.load(str(npz_path))

        w = self.c_weights
        w.quantized = 1

        for key in data.files:
            arr = data[key]

            # scales 条目: "{original_name}.scales"
            if key.endswith(".scales"):
                original_name = key[:-len(".scales")]
                ptr = self._create_tensor_from_numpy(arr.astype(np.float32))
                if ptr is None:
                    continue
                self._map_scale(original_name, ptr)
                continue

            name = key
            is_quantized_weight = (name in quantized_names) or (arr.dtype == np.int8)

            if is_quantized_weight:
                # INT8 量化权重
                ptr = self._create_tensor_from_numpy(arr.astype(np.int8))
            else:
                # FP16 权重 (embedding, norm, bias 等)
                ptr = self._create_tensor_from_numpy(arr.astype(np.float16))

            if ptr is None:
                continue

            self._map_weight_ptr(name, ptr)

        print(f"  Quantized weights loaded successfully (W8A16 mode)")

    def _map_weight_ptr(self, name: str, ptr):
        """映射权重 handle 到 C 结构体 (不做类型转换, 直接用 ptr)"""
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
            except Exception:
                return
            if layer_idx >= self.meta.nlayer:
                return

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
            elif suffix == "mlp.gate_proj.weight":
                w.mlp_gate_w[layer_idx] = ptr
            elif suffix == "mlp.up_proj.weight":
                w.mlp_up_w[layer_idx] = ptr
            elif suffix == "mlp.down_proj.weight":
                w.mlp_down_w[layer_idx] = ptr

    def _map_scale(self, weight_name: str, ptr):
        """映射 scales handle 到 C 结构体"""
        w = self.c_weights

        if weight_name == "lm_head.weight":
            w.out_embed_scales = ptr
        elif weight_name.startswith("model.layers."):
            parts = weight_name.split('.')
            try:
                layer_idx = int(parts[2])
                suffix = ".".join(parts[3:])
            except Exception:
                return
            if layer_idx >= self.meta.nlayer:
                return

            if suffix == "self_attn.q_proj.weight":
                w.attn_q_w_scales[layer_idx] = ptr
            elif suffix == "self_attn.k_proj.weight":
                w.attn_k_w_scales[layer_idx] = ptr
            elif suffix == "self_attn.v_proj.weight":
                w.attn_v_w_scales[layer_idx] = ptr
            elif suffix == "self_attn.o_proj.weight":
                w.attn_o_w_scales[layer_idx] = ptr
            elif suffix == "mlp.gate_proj.weight":
                w.mlp_gate_w_scales[layer_idx] = ptr
            elif suffix == "mlp.up_proj.weight":
                w.mlp_up_w_scales[layer_idx] = ptr
            elif suffix == "mlp.down_proj.weight":
                w.mlp_down_w_scales[layer_idx] = ptr

    def _load_quantized_int4(self, path: Path):
        """加载 INT4 量化权重 (quantized_weights_int4.npz)"""
        npz_path = path / "quantized_weights_int4.npz"
        if not npz_path.exists():
            # 尝试 INT4 子目录
            int4_dir = path.parent / (path.name + "-INT4")
            npz_path = int4_dir / "quantized_weights_int4.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"INT4 weights not found at {npz_path}")
            path = int4_dir

        config_path = path / "quantize_config.json"
        group_size = 128
        if config_path.exists():
            with open(config_path, "r") as f:
                quant_config = json.load(f)
            group_size = quant_config.get("group_size", 128)
            quantized_names = set(quant_config.get("quantized_weights", []))
            print(f"  INT4 quantized weights: {len(quantized_names)}, group_size={group_size}")
        else:
            quantized_names = set()

        print(f"  Loading {npz_path.name}...")
        data = np.load(str(npz_path))

        w = self.c_weights
        w.quantized = 2  # INT4 mode
        w.int4_group_size = group_size

        # 填充 int4_K_orig 数组
        nlayer = self.meta.nlayer
        # weight name → offset mapping (层内 q=0,k=1,v=2,o=3, gate=4,up=5,down=6)
        suffix_to_offset = {
            "self_attn.q_proj.weight": 0,
            "self_attn.k_proj.weight": 1,
            "self_attn.v_proj.weight": 2,
            "self_attn.o_proj.weight": 3,
            "mlp.gate_proj.weight": 4,
            "mlp.up_proj.weight": 5,
            "mlp.down_proj.weight": 6,
        }

        for key in data.files:
            arr = data[key]

            # .meta 条目: 提取 K_orig 并存入 int4_K_orig
            if key.endswith(".meta"):
                weight_name = key[:-len(".meta")]
                meta_vals = arr  # [N_orig, K_orig, K_padded, group_size]
                K_orig = int(meta_vals[1])

                if weight_name == "lm_head.weight":
                    w.int4_K_orig[nlayer * 7] = K_orig
                elif weight_name.startswith("model.layers."):
                    parts = weight_name.split(".")
                    layer_idx = int(parts[2])
                    suffix = ".".join(parts[3:])
                    if suffix in suffix_to_offset and layer_idx < nlayer:
                        w.int4_K_orig[layer_idx * 7 + suffix_to_offset[suffix]] = K_orig
                continue

            # .scales 条目: FP16 group scales
            if key.endswith(".scales"):
                original_name = key[:-len(".scales")]
                ptr = self._create_tensor_from_numpy(arr.astype(np.float16))
                if ptr is None:
                    continue
                self._map_scale(original_name, ptr)
                continue

            name = key
            is_quantized_weight = (name in quantized_names) or (arr.dtype == np.uint8)

            if is_quantized_weight:
                # INT4 packed权重 (U8)
                ptr = self._create_tensor_from_numpy(arr.astype(np.uint8))
            else:
                # FP16 权重 (embedding, norm, bias 等)
                ptr = self._create_tensor_from_numpy(arr.astype(np.float16))

            if ptr is None:
                continue
            self._map_weight_ptr(name, ptr)

        print(f"  INT4 quantized weights loaded successfully (W4A16 mode, group_size={group_size})")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1, # top-k采样 (1=贪心)
        top_p: float = 0.8,  # top-p采样
        temperature: float = 0.8,
        seed: int = 0,
    ) -> list:
        """
        生成tokens（Batch Prefill 优化版）

        LLM的推理分为两个阶段：

        1. Prefill (预填充): 一次性 batch 处理所有输入 tokens
           - 目的：单次 forward pass 计算所有 KV 缓存 + 预测第一个 token

        2. Decoding (解码): 逐个生成新tokens
           - 目的：根据KV缓存，每次生成一个token

        支持 Temperature / Top-K / Top-P 随机采样。
        当 top_k=1 时退化为 argmax 贪心解码。
        """
        current_ids = list(inputs)
        use_sampling = not (top_k == 1)

        # === Batch Prefill: 一次性发送所有 prompt tokens ===
        n = len(current_ids)
        PrefillArrayType = ctypes.c_int64 * n
        all_tokens = PrefillArrayType(*current_ids)

        if use_sampling:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInferEx(
                self.handle, all_tokens, ctypes.c_size_t(n),
                ctypes.c_float(temperature),
                ctypes.c_int(top_k),
                ctypes.c_float(top_p),
                ctypes.c_uint64(seed),
            )
        else:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle, all_tokens, ctypes.c_size_t(n)
            )
        current_ids.append(next_token)

        # === Decoding: 逐个生成剩余 tokens ===
        InArrayType = ctypes.c_int64 * 1
        for _ in range(max_new_tokens - 1):
            in_ptr = InArrayType(next_token)

            if use_sampling:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInferEx(
                    self.handle, in_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature),
                    ctypes.c_int(top_k),
                    ctypes.c_float(top_p),
                    ctypes.c_uint64(seed),
                )
            else:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
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
        """
        流式生成tokens（Python生成器，Batch Prefill 优化版）。
        一次性 batch 处理所有 prompt tokens，然后逐个 yield 解码 token。
        """
        use_sampling = not (top_k == 1)

        # === Batch Prefill ===
        n = len(inputs)
        PrefillArrayType = ctypes.c_int64 * n
        all_tokens = PrefillArrayType(*inputs)

        if use_sampling:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInferEx(
                self.handle, all_tokens, ctypes.c_size_t(n),
                ctypes.c_float(temperature), ctypes.c_int(top_k),
                ctypes.c_float(top_p), ctypes.c_uint64(seed),
            )
        else:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.handle, all_tokens, ctypes.c_size_t(n)
            )
        yield next_token

        # === Decode loop ===
        InArrayType = ctypes.c_int64 * 1
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                return
            in_ptr = InArrayType(next_token)
            if use_sampling:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInferEx(
                    self.handle, in_ptr, ctypes.c_size_t(1),
                    ctypes.c_float(temperature), ctypes.c_int(top_k),
                    ctypes.c_float(top_p), ctypes.c_uint64(seed),
                )
            else:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.handle, in_ptr, ctypes.c_size_t(1)
                )
            yield next_token

    def reset(self):
        """重置模型状态（清空 KV cache），用于新一轮对话"""
        LIB_LLAISYS.llaisysQwen2ModelReset(self.handle)
