"""
DeepSeek-V2 ctypes 绑定 — 定义 C 结构体和函数签名

权重结构说明:
  MLA 的 q_proj / kv_b_proj 在 Python 加载时拆分为独立子投影，
  所以 C 结构体中有 q_nope_proj / q_pe_proj / kv_b_nope / kv_b_value 等字段。
  MoE experts 用扁平化一维数组 [nlayer * n_routed_experts]。
"""
import ctypes
from ctypes import POINTER, c_void_p, c_size_t, c_int64, c_int
from .llaisys_types import llaisysDeviceType_t

TensorHandle = c_void_p


class LlaisysDeepSeekV2Meta(ctypes.Structure):
    """对应 C 结构体 LlaisysDeepSeekV2Meta"""
    _fields_ = [
        ("dtype", c_int),
        ("num_hidden_layers", c_size_t),
        ("hidden_size", c_size_t),
        ("num_attention_heads", c_size_t),
        ("max_position_embeddings", c_size_t),
        ("vocab_size", c_size_t),
        ("rms_norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("eos_token_id", c_int64),

        # MLA 参数
        ("qk_nope_head_dim", c_size_t),
        ("qk_rope_head_dim", c_size_t),
        ("kv_lora_rank", c_size_t),
        ("v_head_dim", c_size_t),
        ("q_lora_rank", c_size_t),
        ("softmax_scale", ctypes.c_float),

        # MoE 参数
        ("intermediate_size", c_size_t),
        ("moe_intermediate_size", c_size_t),
        ("n_routed_experts", c_size_t),
        ("n_shared_experts", c_size_t),
        ("num_experts_per_tok", c_size_t),
        ("first_k_dense_replace", c_size_t),
        ("routed_scaling_factor", ctypes.c_float),
    ]


class LlaisysDeepSeekV2Weights(ctypes.Structure):
    """对应 C 结构体 LlaisysDeepSeekV2Weights"""
    _fields_ = [
        # 全局权重
        ("in_embed", TensorHandle),
        ("out_embed", TensorHandle),
        ("out_norm_w", TensorHandle),

        # Per-layer: LayerNorm
        ("attn_norm_w", POINTER(TensorHandle)),
        ("post_attn_norm_w", POINTER(TensorHandle)),

        # Per-layer: MLA Attention (Python 已拆分)
        ("attn_q_nope_proj_w", POINTER(TensorHandle)),
        ("attn_q_pe_proj_w", POINTER(TensorHandle)),
        ("attn_kv_compressed_w", POINTER(TensorHandle)),
        ("attn_kv_rope_w", POINTER(TensorHandle)),
        ("attn_kv_a_norm_w", POINTER(TensorHandle)),
        ("attn_kv_b_nope_w", POINTER(TensorHandle)),
        ("attn_kv_b_value_w", POINTER(TensorHandle)),
        ("attn_o_proj_w", POINTER(TensorHandle)),

        # Per-layer: Dense MLP
        ("mlp_gate_w", POINTER(TensorHandle)),
        ("mlp_up_w", POINTER(TensorHandle)),
        ("mlp_down_w", POINTER(TensorHandle)),

        # Per-layer: MoE Gate
        ("moe_gate_w", POINTER(TensorHandle)),

        # MoE Experts: 扁平化 [nlayer * n_routed_experts]
        ("expert_gate_w", POINTER(TensorHandle)),
        ("expert_up_w", POINTER(TensorHandle)),
        ("expert_down_w", POINTER(TensorHandle)),

        # Per-layer: Shared Experts
        ("shared_gate_w", POINTER(TensorHandle)),
        ("shared_up_w", POINTER(TensorHandle)),
        ("shared_down_w", POINTER(TensorHandle)),
    ]


def load_deepseek_v2(lib):
    """注册 DeepSeek-V2 的 C API 函数签名"""

    lib.llaisysDeepSeekV2ModelCreate.argtypes = [
        POINTER(LlaisysDeepSeekV2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysDeepSeekV2ModelCreate.restype = c_void_p

    lib.llaisysDeepSeekV2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysDeepSeekV2ModelDestroy.restype = None

    lib.llaisysDeepSeekV2ModelWeights.argtypes = [c_void_p]
    lib.llaisysDeepSeekV2ModelWeights.restype = POINTER(LlaisysDeepSeekV2Weights)

    lib.llaisysDeepSeekV2ModelInfer.argtypes = [
        c_void_p, POINTER(c_int64), c_size_t,
    ]
    lib.llaisysDeepSeekV2ModelInfer.restype = c_int64

    lib.llaisysDeepSeekV2ModelInferEx.argtypes = [
        c_void_p, POINTER(c_int64), c_size_t,
        ctypes.c_float, c_int, ctypes.c_float, ctypes.c_uint64,
    ]
    lib.llaisysDeepSeekV2ModelInferEx.restype = c_int64

    lib.llaisysDeepSeekV2ModelReset.argtypes = [c_void_p]
    lib.llaisysDeepSeekV2ModelReset.restype = None
