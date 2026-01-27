import ctypes
from ctypes import POINTER, c_void_p, c_size_t, c_int64, c_int
from .llaisys_types import llaisysDeviceType_t

# Qwen2 structures
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", c_int64)
    ]


TensorHandle = c_void_p


class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", TensorHandle),
        ("out_embed", TensorHandle),
        ("out_norm_w", TensorHandle),
        ("attn_norm_w", POINTER(TensorHandle)),
        ("attn_q_w", POINTER(TensorHandle)),
        ("attn_q_b", POINTER(TensorHandle)),
        ("attn_k_w", POINTER(TensorHandle)),
        ("attn_k_b", POINTER(TensorHandle)),
        ("attn_v_w", POINTER(TensorHandle)),
        ("attn_v_b", POINTER(TensorHandle)),
        ("attn_o_w", POINTER(TensorHandle)),
        ("mlp_norm_w", POINTER(TensorHandle)),
        ("mlp_gate_w", POINTER(TensorHandle)),
        ("mlp_up_w", POINTER(TensorHandle)),
        ("mlp_down_w", POINTER(TensorHandle)),
    ]


def load_qwen2(lib):
    """Register Qwen2 C API function signatures"""
    
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,
        POINTER(c_int64),
        c_size_t
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
