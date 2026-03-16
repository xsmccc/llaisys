import ctypes
from ctypes import POINTER, c_void_p, c_size_t, c_int64, c_int

TensorHandle = c_void_p


class LlaisysLlama3Meta(ctypes.Structure):
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
        ("end_token", c_int64),
        ("tie_embeddings", c_int),
    ]


class LlaisysLlama3Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", TensorHandle),
        ("out_embed", TensorHandle),
        ("out_norm_w", TensorHandle),

        ("attn_norm_w", POINTER(TensorHandle)),
        ("attn_q_w", POINTER(TensorHandle)),
        ("attn_k_w", POINTER(TensorHandle)),
        ("attn_v_w", POINTER(TensorHandle)),
        ("attn_o_w", POINTER(TensorHandle)),

        ("mlp_norm_w", POINTER(TensorHandle)),
        ("mlp_gate_w", POINTER(TensorHandle)),
        ("mlp_up_w", POINTER(TensorHandle)),
        ("mlp_down_w", POINTER(TensorHandle)),
    ]


def load_llama3(lib):
    """Register LLaMA3 C function signatures"""
    lib.llaisysLlama3ModelCreate.argtypes = [
        POINTER(LlaisysLlama3Meta),
        c_int,  # device type
        POINTER(c_int),
        c_int
    ]
    lib.llaisysLlama3ModelCreate.restype = c_void_p

    lib.llaisysLlama3ModelDestroy.argtypes = [c_void_p]
    lib.llaisysLlama3ModelDestroy.restype = None

    lib.llaisysLlama3ModelWeights.argtypes = [c_void_p]
    lib.llaisysLlama3ModelWeights.restype = POINTER(LlaisysLlama3Weights)

    lib.llaisysLlama3ModelInfer.argtypes = [
        c_void_p, POINTER(c_int64), c_size_t
    ]
    lib.llaisysLlama3ModelInfer.restype = c_int64

    lib.llaisysLlama3ModelInferEx.argtypes = [
        c_void_p, POINTER(c_int64), c_size_t,
        ctypes.c_float, c_int, ctypes.c_float, ctypes.c_uint64
    ]
    lib.llaisysLlama3ModelInferEx.restype = c_int64

    lib.llaisysLlama3ModelReset.argtypes = [c_void_p]
    lib.llaisysLlama3ModelReset.restype = None
