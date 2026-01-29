import ctypes #导入Ctypes，用于与C共享库交互，提供C数据类型的python版本
'''
POINTER      指针类型 (C中的 *)
c_void_p     void* (无类型指针)
c_size_t     size_t (无符号整数，通常用于大小)
c_int64      int64_t (64位有符号整数)
c_int        int (32位有符号整数)
'''
from ctypes import POINTER, c_void_p, c_size_t, c_int64, c_int
from .llaisys_types import llaisysDeviceType_t

# Qwen2 structures
# python类 代表C中的结构体
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", c_int),              # 数据类型 (0=F32)
        ("nlayer", c_size_t),          # 层数 (28)
        ("hs", c_size_t),              # 隐层大小 (2048)
        ("nh", c_size_t),              # 注意力头数 (12)
        ("nkvh", c_size_t),            # KV头数 (通常等于nh)
        ("dh", c_size_t),              # 每个头的维度 (hs/nh)
        ("di", c_size_t),              # MLP中间层大小
        ("maxseq", c_size_t),          # 最大序列长度
        ("voc", c_size_t),             # 词表大小
        ("epsilon", ctypes.c_float),   # RMSNorm的epsilon
        ("theta", ctypes.c_float),     # RoPE的theta参数
        ("end_token", c_int64)         # EOS令牌ID (151643)
    ]

# TensorHandle 定义 即void*
TensorHandle = c_void_p

# LlaisysQwen2Weights 结构体（权重结构体）
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        # ===== 嵌入层 =====
        ("in_embed", TensorHandle),          # 输入嵌入权重 (vocab_size, hidden_size)
        ("out_embed", TensorHandle),         # 输出嵌入权重 (hidden_size, vocab_size)
        ("out_norm_w", TensorHandle),        # 输出前的LayerNorm权重
        
        # ===== 注意力层权重 (数组，28个层) 指针的指针 =====
        ("attn_norm_w", POINTER(TensorHandle)),   # 每层的输入norm权重 [28]
        ("attn_q_w", POINTER(TensorHandle)),      # Q投影权重 [28]
        ("attn_q_b", POINTER(TensorHandle)),      # Q投影偏置 [28]
        ("attn_k_w", POINTER(TensorHandle)),      # K投影权重 [28]
        ("attn_k_b", POINTER(TensorHandle)),      # K投影偏置 [28]
        ("attn_v_w", POINTER(TensorHandle)),      # V投影权重 [28]
        ("attn_v_b", POINTER(TensorHandle)),      # V投影偏置 [28]
        ("attn_o_w", POINTER(TensorHandle)),      # 输出投影权重 [28]
        
        # ===== MLP层权重 (数组，28个层) =====
        ("mlp_norm_w", POINTER(TensorHandle)),    # MLP前的norm权重 [28]
        ("mlp_gate_w", POINTER(TensorHandle)),    # Gate投影权重 [28]
        ("mlp_up_w", POINTER(TensorHandle)),      # Up投影权重 [28]
        ("mlp_down_w", POINTER(TensorHandle)),    # Down投影权重 [28]
    ]

#  函数签名注册
"""
    注册Qwen2的所有C API函数
    告诉ctypes：
    1. 每个函数有什么参数？
    2. 每个参数是什么类型？
    3. 函数返回什么类型？
    """
def load_qwen2(lib):

    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p # 返回值：void*指针

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [c_void_p] #入参类型
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights) #返回值类型

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,
        POINTER(c_int64),
        c_size_t
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
