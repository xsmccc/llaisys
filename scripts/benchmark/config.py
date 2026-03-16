"""
硬件规格与测试配置
"""

# ============================================================
#  RTX 4060 Ti (AD106, sm_89) 硬件参数
# ============================================================
GPU_SPECS = {
    "name": "RTX 4060 Ti",
    "arch": "sm_89",
    "mem_bandwidth_GBs": 288.0,
    "fp32_tflops": 22.06,
    "fp16_tflops": 22.06,
    "fp16_tc_tflops": 176.5,
    "int8_tc_tops": 353.0,
    "fp8_tc_tflops": 353.0,
    "vram_GB": 8.0,
    "l2_cache_MB": 32.0,
    "sm_count": 34,
    "shared_mem_per_sm_KB": 100,
}

# ============================================================
#  Qwen2-1.5B 模型参数
# ============================================================
MODEL_PARAMS = {
    "name": "DeepSeek-R1-Distill-Qwen-1.5B",
    "nlayer": 28,
    "hidden_size": 1536,
    "num_heads": 12,
    "num_kv_heads": 12,
    "head_dim": 128,
    "intermediate_size": 8960,
    "vocab_size": 151936,
    "max_position_embeddings": 131072,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
}

# ============================================================
#  Benchmark 测试配置
# ============================================================
TIMING = {
    "warmup": 20,
    "repeat": 200,
}

OP_CONFIGS = {
    "add": {
        "configs": [
            {"shape": [512, 1536], "dtype": "f32", "label": "small"},
            {"shape": [512, 1536], "dtype": "f16", "label": "small_f16"},
            {"shape": [2048, 1536], "dtype": "f32", "label": "medium"},
            {"shape": [2048, 8960], "dtype": "f32", "label": "large"},
            {"shape": [2048, 8960], "dtype": "f16", "label": "large_f16"},
        ],
        "flops_fn": lambda shape, dtype: _numel(shape),
        "bytes_fn": lambda shape, dtype: 3 * _numel(shape) * _elem_size(dtype),
    },
    "rms_norm": {
        "configs": [
            {"shape": [512, 1536], "dtype": "f32", "label": "small"},
            {"shape": [512, 1536], "dtype": "f16", "label": "small_f16"},
            {"shape": [2048, 1536], "dtype": "f32", "label": "medium"},
            {"shape": [2048, 8960], "dtype": "f32", "label": "large"},
        ],
        "flops_fn": lambda shape, dtype: 5 * _numel(shape),
        "bytes_fn": lambda shape, dtype: (2 * _numel(shape) + shape[-1]) * _elem_size(dtype),
    },
    "swiglu": {
        "configs": [
            {"shape": [512, 8960], "dtype": "f32", "label": "small"},
            {"shape": [512, 8960], "dtype": "f16", "label": "small_f16"},
            {"shape": [2048, 8960], "dtype": "f32", "label": "large"},
        ],
        "flops_fn": lambda shape, dtype: 4 * _numel(shape),
        "bytes_fn": lambda shape, dtype: 3 * _numel(shape) * _elem_size(dtype),
    },
    "linear": {
        "configs": [
            {"x_shape": [1, 1536], "w_shape": [1536, 1536], "dtype": "f32", "label": "decode_hs"},
            {"x_shape": [1, 1536], "w_shape": [8960, 1536], "dtype": "f32", "label": "decode_mlp"},
            {"x_shape": [512, 1536], "w_shape": [1536, 1536], "dtype": "f32", "label": "prefill_hs"},
            {"x_shape": [512, 1536], "w_shape": [8960, 1536], "dtype": "f32", "label": "prefill_mlp"},
            {"x_shape": [512, 1536], "w_shape": [1536, 1536], "dtype": "f16", "label": "prefill_hs_f16"},
            {"x_shape": [512, 1536], "w_shape": [8960, 1536], "dtype": "f16", "label": "prefill_mlp_f16"},
        ],
        "flops_fn": lambda cfg, dtype: 2 * cfg["x_shape"][0] * cfg["w_shape"][0] * cfg["x_shape"][1],
        "bytes_fn": lambda cfg, dtype: (
            _numel(cfg["x_shape"]) + _numel(cfg["w_shape"]) + cfg["x_shape"][0] * cfg["w_shape"][0]
        ) * _elem_size(dtype),
    },
    "rope": {
        "configs": [
            {"shape": [512, 12, 128], "pos_range": [0, 512], "dtype": "f32", "label": "prefill"},
            {"shape": [1, 12, 128], "pos_range": [511, 512], "dtype": "f32", "label": "decode"},
            {"shape": [2048, 12, 128], "pos_range": [0, 2048], "dtype": "f32", "label": "long_prefill"},
        ],
        "flops_fn": lambda shape, dtype: 6 * _numel(shape),
        "bytes_fn": lambda shape, dtype: 2 * _numel(shape) * _elem_size(dtype),
    },
    "embedding": {
        "configs": [
            {"idx_shape": [1], "embd_shape": [151936, 1536], "dtype": "f32", "label": "single_token"},
            {"idx_shape": [512], "embd_shape": [151936, 1536], "dtype": "f32", "label": "prefill_512"},
        ],
        "flops_fn": lambda cfg, dtype: 0,
        "bytes_fn": lambda cfg, dtype: (cfg["idx_shape"][0] * cfg["embd_shape"][1]) * _elem_size(dtype) * 2,
    },
    "argmax": {
        "configs": [
            {"shape": [151936], "dtype": "f32", "label": "vocab"},
        ],
        "flops_fn": lambda shape, dtype: _numel(shape),
        "bytes_fn": lambda shape, dtype: _numel(shape) * _elem_size(dtype),
    },
    "self_attention": {
        "configs": [
            {"params": [1, 128, 12, 12, 128], "dtype": "f32", "label": "decode_128"},
            {"params": [1, 256, 12, 12, 128], "dtype": "f32", "label": "decode_256"},
            {"params": [1, 512, 12, 12, 128], "dtype": "f32", "label": "decode_512"},
            {"params": [128, 128, 12, 12, 128], "dtype": "f32", "label": "prefill_128"},
            {"params": [256, 256, 12, 12, 128], "dtype": "f32", "label": "prefill_256"},
            {"params": [512, 512, 12, 12, 128], "dtype": "f32", "label": "prefill_512"},
            {"params": [1, 512, 12, 12, 128], "dtype": "f16", "label": "decode_512_f16"},
            {"params": [512, 512, 12, 12, 128], "dtype": "f16", "label": "prefill_512_f16"},
        ],
        "flops_fn": lambda params, dtype: (
            2 * params[0] * params[1] * params[4] * params[2]
            + 2 * params[0] * params[1] * params[4] * params[2]
        ),
        "bytes_fn": lambda params, dtype: (
            (params[0] * params[2] * params[4]
             + params[1] * params[3] * params[4]
             + params[1] * params[3] * params[4]
             + params[0] * params[2] * params[4])
            * _elem_size(dtype)
        ),
    },
}

INFERENCE_CONFIGS = [
    {"prompt": "Hello", "max_tokens": 32, "label": "short_gen"},
    {"prompt": "Explain quantum computing in simple terms.", "max_tokens": 64, "label": "medium_gen"},
    {"prompt": "Write a Python function to sort a list.", "max_tokens": 128, "label": "long_gen"},
]

# ============================================================
#  辅助函数
# ============================================================

def _numel(shape):
    result = 1
    for s in shape:
        result *= s
    return result

def _elem_size(dtype):
    return {"f32": 4, "f16": 2, "bf16": 2, "i64": 8, "i8": 1}.get(dtype, 4)

def calc_tflops(flops, time_s):
    if time_s <= 0:
        return 0.0
    return flops / time_s / 1e12

def calc_bandwidth_GBs(bytes_transferred, time_s):
    if time_s <= 0:
        return 0.0
    return bytes_transferred / time_s / 1e9

def arithmetic_intensity(flops, bytes_transferred):
    if bytes_transferred <= 0:
        return 0.0
    return flops / bytes_transferred

def roofline_peak_tflops(ai, dtype="f32"):
    bw = GPU_SPECS["mem_bandwidth_GBs"]
    peak = GPU_SPECS["fp32_tflops"] if dtype == "f32" else GPU_SPECS["fp16_tflops"]
    return min(peak, bw * ai / 1e3)
