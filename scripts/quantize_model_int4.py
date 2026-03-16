#!/usr/bin/env python3
"""
W4A32 离线权重量化工具 (INT4 Group Quantization)

将 FP32 safetensors 权重转换为 INT4 group-wise 对称量化格式。

量化方案 (Absmax Group-Wise):
  - 将 K 维度分为 num_groups = K / group_size 个组
  - 对每个 (n, g): scale[n, g] = max(|W[n, g*gs:(g+1)*gs]|) / 7.0
  - W_int4[n, k] = clamp(round(W[n, k] / scale[n, k//gs]), -8, 7)
  - 还原: W_fp32[n, k] ≈ W_int4[n, k] * scale[n, k // gs]

存储格式:
  - 两个 INT4 值打包为一个 uint8:
    packed[n, k//2] = (W_int4[n, k] & 0xF) | (W_int4[n, k+1] << 4)
  - Scales: FP16 [N, num_groups]

用法:
  python scripts/quantize_model_int4.py \
      --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \
      --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 \
      [--group-size 128]
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file


def quantize_int4_group(weight_fp32, group_size=128):
    """INT4 Group-Wise Absmax symmetric quantization."""
    assert weight_fp32.ndim == 2
    N, K = weight_fp32.shape

    # Pad K to multiple of group_size
    if K % group_size != 0:
        pad_k = group_size - (K % group_size)
        weight_fp32 = np.pad(weight_fp32, ((0, 0), (0, pad_k)), mode='constant')
        K = weight_fp32.shape[1]

    num_groups = K // group_size
    w_grouped = weight_fp32.reshape(N, num_groups, group_size)

    # Per-group absmax
    abs_max = np.max(np.abs(w_grouped), axis=2)  # [N, num_groups]
    abs_max = np.maximum(abs_max, 1e-10)
    scales = (abs_max / 7.0).astype(np.float16)  # [N, num_groups]

    # Quantize
    s_exp = scales.astype(np.float32)[:, :, np.newaxis]
    w_int4 = np.clip(np.round(w_grouped / s_exp), -8, 7).astype(np.int8)
    w_int4_flat = w_int4.reshape(N, K)

    # Pack: 2 x INT4 -> 1 x uint8
    assert K % 2 == 0
    low = w_int4_flat[:, 0::2].astype(np.uint8) & 0x0F
    high = (w_int4_flat[:, 1::2].astype(np.uint8) & 0x0F) << 4
    packed = (low | high).astype(np.uint8)  # [N, K//2]

    return packed, scales, K


def should_quantize(name, skip_lm_head=False):
    """判断权重是否应该被量化"""
    if "embed" in name and "weight" in name:
        if name == "lm_head.weight" and not skip_lm_head:
            return True
        return False
    if "norm" in name:
        return False
    if "bias" in name:
        return False
    if "weight" in name:
        return True
    return False


def compute_quantization_error(original, packed, scales, group_size):
    """计算 INT4 量化误差"""
    N, K_orig = original.shape
    K = packed.shape[1] * 2

    low = (packed & 0x0F).astype(np.int8)
    high = ((packed >> 4) & 0x0F).astype(np.int8)
    low = np.where(low >= 8, low - 16, low).astype(np.float32)
    high = np.where(high >= 8, high - 16, high).astype(np.float32)

    w_int4 = np.empty((N, K), dtype=np.float32)
    w_int4[:, 0::2] = low
    w_int4[:, 1::2] = high

    num_groups = K // group_size
    w_grouped = w_int4.reshape(N, num_groups, group_size)
    s = scales.astype(np.float32)[:, :, np.newaxis]
    dequant = (w_grouped * s).reshape(N, K)[:, :K_orig]

    diff = original - dequant
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    rel_err = np.sqrt(mse) / (np.std(original) + 1e-10)

    return {"mse": float(mse), "max_abs_error": float(max_err),
            "relative_error": float(rel_err)}


def main():
    parser = argparse.ArgumentParser(description="W4A32 INT4 Group Quantization")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gs = args.group_size

    print(f"=== INT4 Group Quantization (group_size={gs}) ===\n")

    # 1. Copy configs
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")

    # 2. Load weights
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        print(f"[ERROR] No safetensors in {model_path}")
        return

    all_weights = {}
    for f in files:
        print(f"  Loading {f.name}...")
        all_weights.update(load_file(str(f)))
    print(f"  Total: {len(all_weights)} tensors\n")

    # 3. Quantize
    save_dict = {}
    quantized_names = []
    skipped_names = []
    total_orig = 0
    total_quant = 0

    for name, tensor in sorted(all_weights.items()):
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor = tensor.float()
        arr = tensor.numpy()
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        orig_bytes = arr.nbytes
        total_orig += orig_bytes

        if should_quantize(name, args.skip_lm_head) and arr.ndim == 2:
            packed, scales, padded_K = quantize_int4_group(arr, gs)

            save_dict[name] = packed
            save_dict[name + ".scales"] = scales
            save_dict[name + ".meta"] = np.array(
                [arr.shape[0], arr.shape[1], padded_K, gs], dtype=np.int32)
            quantized_names.append(name)

            q_bytes = packed.nbytes + scales.nbytes
            total_quant += q_bytes

            if args.verbose:
                stats = compute_quantization_error(arr, packed, scales, gs)
                print(f"  [INT4] {name}: {arr.shape} "
                      f"({orig_bytes//1024}KB -> {packed.nbytes//1024}KB + "
                      f"scales {scales.nbytes//1024}KB) "
                      f"rel_err={stats['relative_error']:.6f}")
            else:
                ratio = orig_bytes / q_bytes if q_bytes > 0 else 0
                print(f"  [INT4] {name}: {arr.shape} "
                      f"({orig_bytes//1024}KB -> {q_bytes//1024}KB, {ratio:.1f}x)")
        else:
            save_dict[name] = arr.astype(np.float32)
            skipped_names.append(name)
            total_quant += arr.nbytes
            print(f"  [SKIP] {name}: {arr.shape} ({orig_bytes//1024}KB)")

    # 4. Save
    npz_path = output_dir / "quantized_weights_int4.npz"
    print(f"\n  Saving to {npz_path}...")
    np.savez(str(npz_path), **save_dict)
    npz_size = npz_path.stat().st_size

    # 5. Write config
    config = {
        "quantization_method": "absmax_group_int4",
        "quantization_bits": 4,
        "group_size": gs,
        "symmetric": True,
        "packing": "uint8_low_high_nibble",
        "scales_dtype": "float16",
        "quantized_weights": quantized_names,
        "skipped_weights": skipped_names,
        "original_bytes": total_orig,
        "quantized_bytes": total_quant,
        "compression_ratio": round(total_orig / total_quant, 2) if total_quant else 0,
        "npz_compressed_bytes": npz_size,
    }
    with open(output_dir / "quantize_config.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Original: {total_orig / 1024**3:.2f} GB")
    print(f"  Quantized: {total_quant / 1024**3:.2f} GB")
    print(f"  Compression: {total_orig / total_quant:.2f}x")
    print(f"  NPZ file: {npz_size / 1024**2:.1f} MB")
    print(f"  Quantized: {len(quantized_names)} weights")
    print(f"  Skipped: {len(skipped_names)} weights")


if __name__ == "__main__":
    main()
