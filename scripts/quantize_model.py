#!/usr/bin/env python3
"""
W8A32 离线权重量化工具

将 FP32 safetensors 权重转换为 INT8 per-channel 对称量化格式。

量化方案 (Absmax Per-Channel):
  - 对每个输出通道 n: scale[n] = max(|W[n, :]|) / 127.0
  - W_int8[n, k] = clamp(round(W[n, k] / scale[n]), -128, 127)
  - 还原: W_fp32[n, k] ≈ W_int8[n, k] * scale[n]

只量化 Linear 层的 weight (2D 矩阵), 不量化:
  - Embedding 权重 (查表操作, 量化收益小)
  - RMSNorm 权重 (1D, 元素少)
  - Bias (1D, 元素少)
  - 位置编码 (精度敏感)

输出格式:
  - 量化权重: {name}.int8.npy  (INT8 ndarray)
  - 缩放因子: {name}.scales.npy (F32 ndarray)
  - 非量化权重: {name}.npy (F32 ndarray, 原封不动)

用法:
  python scripts/quantize_model.py \\
      --model-path models/DeepSeek-R1-Distill-Qwen-1.5B \\
      --output-dir models/DeepSeek-R1-Distill-Qwen-1.5B-INT8 \\
      [--skip-lm-head]

  生成的目录结构:
  output_dir/
    config.json               (复制原始)
    tokenizer.json            (复制原始)
    tokenizer_config.json     (复制原始)
    quantized_weights.npz     (所有量化+非量化权重的归档)
    quantize_config.json      (量化配置: 方法、跳过列表等)
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file


def quantize_per_channel_absmax(weight_fp32: np.ndarray):
    """
    Per-channel absmax 对称量化。
    
    Args:
        weight_fp32: F32 权重 [N, K] (out_features × in_features)
    
    Returns:
        weight_int8: INT8 权重 [N, K]
        scales: F32 缩放因子 [N]
    """
    assert weight_fp32.ndim == 2, f"Expected 2D weight, got {weight_fp32.ndim}D"
    
    # Per-channel (per-row) 最大绝对值
    abs_max = np.max(np.abs(weight_fp32), axis=1)  # [N]
    
    # 避免除零: 对于全零行, scale 设为 1.0
    abs_max = np.maximum(abs_max, 1e-10)
    
    # 计算 scale: scale = abs_max / 127
    scales = abs_max / 127.0  # [N]
    
    # 量化: W_int8 = clamp(round(W / scale), -128, 127)
    scales_expanded = scales[:, np.newaxis]  # [N, 1] for broadcasting
    weight_int8 = np.clip(
        np.round(weight_fp32 / scales_expanded),
        -128, 127
    ).astype(np.int8)
    
    return weight_int8, scales.astype(np.float32)


def should_quantize(name: str, skip_lm_head: bool = False) -> bool:
    """判断权重是否应该被量化"""
    # 只量化 Linear 层的 weight (2D矩阵)
    # 跳过: embedding, norm, bias
    if "embed" in name and "weight" in name:
        if name == "lm_head.weight" and not skip_lm_head:
            return True  # LM head 默认量化
        return False  # input embedding 不量化
    
    if "norm" in name:
        return False
    
    if "bias" in name:
        return False
    
    if "weight" in name:
        # 包含 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        return True
    
    return False


def compute_quantization_error(original: np.ndarray, int8: np.ndarray, 
                                scales: np.ndarray) -> dict:
    """计算量化误差统计"""
    # 反量化
    dequantized = int8.astype(np.float32) * scales[:, np.newaxis]
    
    # 误差
    diff = original - dequantized
    mse = np.mean(diff ** 2)
    max_err = np.max(np.abs(diff))
    rel_err = np.sqrt(mse) / (np.std(original) + 1e-10)
    
    return {
        "mse": float(mse),
        "max_abs_error": float(max_err),
        "relative_error": float(rel_err),
        "original_range": [float(original.min()), float(original.max())],
    }


def main():
    parser = argparse.ArgumentParser(description="W8A32 模型权重量化工具")
    parser.add_argument("--model-path", type=str, required=True,
                        help="原始模型目录 (包含 safetensors)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="量化输出目录")
    parser.add_argument("--skip-lm-head", action="store_true",
                        help="跳过 lm_head 的量化 (保持 F32)")
    parser.add_argument("--verbose", action="store_true",
                        help="显示每个权重的量化误差")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. 复制 config & tokenizer 文件 ──
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"  Copied {fname}")
    
    # ── 2. 加载所有 safetensors ──
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        print(f"[ERROR] No safetensors found in {model_path}")
        return
    
    all_weights = {}
    for f in files:
        print(f"  Loading {f.name}...")
        all_weights.update(load_file(str(f)))
    
    print(f"\n  Total weights: {len(all_weights)}")
    
    # ── 3. 量化 ──
    quantized_weights = {}   # name → np.ndarray (int8 or fp32)
    scales_dict = {}         # name → np.ndarray (fp32 scales)
    quantized_names = []     # 被量化的权重名
    skipped_names = []       # 未量化的权重名
    
    total_original_bytes = 0
    total_quantized_bytes = 0
    error_stats = {}
    
    for name, tensor in sorted(all_weights.items()):
        # 转为 F32 numpy
        if tensor.dtype in [torch.bfloat16, torch.float16]:
            tensor = tensor.float()
        arr = tensor.numpy()
        
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        original_bytes = arr.nbytes
        total_original_bytes += original_bytes
        
        if should_quantize(name, args.skip_lm_head) and arr.ndim == 2:
            # 量化
            weight_int8, weight_scales = quantize_per_channel_absmax(arr)
            
            quantized_weights[name] = weight_int8
            scales_dict[name + ".scales"] = weight_scales
            quantized_names.append(name)
            
            q_bytes = weight_int8.nbytes + weight_scales.nbytes
            total_quantized_bytes += q_bytes
            
            if args.verbose:
                stats = compute_quantization_error(arr, weight_int8, weight_scales)
                error_stats[name] = stats
                print(f"  [QUANT] {name}: {arr.shape} F32({original_bytes//1024}KB) → "
                      f"INT8({weight_int8.nbytes//1024}KB) + scales({weight_scales.nbytes}B)  "
                      f"rel_err={stats['relative_error']:.6f}")
            else:
                print(f"  [QUANT] {name}: {arr.shape} → INT8 "
                      f"({original_bytes//1024}KB → {weight_int8.nbytes//1024}KB)")
        else:
            # 不量化, 保持 F32
            quantized_weights[name] = arr
            skipped_names.append(name)
            total_quantized_bytes += original_bytes
            print(f"  [SKIP ] {name}: {arr.shape} (kept as F32)")
    
    # ── 4. 保存量化权重 ──
    # 合并所有权重和 scales 到一个 npz 文件
    save_dict = {}
    save_dict.update(quantized_weights)
    save_dict.update(scales_dict)
    
    npz_path = output_dir / "quantized_weights.npz"
    print(f"\n  Saving to {npz_path}...")
    np.savez(str(npz_path), **save_dict)
    
    # ── 5. 保存量化配置 ──
    quant_config = {
        "quantization_method": "absmax_per_channel",
        "quantization_bits": 8,
        "symmetric": True,
        "quantized_weights": quantized_names,
        "skipped_weights": skipped_names,
        "skip_lm_head": args.skip_lm_head,
        "original_size_mb": total_original_bytes / (1024 * 1024),
        "quantized_size_mb": total_quantized_bytes / (1024 * 1024),
        "compression_ratio": total_original_bytes / max(total_quantized_bytes, 1),
    }
    
    if error_stats:
        quant_config["error_stats"] = error_stats
    
    config_path = output_dir / "quantize_config.json"
    with open(config_path, "w") as f:
        json.dump(quant_config, f, indent=2)
    
    # ── 6. 汇总 ──
    print(f"\n{'='*60}")
    print(f"  量化完成!")
    print(f"  量化权重数: {len(quantized_names)}")
    print(f"  跳过权重数: {len(skipped_names)}")
    print(f"  原始大小:   {total_original_bytes / (1024*1024):.1f} MB")
    print(f"  量化后大小: {total_quantized_bytes / (1024*1024):.1f} MB")
    print(f"  压缩比:     {total_original_bytes / max(total_quantized_bytes, 1):.2f}x")
    print(f"  输出目录:   {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
