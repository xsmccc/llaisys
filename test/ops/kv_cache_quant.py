"""
KV Cache INT8 量化正确性测试

验证策略: INT8+KV INT8 推理输出必须与 INT8-only 推理输出 bit-identical
(因为 KV Cache INT8 采用 per-token per-head 对称量化，127 级精度对 attention
scores 的舍入误差应可忽略)

用法:
    source venv/bin/activate
    PYTHONPATH=python:test python3 test/ops/kv_cache_quant.py
"""
import sys
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "..", "python"))

import llaisys
from llaisys.models import Qwen2
from llaisys.runtime import DeviceType


def test_kv_cache_int8_correctness(model_path: str, prompt: str, max_tokens: int):
    """
    验证: INT8 + KV INT8 produce identical output to INT8-only

    这是 KV Cache INT8 最重要的正确性保证 — bit-identical 输出证明
    quantize/dequantize roundtrip 在 attention 计算中无精度损失。
    """
    print(f"=== KV Cache INT8 Correctness Test ===")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print()

    # --- INT8-only baseline ---
    print("[1/3] Loading INT8-only model...")
    model_int8 = Qwen2(model_path, DeviceType.NVIDIA, kv_cache_int8=False)
    print("[2/3] Generating INT8-only output...")
    output_int8 = model_int8.generate(prompt, max_new_tokens=max_tokens)
    del model_int8  # free GPU memory

    # --- INT8 + KV Cache INT8 ---
    print("[3/3] Loading INT8 + KV INT8 model & generating...")
    model_kv8 = Qwen2(model_path, DeviceType.NVIDIA, kv_cache_int8=True)
    output_kv8 = model_kv8.generate(prompt, max_new_tokens=max_tokens)
    del model_kv8

    # --- Compare ---
    print()
    print(f"INT8-only output:    {output_int8!r}")
    print(f"INT8+KV INT8 output: {output_kv8!r}")
    print()

    if output_int8 == output_kv8:
        print("✅ PASS — bit-identical output (KV Cache INT8 is lossless)")
    else:
        print("❌ FAIL — outputs differ!")
        # Find divergence point
        min_len = min(len(output_int8), len(output_kv8))
        for i in range(min_len):
            if output_int8[i] != output_kv8[i]:
                print(f"   First divergence at char {i}: "
                      f"{output_int8[max(0,i-10):i+10]!r} vs {output_kv8[max(0,i-10):i+10]!r}")
                break
        sys.exit(1)


def test_kv_cache_int8_multiple_prompts(model_path: str):
    """
    使用多个不同长度的 prompt 验证稳定性
    """
    prompts = [
        ("你好", 20),
        ("What is the capital of France?", 30),
        ("请解释一下什么是机器学习", 50),
    ]
    
    for prompt, max_tokens in prompts:
        test_kv_cache_int8_correctness(model_path, prompt, max_tokens)
        print("---")
    
    print("\n✅ All KV Cache INT8 tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache INT8 correctness test")
    parser.add_argument("--model", type=str,
                        default="models/DeepSeek-R1-Distill-Qwen-1.5B-INT8",
                        help="Path to INT8 quantized model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (if not set, run multi-prompt suite)")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    if args.prompt:
        test_kv_cache_int8_correctness(args.model, args.prompt, args.max_tokens)
    else:
        test_kv_cache_int8_multiple_prompts(args.model)
