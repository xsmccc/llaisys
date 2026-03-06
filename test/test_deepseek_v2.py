#!/usr/bin/env python3
"""
DeepSeek-V2-Lite 推理测试

用法:
  CPU:  python test/test_deepseek_v2.py --device cpu
  GPU:  python test/test_deepseek_v2.py --device nvidia  (需要 A100 80GB)
"""

import argparse
import sys
import time
sys.path.insert(0, "python")

from llaisys.models.deepseek_v2 import DeepSeekV2
from llaisys.libllaisys import DeviceType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/DeepSeek-V2-Lite",
                        help="模型路径")
    parser.add_argument("--device", choices=["cpu", "nvidia"], default="cpu",
                        help="推理设备")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="KV cache 最大序列长度")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="最大生成 token 数")
    parser.add_argument("--prompt", default="Hello, how are you?",
                        help="测试 prompt")
    args = parser.parse_args()

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU

    # 创建模型
    print("=" * 60)
    model = DeepSeekV2(args.model, device=device, max_seq_len=args.max_seq_len)

    # 加载分词器
    import json
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f"{args.model}/tokenizer.json")

    # 编码
    prompt = args.prompt
    print(f"\nPrompt: {prompt}")
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    print(f"Token IDs ({len(token_ids)}): {token_ids[:20]}...")

    # 推理
    print(f"\nGenerating (max {args.max_tokens} tokens)...")
    t0 = time.time()
    output_ids = model.generate(
        token_ids,
        max_new_tokens=args.max_tokens,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
    )
    elapsed = time.time() - t0

    # 解码
    generated_ids = output_ids[len(token_ids):]
    text = tokenizer.decode(generated_ids)
    total_tokens = len(generated_ids)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Generated: {text}")
    print(f"{'=' * 60}")
    print(f"Tokens: {total_tokens}, Time: {elapsed:.2f}s, Speed: {tokens_per_sec:.1f} tok/s")

    # 清理
    model.reset()
    print("Done.")


if __name__ == "__main__":
    main()
