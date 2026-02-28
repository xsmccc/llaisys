"""
SelfAttention 算子 ncu profiling 脚本
Qwen2-1.5B 参数: nhead=12, kv_head=2, head_dim=128
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test"))

import argparse
import llaisys
from test_utils import random_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"])
    parser.add_argument("--qlen", type=int, default=1, help="query length (常为1, decode phase)")
    parser.add_argument("--kvlen", type=int, default=512, help="total KV length")
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--nkvhead", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    scale = 1.0 / (args.headdim ** 0.5)
    print(f"Profiling SelfAttention:")
    print(f"  Q: [{args.qlen}, {args.nhead}, {args.headdim}] ({args.dtype})")
    print(f"  K: [{args.kvlen}, {args.nkvhead}, {args.headdim}] ({args.dtype})")
    print(f"  V: [{args.kvlen}, {args.nkvhead}, {args.headdim}] ({args.dtype})")
    print(f"  out: [{args.qlen}, {args.nhead}, {args.headdim}] ({args.dtype})")
    print(f"  scale: {scale:.6f}")

    q, q_ = random_tensor((args.qlen, args.nhead, args.headdim), args.dtype, "nvidia", scale=0.1)
    k, k_ = random_tensor((args.kvlen, args.nkvhead, args.headdim), args.dtype, "nvidia", scale=0.1)
    v, v_ = random_tensor((args.kvlen, args.nkvhead, args.headdim), args.dtype, "nvidia", scale=0.1)
    out, out_ = random_tensor((args.qlen, args.nhead, args.headdim), args.dtype, "nvidia")

    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.self_attention(out_, q_, k_, v_, scale)

    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.self_attention(out_, q_, k_, v_, scale)

    print("Done!")

if __name__ == "__main__":
    main()
