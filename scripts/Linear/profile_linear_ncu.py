"""
Linear 算子 ncu profiling 脚本
主要分析 bias add kernel（GEMM 由 cuBLAS 处理）
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
    parser.add_argument("--M", type=int, default=512, help="rows (seq_len)")
    parser.add_argument("--K", type=int, default=4096, help="in_features")
    parser.add_argument("--N", type=int, default=4096, help="out_features")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print(f"Profiling Linear:")
    print(f"  in:     [{args.M}, {args.K}] ({args.dtype})")
    print(f"  weight: [{args.N}, {args.K}] ({args.dtype})")
    print(f"  bias:   [{args.N}] ({args.dtype})")
    print(f"  out:    [{args.M}, {args.N}] ({args.dtype})")
    flops = 2 * args.M * args.N * args.K
    print(f"  FLOPs:  {flops / 1e9:.2f} GFLOP")

    x, x_ = random_tensor((args.M, args.K), args.dtype, "nvidia", scale=0.1)
    w, w_ = random_tensor((args.N, args.K), args.dtype, "nvidia", scale=0.01)
    bias, bias_ = random_tensor((args.N,), args.dtype, "nvidia")
    out, out_ = random_tensor((args.M, args.N), args.dtype, "nvidia")

    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.linear(out_, x_, w_, bias_)

    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.linear(out_, x_, w_, bias_)

    print("Done!")

if __name__ == "__main__":
    main()
