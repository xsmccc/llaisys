"""
Argmax 算子 ncu profiling 脚本
用法:
  python scripts/Argmax/profile_argmax_ncu.py --dtype f32 --numel 151936
  sudo ncu --set full --kernel-name regex:"argmax_kernel.*" --launch-skip 5 --launch-count 1 \
    python scripts/Argmax/profile_argmax_ncu.py --dtype f32 --numel 151936
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test"))

import argparse
import llaisys
from test_utils import random_tensor, zero_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"])
    parser.add_argument("--numel", type=int, default=151936, help="Number of elements")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2}[args.dtype]
    data_bytes = args.numel * dtype_bytes
    print(f"Profiling Argmax:")
    print(f"  vals:     [{args.numel}] ({args.dtype})")
    print(f"  数据量:   {data_bytes / 1024:.2f} KB (读vals)")

    vals, vals_ = random_tensor((args.numel,), args.dtype, "nvidia")
    max_idx, max_idx_ = zero_tensor((1,), "i64", "nvidia")
    max_val, max_val_ = zero_tensor((1,), args.dtype, "nvidia")

    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    print("Done!")

if __name__ == "__main__":
    main()
