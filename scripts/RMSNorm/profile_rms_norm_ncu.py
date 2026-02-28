"""
RMSNorm 算子 ncu profiling 脚本
用法:
  python scripts/RMSNorm/profile_rms_norm_ncu.py --dtype f32 --rows 512 --cols 4096
  sudo ncu --set full --kernel-name regex:"rms_norm_kernel.*" --launch-skip 5 --launch-count 1 \
    python scripts/RMSNorm/profile_rms_norm_ncu.py --dtype f32 --rows 512 --cols 4096
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
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2}[args.dtype]
    # 读 in(2次) + 读 weight(1次) + 写 out(1次) = 3*rows*cols + cols
    data_bytes = (3 * args.rows * args.cols + args.cols) * dtype_bytes
    print(f"Profiling RMSNorm:")
    print(f"  input:    [{args.rows}, {args.cols}] ({args.dtype})")
    print(f"  weight:   [{args.cols}] ({args.dtype})")
    print(f"  数据搬运量: {data_bytes / 1024 / 1024:.2f} MB (2×读in + 读weight + 写out)")

    x, x_ = random_tensor((args.rows, args.cols), args.dtype, "nvidia")
    w, w_ = random_tensor((args.cols,), args.dtype, "nvidia")
    out, out_ = random_tensor((args.rows, args.cols), args.dtype, "nvidia")
    eps = 1e-5

    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.rms_norm(out_, x_, w_, eps)

    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.rms_norm(out_, x_, w_, eps)

    print("Done!")

if __name__ == "__main__":
    main()
