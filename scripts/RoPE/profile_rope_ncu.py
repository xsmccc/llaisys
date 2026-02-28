"""
RoPE 算子 ncu profiling 脚本
用法:
  python scripts/RoPE/profile_rope_ncu.py --dtype f32 --seqlen 512 --nhead 4 --headdim 4096
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test"))

import argparse
import llaisys
from test_utils import random_tensor, arrange_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"])
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=4096)
    parser.add_argument("--start-pos", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    shape = (args.seqlen, args.nhead, args.headdim)
    dtype_bytes = {"f32": 4, "f16": 2, "bf16": 2}[args.dtype]
    total_elements = args.seqlen * args.nhead * args.headdim
    data_bytes = total_elements * dtype_bytes * 2  # read + write
    print(f"Profiling RoPE:")
    print(f"  input/output: {shape} ({args.dtype})")
    print(f"  pos_ids:      [{args.seqlen}] (range [{args.start_pos}, {args.start_pos + args.seqlen}))")
    print(f"  数据搬运量:   {data_bytes / 1024 / 1024:.2f} MB (读in + 写out)")

    x, x_ = random_tensor(shape, args.dtype, "nvidia")
    y, y_ = random_tensor(shape, args.dtype, "nvidia")
    pos_ids, pos_ids_ = arrange_tensor(args.start_pos, args.start_pos + args.seqlen, "nvidia")
    theta = 10000.0

    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.rope(y_, x_, pos_ids_, theta)

    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.rope(y_, x_, pos_ids_, theta)

    print("Done!")

if __name__ == "__main__":
    main()
