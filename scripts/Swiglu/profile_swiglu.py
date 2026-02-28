#!/usr/bin/env python3
"""
SwiGLU 算子性能分析脚本
用于 ncu/nsys profiling
"""

import sys
import os
import argparse

# 添加路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(parent_dir, "python"))
sys.path.insert(0, os.path.join(parent_dir, "test"))

import llaisys
from test_utils import random_tensor


def main():
    parser = argparse.ArgumentParser(description="Profile swiglu operator")
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f32")
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 4096])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    
    shape = tuple(args.shape)
    dtype = args.dtype
    
    print(f"Profiling SwiGLU: shape={shape}, dtype={dtype}")
    print(f"Elements: {shape[0] * shape[1]:,}")
    
    # 创建测试数据
    _, gate = random_tensor(shape, dtype, "nvidia")
    _, up = random_tensor(shape, dtype, "nvidia")
    _, out = random_tensor(shape, dtype, "nvidia")
    
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    
    # 预热
    print(f"Warmup: {args.warmup} iterations...")
    for _ in range(args.warmup):
        llaisys.Ops.swiglu(out, gate, up)
    api.device_synchronize()
    
    # 实际测试
    print(f"Running: {args.iterations} iterations...")
    for _ in range(args.iterations):
        llaisys.Ops.swiglu(out, gate, up)
    api.device_synchronize()
    
    print("Done!")


if __name__ == "__main__":
    main()
