#!/usr/bin/env python3
"""
简单的性能对比脚本，对比不同数据类型的add算子性能
运行方式：
    python scripts/benchmark_add.py
    python scripts/benchmark_add.py --dtypes f32 f16 --shape 1024 4096
"""

import sys
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(parent_dir, "python"))
sys.path.insert(0, os.path.join(parent_dir, "test"))

import llaisys
import torch
import time
from test_utils import random_tensor


def benchmark_add(dtype_name, shape, warmup=50, iterations=1000):
    """测试add算子性能"""
    device_name = "nvidia"
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    
    # 创建tensors
    numel = shape[0] * shape[1]
    _, a = random_tensor(shape, dtype_name, device_name)
    _, b = random_tensor(shape, dtype_name, device_name)
    _, c = random_tensor(shape, dtype_name, device_name)
    
    # Warmup
    for _ in range(warmup):
        llaisys.Ops.add(c, a, b)
    api.device_synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        llaisys.Ops.add(c, a, b)
    api.device_synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / iterations * 1000
    
    # 计算带宽 (3次内存访问: 读a, 读b, 写c)
    dtype_sizes = {"f32": 4, "f16": 2, "bf16": 2}
    bytes_transferred = numel * 3 * dtype_sizes[dtype_name]
    bandwidth_gbs = bytes_transferred / (1024**3) / ((end - start) / iterations)
    
    return avg_time_ms, bandwidth_gbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark add operator across dtypes")
    parser.add_argument("--dtypes", nargs="+", default=["f32", "f16", "bf16"],
                        choices=["f32", "f16", "bf16"],
                        help="Data types to benchmark")
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 4096],
                        help="Tensor shape")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Benchmark iterations")
    
    args = parser.parse_args()
    shape = tuple(args.shape)
    numel = shape[0] * shape[1]
    
    print("=" * 70)
    print(f"Add Operator Benchmark")
    print(f"Shape: {shape} ({numel:,} elements)")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 70)
    print(f"{'DType':<10} {'Time (ms)':<15} {'Bandwidth (GB/s)':<20} {'Speedup':<10}")
    print("-" * 70)
    
    results = {}
    for dtype in args.dtypes:
        avg_time, bandwidth = benchmark_add(dtype, shape, args.warmup, args.iterations)
        results[dtype] = (avg_time, bandwidth)
    
    # 以f32为基准计算speedup
    f32_time = results.get("f32", (None, None))[0]
    
    for dtype in args.dtypes:
        avg_time, bandwidth = results[dtype]
        speedup = f32_time / avg_time if f32_time else 1.0
        print(f"{dtype:<10} {avg_time:<15.6f} {bandwidth:<20.2f} {speedup:<10.2f}x")
    
    print("=" * 70)
    
    # 理论峰值带宽分析 (假设是现代GPU)
    print("\n提示:")
    print("  - 现代GPU内存带宽通常在 400-900 GB/s 之间")
    print("  - Add算子是内存受限操作，带宽利用率是关键指标")
    print("  - F16/BF16应该通过向量化获得更高的有效带宽")
