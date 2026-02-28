#!/usr/bin/env python3
"""
SwiGLU 算子性能对比脚本
对比 naive 版本和 PyTorch 的性能
"""

import sys
import os
import time
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(parent_dir, "python"))
sys.path.insert(0, os.path.join(parent_dir, "test"))

import llaisys
import torch
from test_utils import random_tensor


def benchmark_swiglu(dtype_name, shape, warmup=50, iterations=1000):
    """测试 swiglu 算子性能"""
    device_name = "nvidia"
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    numel = shape[0] * shape[1]

    # 创建 llaisys tensors
    _, gate = random_tensor(shape, dtype_name, device_name)
    _, up = random_tensor(shape, dtype_name, device_name)
    _, out = random_tensor(shape, dtype_name, device_name)

    # 创建 PyTorch tensors
    torch_dtypes = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    torch_gate = torch.randn(shape, dtype=torch_dtypes[dtype_name], device="cuda")
    torch_up = torch.randn(shape, dtype=torch_dtypes[dtype_name], device="cuda")

    # ---- LLAISYS Benchmark ----
    for _ in range(warmup):
        llaisys.Ops.swiglu(out, gate, up)
    api.device_synchronize()

    start = time.time()
    for _ in range(iterations):
        llaisys.Ops.swiglu(out, gate, up)
    api.device_synchronize()
    llaisys_time_ms = (time.time() - start) / iterations * 1000

    # ---- PyTorch Benchmark ----
    for _ in range(warmup):
        _ = torch.nn.functional.silu(torch_gate) * torch_up
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        _ = torch.nn.functional.silu(torch_gate) * torch_up
    torch.cuda.synchronize()
    torch_time_ms = (time.time() - start) / iterations * 1000

    # 计算带宽 (读 gate + 读 up + 写 out = 3次内存访问)
    dtype_sizes = {"f32": 4, "f16": 2, "bf16": 2}
    bytes_transferred = numel * 3 * dtype_sizes[dtype_name]
    llaisys_bw = bytes_transferred / (1024**3) / (llaisys_time_ms / 1000)
    torch_bw = bytes_transferred / (1024**3) / (torch_time_ms / 1000)

    return llaisys_time_ms, llaisys_bw, torch_time_ms, torch_bw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark swiglu operator")
    parser.add_argument("--dtypes", nargs="+", default=["f32", "f16", "bf16"],
                        choices=["f32", "f16", "bf16"])
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 4096])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    shape = tuple(args.shape)
    numel = shape[0] * shape[1]

    print("=" * 80)
    print(f"SwiGLU Operator Benchmark (Naive CUDA)")
    print(f"Shape: {shape} ({numel:,} elements)")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 80)
    print(f"{'DType':<8} {'LLAISYS (ms)':<14} {'BW (GB/s)':<12} "
          f"{'PyTorch (ms)':<14} {'BW (GB/s)':<12} {'Ratio':<8}")
    print("-" * 80)

    for dtype in args.dtypes:
        ll_time, ll_bw, pt_time, pt_bw = benchmark_swiglu(
            dtype, shape, args.warmup, args.iterations)
        ratio = ll_time / pt_time
        print(f"{dtype:<8} {ll_time:<14.6f} {ll_bw:<12.2f} "
              f"{pt_time:<14.6f} {pt_bw:<12.2f} {ratio:<8.2f}x")

    print("=" * 80)
    print("\n提示:")
    print("  - Ratio < 1.0 表示 LLAISYS 更快, > 1.0 表示 PyTorch 更快")
    print("  - SwiGLU 是访存密集型算子, 带宽利用率是关键指标")
    print("  - RTX 4070 Laptop 理论带宽 ~256 GB/s")
