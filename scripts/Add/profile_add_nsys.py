#!/usr/bin/env python3
"""
使用此脚本生成nsys profiling数据
运行方式：
    nsys profile --stats=true -o add_profile python scripts/profile_add_nsys.py --dtype f32
    
查看结果：
    nsys stats add_profile.nsys-rep
    或使用 Nsight Systems GUI 打开 add_profile.nsys-rep
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


def profile_add_kernel(dtype_name, shape, warmup=10, iterations=1000):
    """针对特定dtype的add算子进行profiling"""
    print(f"Profiling add operator:")
    print(f"  Data type: {dtype_name}")
    print(f"  Shape: {shape}")
    print(f"  Iterations: {iterations}")
    print("-" * 50)

    # 创建设备
    device_name = "nvidia"
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    
    # 创建tensor
    numel = shape[0] * shape[1]
    _, a = random_tensor(shape, dtype_name, device_name)
    _, b = random_tensor(shape, dtype_name, device_name)
    _, c = random_tensor(shape, dtype_name, device_name)
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        llaisys.Ops.add(c, a, b)
    api.device_synchronize()
    
    # 开始profiling
    print("Starting profiling...")
    api.device_synchronize()
    start = time.time()
    
    for _ in range(iterations):
        llaisys.Ops.add(c, a, b)
    
    api.device_synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000  # ms
    bandwidth = (numel * 3 * dtype_size(dtype_name) / (1024**3)) / ((end - start) / iterations)  # GB/s
    
    print(f"Average time: {avg_time:.5f} ms")
    print(f"Bandwidth: {bandwidth:.2f} GB/s")
    print(f"Total elements: {numel}")
    

def dtype_size(dtype_name):
    """返回数据类型的字节数"""
    sizes = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
    }
    return sizes[dtype_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CUDA add operator with nsys")
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"], 
                        help="Data type to profile")
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 4096],
                        help="Tensor shape (default: 512 4096)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of profiling iterations")
    
    args = parser.parse_args()
    
    profile_add_kernel(
        dtype_name=args.dtype,
        shape=tuple(args.shape),
        warmup=args.warmup,
        iterations=args.iterations
    )
