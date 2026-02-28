#!/usr/bin/env python3
"""
==========================================================
Add 算子 ncu Profiling 脚本
==========================================================

【这个脚本做什么？】
  它就是一个"跑 add kernel 的小程序"。
  ncu 不能直接分析 CUDA kernel —— 它需要一个"宿主程序"来触发 kernel launch。
  这个脚本就是那个宿主程序：创建 GPU 张量 → 调用 add 算子 → 触发 CUDA kernel。
  ncu 会在 kernel launch 时"拦截"并分析。

【运行方式】
  不要直接 python xxx.py，而是用 ncu 包裹它：
  
    ncu [ncu的参数] python scripts/Add/profile_add_ncu.py [脚本的参数]
    ~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ncu 分析器        被分析的 Python 程序
  
  例如：
    ncu --set full --kernel-name regex:"add_kernel.*" --launch-skip 5 --launch-count 1 \
        python scripts/Add/profile_add_ncu.py --dtype f32 --iterations 10

【ncu 参数说明】
  --set full           : 采集所有指标（内存、计算、occupancy 等）
  --kernel-name regex: : 只分析名字匹配的 kernel（过滤掉 PyTorch 内部的 kernel）
  --launch-skip N      : 跳过前 N 次 kernel launch（跳过 warmup 阶段）
  --launch-count N     : 只分析 N 次 kernel launch
  -o filename          : 保存报告到 filename.ncu-rep（可用 ncu-ui 可视化打开）

【脚本参数说明】
  --dtype f32/f16/bf16 : 选择测试的数据类型
  --shape 512 4096     : 张量形状（两个整数）
  --iterations 10      : kernel 执行次数（ncu 从中采样分析）
"""

import sys
import os
import argparse

# ============================================================
# 路径设置
# ============================================================
# 脚本在 scripts/Add/ 目录下，需要将项目根目录加入 Python 路径
# 这样才能 import llaisys 和 test_utils
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(parent_dir, "python"))  # → 能 import llaisys
sys.path.insert(0, os.path.join(parent_dir, "test"))    # → 能 import test_utils

import llaisys      # 我们的推理框架（C++ 后端 + Python 前端）
import torch         # PyTorch，用于生成随机测试数据
from test_utils import random_tensor
# random_tensor(shape, dtype, device) 做了以下事情：
#   1. 用 PyTorch 创建一个随机张量（在 GPU 上）
#   2. 用 llaisys 创建一个同形状的空张量（在 GPU 上）
#   3. 把 PyTorch 张量的数据拷贝到 llaisys 张量
#   4. 返回 (torch_tensor, llaisys_tensor)


def profile_single_kernel(dtype_name, shape, iterations=100):
    """
    执行 add kernel 若干次，供 ncu 捕获分析。
    
    【为什么要执行多次？】
    ncu 通过 "kernel replay" 机制工作：它会对同一个 kernel launch 重放多次，
    每次采集不同的硬件计数器（因为硬件计数器数量有限，不能一次全采）。
    --set full 大约需要 8 个 pass（你之前看到的 "0%....50%....100% - 8 passes"）。
    
    iterations 参数控制我们主动执行多少次 kernel launch。
    配合 --launch-skip 和 --launch-count，ncu 只会挑选其中一次来分析。
    """
    print(f"Running add kernel for NCU profiling:")
    print(f"  Data type: {dtype_name}")
    print(f"  Shape: {shape}")
    print(f"  Iterations: {iterations}")
    print("-" * 50)

    # ------- 第1步：初始化 GPU 环境 -------
    device_name = "nvidia"
    # 创建 RuntimeAPI 对象，这会触发 CUDA context 初始化
    # （第一次创建时 CUDA driver 会做很多初始化工作，比较慢）
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    
    # ------- 第2步：创建 GPU 上的测试张量 -------
    numel = shape[0] * shape[1]
    # random_tensor 返回 (torch_tensor, llaisys_tensor)
    # 我们只需要 llaisys_tensor（下划线开头的变量），torch_tensor 用不到
    _, a = random_tensor(shape, dtype_name, device_name)  # 输入 A
    _, b = random_tensor(shape, dtype_name, device_name)  # 输入 B
    _, c = random_tensor(shape, dtype_name, device_name)  # 输出 C
    
    # ------- 第3步：Warmup（预热）-------
    # 【为什么需要 warmup？】
    # 第一次调用 CUDA kernel 时会有额外开销：
    #   - JIT 编译（如果用了 CUDA Graph 等）
    #   - GPU 从低功耗状态唤醒、频率爬升
    #   - 内存分配器首次分配
    # warmup 几次后，GPU 进入稳定状态，后续的测量才准确
    for _ in range(5):
        llaisys.Ops.add(c, a, b)  # 每次调用都会 launch 一次 CUDA kernel
    
    # device_synchronize: 等待 GPU 上所有操作完成
    # CUDA kernel 是异步的（CPU 发出指令后不等 GPU 完成就继续执行）
    # 这里同步一下确保 warmup 全部完成
    api.device_synchronize()
    
    print("Starting kernel execution for NCU capture...")
    
    # ------- 第4步：正式执行（被 ncu 分析的部分）-------
    # ncu 的 --launch-skip 5 会跳过 warmup 的 5 次 launch
    # 然后 --launch-count 1 会捕获这里的第 1 次 launch 来分析
    for i in range(iterations):
        llaisys.Ops.add(c, a, b)  # ← ncu 在这里拦截 kernel 并分析
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{iterations}")
    
    # 等待所有操作完成
    api.device_synchronize()
    print("Kernel execution completed!")


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CUDA add operator with ncu")
    
    # --dtype: 选择要分析的数据类型
    # 不同 dtype 会触发不同的 kernel（f32→add_kernel_f32_vec, f16→add_kernel_f16_vec）
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "bf16"], 
                        help="Data type to profile")
    
    # --shape: 张量形状，影响 kernel 的 grid 大小和执行时间
    # 较大的 shape 能让 kernel 运行够久，ncu 分析更准确
    parser.add_argument("--shape", nargs=2, type=int, default=[512, 4096],
                        help="Tensor shape (default: 512 4096)")
    
    # --iterations: kernel 执行次数
    # 配合 ncu 的 --launch-skip 和 --launch-count 使用
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of kernel launches (ncu will sample from these)")
    
    args = parser.parse_args()
    
    # 调用主函数
    profile_single_kernel(
        dtype_name=args.dtype,
        shape=tuple(args.shape),
        iterations=args.iterations
    )
