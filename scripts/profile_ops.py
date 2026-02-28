#!/usr/bin/env python3
"""
算子性能分析脚本
用于测试并记录各个算子的时间耗损占比
"""

import sys
import os
import time
from datetime import datetime

# 添加父目录到路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch


def benchmark_op(torch_func, llaisys_func, device_name, warmup=10, repeat=100):
    """性能测试函数"""
    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def time_op(func):
        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        end = time.time()
        return (end - start) / repeat

    torch_time = time_op(torch_func)
    llaisys_time = time_op(llaisys_func)
    return torch_time * 1000, llaisys_time * 1000  # 转换为毫秒


def llaisys_device(device_name: str):
    """设备类型转换"""
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def torch_device(device_name: str, device_id=0):
    """PyTorch 设备转换"""
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def llaisys_dtype(dtype_name: str):
    """dtype字符串转llaisys类型"""
    if dtype_name == "f16":
        return llaisys.DataType.F16
    elif dtype_name == "f32":
        return llaisys.DataType.F32
    elif dtype_name == "f64":
        return llaisys.DataType.F64
    elif dtype_name == "bf16":
        return llaisys.DataType.BF16
    elif dtype_name == "i64":
        return llaisys.DataType.I64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")


def torch_dtype(dtype_name: str):
    """dtype字符串转torch类型"""
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i64":
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")


def random_tensor(shape, dtype_name, device_name):
    """创建随机张量"""
    torch_tensor = torch.randn(shape, dtype=torch_dtype(dtype_name), device=torch_device(device_name))
    
    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=0
    )
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D
    )
    
    return torch_tensor, llaisys_tensor


def zero_tensor(shape, dtype_name, device_name):
    """创建零张量"""
    torch_tensor = torch.zeros(shape, dtype=torch_dtype(dtype_name), device=torch_device(device_name))
    
    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=0
    )
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D
    )
    
    return torch_tensor, llaisys_tensor


def arrange_tensor(start, end, device_name):
    """创建递增序列张量"""
    torch_tensor = torch.arange(start, end, device=torch_device(device_name))
    
    llaisys_tensor = llaisys.Tensor(
        (end - start,),
        dtype=llaisys_dtype("i64"),
        device=llaisys_device(device_name),
        device_id=0
    )
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D
    )
    
    return torch_tensor, llaisys_tensor


def random_int_tensor(shape, device_name, low=0, high=512):
    """创建随机整数张量"""
    torch_tensor = torch.randint(low, high, shape, dtype=torch.int64, device=torch_device(device_name))
    
    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype("i64"),
        device=llaisys_device(device_name),
        device_id=0
    )
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        llaisys.MemcpyKind.D2D
    )
    
    return torch_tensor, llaisys_tensor


# ==================== 算子测试函数 ====================

def profile_add(device_name, warmup, repeat):
    """Add 算子性能测试"""
    results = []
    test_configs = [
        ([512, 4096], "f32"),
        ([512, 4096], "f16"),
        ([512, 4096], "bf16"),
    ]
    
    for shape, dtype in test_configs:
        a, a_ = random_tensor(shape, dtype, device_name)
        b, b_ = random_tensor(shape, dtype, device_name)
        c, c_ = random_tensor(shape, dtype, device_name)
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch.add(a, b, out=c),
            lambda: llaisys.Ops.add(c_, a_, b_),
            device_name, warmup, repeat
        )
        results.append((f"add_{shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_argmax(device_name, warmup, repeat):
    """Argmax 算子性能测试"""
    results = []
    test_configs = [
        ([4096], "f32"),
        ([4096], "f16"),
        ([4096], "bf16"),
    ]
    
    def torch_argmax(max_idx, max_val, vals):
        max_idx.fill_(vals.argmax().item())
        max_val.copy_(vals.max())
    
    for shape, dtype in test_configs:
        vals, vals_ = random_tensor(shape, dtype, device_name)
        max_idx, max_idx_ = zero_tensor([1], "i64", device_name)
        max_val, max_val_ = zero_tensor([1], dtype, device_name)
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name, warmup, repeat
        )
        results.append((f"argmax_{shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_embedding(device_name, warmup, repeat):
    """Embedding 算子性能测试"""
    results = []
    test_configs = [
        ([50], [512, 4096], "f32"),
        ([50], [512, 4096], "f16"),
    ]
    
    def torch_embedding(out, idx, embd):
        out.copy_(torch.embedding(embd, idx))
    
    for idx_shape, embd_shape, dtype in test_configs:
        idx, idx_ = random_int_tensor(idx_shape, device_name, 0, embd_shape[0])
        embd, embd_ = random_tensor(embd_shape, dtype, device_name)
        out, out_ = random_tensor(idx_shape + [embd_shape[1]], dtype, device_name)
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_embedding(out, idx, embd),
            lambda: llaisys.Ops.embedding(out_, idx_, embd_),
            device_name, warmup, repeat
        )
        results.append((f"embedding_{idx_shape}x{embd_shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_linear(device_name, warmup, repeat):
    """Linear 算子性能测试"""
    results = []
    test_configs = [
        ([512, 4096], [8192, 4096], "f32"),
        ([512, 4096], [8192, 4096], "f16"),
    ]
    
    def torch_linear(out, x, w):
        out.copy_(torch.nn.functional.linear(x, w))
    
    for x_shape, w_shape, dtype in test_configs:
        x, x_ = random_tensor(x_shape, dtype, device_name)
        w, w_ = random_tensor(w_shape, dtype, device_name)
        out, out_ = random_tensor([x_shape[0], w_shape[0]], dtype, device_name)
        bias, bias_ = zero_tensor([w_shape[0]], dtype, device_name)  # 添加bias参数
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_linear(out, x, w),
            lambda: llaisys.Ops.linear(out_, x_, w_, bias_),  # 传入bias
            device_name, warmup, repeat
        )
        results.append((f"linear_{x_shape}x{w_shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_rms_norm(device_name, warmup, repeat):
    """RMS Norm 算子性能测试"""
    results = []
    test_configs = [
        ([512, 4096], "f32"),
        ([512, 4096], "f16"),
        ([512, 4096], "bf16"),
    ]
    
    def torch_rms_norm(ans, x, w, eps):
        rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        ans.copy_(x / rms * w)
    
    for shape, dtype in test_configs:
        x, x_ = random_tensor(shape, dtype, device_name)
        w, w_ = random_tensor([shape[-1]], dtype, device_name)
        out, out_ = random_tensor(shape, dtype, device_name)
        eps = 1e-5
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_rms_norm(out, x, w, eps),
            lambda: llaisys.Ops.rms_norm(out_, x_, w_, eps),
            device_name, warmup, repeat
        )
        results.append((f"rms_norm_{shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_rope(device_name, warmup, repeat):
    """RoPE 算子性能测试"""
    results = []
    test_configs = [
        ([512, 4, 128], [512, 1024], "f32"),
        ([512, 4, 128], [512, 1024], "f16"),
    ]
    
    def torch_rope(y, x, pos_ids, theta):
        seq_len, n_heads, head_dim = x.shape
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
        pos = pos_ids.float()
        freqs = torch.outer(pos, freqs).float()
        cos_freqs = freqs.cos().view(seq_len, 1, head_dim // 2)
        sin_freqs = freqs.sin().view(seq_len, 1, head_dim // 2)
        x1, x2 = x.float().reshape(seq_len, n_heads, head_dim // 2, 2).unbind(-1)
        y.copy_((torch.stack([x1 * cos_freqs - x2 * sin_freqs, x2 * cos_freqs + x1 * sin_freqs], dim=-1).flatten(2)).to(x.dtype))
    
    for shape, (start, end), dtype in test_configs:
        x, x_ = random_tensor(shape, dtype, device_name)
        pos_ids, pos_ids_ = arrange_tensor(start, end, device_name)
        y, y_ = random_tensor(shape, dtype, device_name)
        theta = 10000.0
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_rope(y, x, pos_ids, theta),
            lambda: llaisys.Ops.rope(y_, x_, pos_ids_, theta),
            device_name, warmup, repeat
        )
        results.append((f"rope_{shape}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_self_attention(device_name, warmup, repeat):
    """Self-Attention 算子性能测试"""
    results = []
    test_configs = [
        ([32, 32, 8, 8, 128], "f32"),
        ([32, 32, 8, 8, 128], "f16"),
    ]
    
    def torch_self_attention(attn_val, query, key, value, scale):
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_val.copy_(torch.matmul(attn_weights, value))
    
    for config, dtype in test_configs:
        qlen, kvlen, nh, nkvh, hd = config
        q, q_ = random_tensor([qlen, nh, hd], dtype, device_name)
        k, k_ = random_tensor([kvlen, nkvh, hd], dtype, device_name)
        v, v_ = random_tensor([kvlen, nkvh, hd], dtype, device_name)
        attn_val, attn_val_ = random_tensor([qlen, nh, hd], dtype, device_name)
        scale = 1.0 / (hd ** 0.5)
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_self_attention(attn_val, q, k, v, scale),
            lambda: llaisys.Ops.self_attention(attn_val_, q_, k_, v_, scale),
            device_name, warmup, repeat
        )
        results.append((f"self_attention_{qlen}x{kvlen}x{nh}x{hd}_{dtype}", torch_time, llaisys_time))
    
    return results


def profile_swiglu(device_name, warmup, repeat):
    """SwiGLU 算子性能测试"""
    results = []
    test_configs = [
        ([512, 4096], "f32"),
        ([512, 4096], "f16"),
        ([512, 4096], "bf16"),
    ]
    
    def torch_swiglu(out, gate, up):
        out.copy_(torch.nn.functional.silu(gate) * up)
    
    for shape, dtype in test_configs:
        gate, gate_ = random_tensor(shape, dtype, device_name)
        up, up_ = random_tensor(shape, dtype, device_name)
        out, out_ = random_tensor(shape, dtype, device_name)
        
        torch_time, llaisys_time = benchmark_op(
            lambda: torch_swiglu(out, gate, up),
            lambda: llaisys.Ops.swiglu(out_, gate_, up_),
            device_name, warmup, repeat
        )
        results.append((f"swiglu_{shape}_{dtype}", torch_time, llaisys_time))
    
    return results


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="算子性能分析脚本")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], help="设备类型")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--repeat", type=int, default=100, help="重复次数")
    parser.add_argument("--output", default="profile_results.txt", help="输出文件")
    parser.add_argument("--ops", nargs="+", default=None, 
                        help="指定算子 (add, argmax, embedding, linear, rms_norm, rope, self_attention, swiglu)")
    
    args = parser.parse_args()
    
    # 算子列表
    all_ops = {
        "add": profile_add,
        "argmax": profile_argmax,
        "embedding": profile_embedding,
        "linear": profile_linear,
        "rms_norm": profile_rms_norm,
        "rope": profile_rope,
        "self_attention": profile_self_attention,
        "swiglu": profile_swiglu,
    }
    
    # 选择要测试的算子
    ops_to_test = args.ops if args.ops else list(all_ops.keys())
    
    print("=" * 80)
    print(f"算子性能分析")
    print(f"设备: {args.device}")
    print(f"预热次数: {args.warmup}")
    print(f"重复次数: {args.repeat}")
    print(f"测试算子: {', '.join(ops_to_test)}")
    print("=" * 80)
    print()
    
    all_results = []
    
    # 运行测试
    for op_name in ops_to_test:
        if op_name not in all_ops:
            print(f"警告: 未知算子 '{op_name}'，跳过")
            continue
            
        print(f"正在测试 {op_name}...")
        try:
            results = all_ops[op_name](args.device, args.warmup, args.repeat)
            all_results.extend(results)
            for name, torch_time, llaisys_time in results:
                speedup = torch_time / llaisys_time if llaisys_time > 0 else 0
                print(f"  {name:50s} | Torch: {torch_time:8.4f} ms | LLAISYS: {llaisys_time:8.4f} ms | 加速比: {speedup:6.2f}x")
        except Exception as e:
            print(f"  错误: {e}")
        print()
    
    # 计算统计信息
    if all_results:
        total_torch_time = sum(r[1] for r in all_results)
        total_llaisys_time = sum(r[2] for r in all_results)
        
        print("=" * 80)
        print("汇总统计")
        print("=" * 80)
        print(f"总 Torch 时间: {total_torch_time:.4f} ms")
        print(f"总 LLAISYS 时间: {total_llaisys_time:.4f} ms")
        print(f"平均加速比: {total_torch_time / total_llaisys_time:.2f}x")
        print()
        
        # 按耗时排序
        sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
        print("按 LLAISYS 耗时排序 (前10):")
        print("-" * 80)
        for name, torch_time, llaisys_time in sorted_results[:10]:
            percentage = (llaisys_time / total_llaisys_time) * 100
            print(f"  {name:50s} | {llaisys_time:8.4f} ms | {percentage:5.1f}%")
        print()
        
        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"算子性能分析结果\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"设备: {args.device}\n")
            f.write(f"预热次数: {args.warmup}\n")
            f.write(f"重复次数: {args.repeat}\n")
            f.write("=" * 80 + "\n\n")
            
            for name, torch_time, llaisys_time in all_results:
                speedup = torch_time / llaisys_time if llaisys_time > 0 else 0
                percentage = (llaisys_time / total_llaisys_time) * 100
                f.write(f"{name:50s} | Torch: {torch_time:8.4f} ms | LLAISYS: {llaisys_time:8.4f} ms | "
                       f"加速比: {speedup:6.2f}x | 占比: {percentage:5.1f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"总 Torch 时间: {total_torch_time:.4f} ms\n")
            f.write(f"总 LLAISYS 时间: {total_llaisys_time:.4f} ms\n")
            f.write(f"平均加速比: {total_torch_time / total_llaisys_time:.2f}x\n")
        
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
