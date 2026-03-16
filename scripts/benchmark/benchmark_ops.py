#!/usr/bin/env python3
"""
算子级基准测试
测量每个算子的: 执行时间(ms) | TFLOPS/带宽(GB/s) | 算术强度 | vs PyTorch 加速比
"""

import sys
import os
import time
import json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../..")))

import torch
import llaisys

from config import (
    GPU_SPECS, TIMING, OP_CONFIGS,
    calc_tflops, calc_bandwidth_GBs, arithmetic_intensity,
    _numel, _elem_size,
)

# ============================================================
#  设备 / dtype 转换
# ============================================================

def llaisys_device(name):
    return {"cpu": llaisys.DeviceType.CPU, "nvidia": llaisys.DeviceType.NVIDIA}[name]

def torch_device(name, device_id=0):
    if name == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}")

def llaisys_dtype(name):
    return {"f16": llaisys.DataType.F16, "f32": llaisys.DataType.F32,
            "bf16": llaisys.DataType.BF16, "i64": llaisys.DataType.I64}[name]

def torch_dtype(name):
    return {"f16": torch.float16, "f32": torch.float32,
            "bf16": torch.bfloat16, "i64": torch.int64}[name]

# ============================================================
#  张量工厂
# ============================================================

def random_tensor(shape, dtype_name, device_name):
    t = torch.randn(shape, dtype=torch_dtype(dtype_name), device=torch_device(device_name))
    l = llaisys.Tensor(shape, dtype=llaisys_dtype(dtype_name),
                       device=llaisys_device(device_name), device_id=0)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(l.data_ptr(), t.data_ptr(),
                    t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return t, l

def zero_tensor(shape, dtype_name, device_name):
    t = torch.zeros(shape, dtype=torch_dtype(dtype_name), device=torch_device(device_name))
    l = llaisys.Tensor(shape, dtype=llaisys_dtype(dtype_name),
                       device=llaisys_device(device_name), device_id=0)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(l.data_ptr(), t.data_ptr(),
                    t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return t, l

def arrange_tensor(start, end, device_name):
    t = torch.arange(start, end, device=torch_device(device_name))
    l = llaisys.Tensor((end - start,), dtype=llaisys_dtype("i64"),
                       device=llaisys_device(device_name), device_id=0)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(l.data_ptr(), t.data_ptr(),
                    t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return t, l

def random_int_tensor(shape, device_name, low=0, high=512):
    t = torch.randint(low, high, shape, dtype=torch.int64, device=torch_device(device_name))
    l = llaisys.Tensor(shape, dtype=llaisys_dtype("i64"),
                       device=llaisys_device(device_name), device_id=0)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(l.data_ptr(), t.data_ptr(),
                    t.numel() * t.element_size(), llaisys.MemcpyKind.D2D)
    return t, l

# ============================================================
#  精确计时
# ============================================================

def benchmark_op(torch_fn, llaisys_fn, device_name, warmup=None, repeat=None):
    warmup = warmup or TIMING["warmup"]
    repeat = repeat or TIMING["repeat"]
    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def _time(fn):
        for _ in range(warmup):
            fn()
        api.device_synchronize()
        if device_name == "nvidia":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        api.device_synchronize()
        if device_name == "nvidia":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / repeat

    torch_time = _time(torch_fn)
    llaisys_time = _time(llaisys_fn)
    return torch_time, llaisys_time

# ============================================================
#  各算子 benchmark
# ============================================================

def bench_add(cfg, device):
    shape, dtype = cfg["shape"], cfg["dtype"]
    a, a_ = random_tensor(shape, dtype, device)
    b, b_ = random_tensor(shape, dtype, device)
    c, c_ = random_tensor(shape, dtype, device)
    return benchmark_op(
        lambda: torch.add(a, b, out=c),
        lambda: llaisys.Ops.add(c_, a_, b_), device)

def bench_rms_norm(cfg, device):
    shape, dtype = cfg["shape"], cfg["dtype"]
    x, x_ = random_tensor(shape, dtype, device)
    w, w_ = random_tensor([shape[-1]], dtype, device)
    out, out_ = random_tensor(shape, dtype, device)
    eps = 1e-6
    def torch_rms(o, x, w, eps):
        rms = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
        o.copy_((x.float() / rms * w.float()).to(x.dtype))
    return benchmark_op(
        lambda: torch_rms(out, x, w, eps),
        lambda: llaisys.Ops.rms_norm(out_, x_, w_, eps), device)

def bench_swiglu(cfg, device):
    shape, dtype = cfg["shape"], cfg["dtype"]
    gate, gate_ = random_tensor(shape, dtype, device)
    up, up_ = random_tensor(shape, dtype, device)
    out, out_ = random_tensor(shape, dtype, device)
    return benchmark_op(
        lambda: out.copy_(torch.nn.functional.silu(gate) * up),
        lambda: llaisys.Ops.swiglu(out_, gate_, up_), device)

def bench_linear(cfg, device):
    x_shape, w_shape, dtype = cfg["x_shape"], cfg["w_shape"], cfg["dtype"]
    x, x_ = random_tensor(x_shape, dtype, device)
    w, w_ = random_tensor(w_shape, dtype, device)
    out, out_ = random_tensor([x_shape[0], w_shape[0]], dtype, device)
    bias, bias_ = zero_tensor([w_shape[0]], dtype, device)
    return benchmark_op(
        lambda: out.copy_(torch.nn.functional.linear(x, w)),
        lambda: llaisys.Ops.linear(out_, x_, w_, bias_), device)

def bench_rope(cfg, device):
    shape, dtype = cfg["shape"], cfg["dtype"]
    start, end = cfg["pos_range"]
    x, x_ = random_tensor(shape, dtype, device)
    y, y_ = random_tensor(shape, dtype, device)
    pos, pos_ = arrange_tensor(start, end, device)
    theta = 10000.0
    def torch_rope(y, x, pos, theta):
        seq_len, n_heads, head_dim = x.shape
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
        angles = torch.outer(pos.float(), freqs).view(seq_len, 1, head_dim // 2)
        cos_f, sin_f = angles.cos(), angles.sin()
        x1, x2 = x.float().reshape(seq_len, n_heads, head_dim // 2, 2).unbind(-1)
        y.copy_(torch.stack([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1).flatten(2).to(x.dtype))
    return benchmark_op(
        lambda: torch_rope(y, x, pos, theta),
        lambda: llaisys.Ops.rope(y_, x_, pos_, theta), device)

def bench_embedding(cfg, device):
    dtype = cfg["dtype"]
    idx, idx_ = random_int_tensor(cfg["idx_shape"], device, 0, cfg["embd_shape"][0])
    embd, embd_ = random_tensor(cfg["embd_shape"], dtype, device)
    out_shape = list(cfg["idx_shape"]) + [cfg["embd_shape"][1]]
    out, out_ = random_tensor(out_shape, dtype, device)
    return benchmark_op(
        lambda: out.copy_(torch.embedding(embd, idx)),
        lambda: llaisys.Ops.embedding(out_, idx_, embd_), device)

def bench_argmax(cfg, device):
    shape, dtype = cfg["shape"], cfg["dtype"]
    vals, vals_ = random_tensor(shape, dtype, device)
    max_idx, max_idx_ = zero_tensor([1], "i64", device)
    max_val, max_val_ = zero_tensor([1], dtype, device)
    def torch_argmax():
        max_idx.fill_(vals.argmax().item())
        max_val.copy_(vals.max())
    return benchmark_op(
        torch_argmax,
        lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_), device)

def bench_self_attention(cfg, device):
    params, dtype = cfg["params"], cfg["dtype"]
    qlen, kvlen, nh, nkvh, hd = params
    q, q_ = random_tensor([qlen, nh, hd], dtype, device)
    k, k_ = random_tensor([kvlen, nkvh, hd], dtype, device)
    v, v_ = random_tensor([kvlen, nkvh, hd], dtype, device)
    out, out_ = random_tensor([qlen, nh, hd], dtype, device)
    scale = 1.0 / (hd ** 0.5)
    def torch_attn():
        q_4d = q.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_4d = k.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_4d = v.permute(1, 0, 2).unsqueeze(0).contiguous()
        if nh != nkvh:
            k_4d = k_4d.repeat(1, nh // nkvh, 1, 1)
            v_4d = v_4d.repeat(1, nh // nkvh, 1, 1)
        o = torch.nn.functional.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=(qlen == kvlen))
        out.copy_(o.squeeze(0).permute(1, 0, 2).contiguous())
    return benchmark_op(torch_attn,
        lambda: llaisys.Ops.self_attention(out_, q_, k_, v_, scale), device)

# ============================================================
#  调度器
# ============================================================
BENCH_DISPATCH = {
    "add": bench_add, "rms_norm": bench_rms_norm, "swiglu": bench_swiglu,
    "linear": bench_linear, "rope": bench_rope, "embedding": bench_embedding,
    "argmax": bench_argmax, "self_attention": bench_self_attention,
}

def compute_metrics(op_name, cfg, torch_time, llaisys_time, dtype):
    op_cfg = OP_CONFIGS[op_name]
    flops_fn = op_cfg["flops_fn"]
    bytes_fn = op_cfg["bytes_fn"]
    if op_name in ("linear", "embedding"):
        flops = flops_fn(cfg, dtype)
        nbytes = bytes_fn(cfg, dtype)
    elif op_name == "self_attention":
        flops = flops_fn(cfg["params"], dtype)
        nbytes = bytes_fn(cfg["params"], dtype)
    elif op_name == "rope":
        flops = flops_fn(cfg["shape"], dtype)
        nbytes = bytes_fn(cfg["shape"], dtype)
    else:
        flops = flops_fn(cfg["shape"], dtype)
        nbytes = bytes_fn(cfg["shape"], dtype)
    ai = arithmetic_intensity(flops, nbytes)
    return {
        "flops": flops, "bytes": nbytes, "arithmetic_intensity": ai,
        "torch_tflops": calc_tflops(flops, torch_time),
        "llaisys_tflops": calc_tflops(flops, llaisys_time),
        "torch_bandwidth_GBs": calc_bandwidth_GBs(nbytes, torch_time),
        "llaisys_bandwidth_GBs": calc_bandwidth_GBs(nbytes, llaisys_time),
        "speedup": torch_time / llaisys_time if llaisys_time > 0 else 0,
    }

def run_all_ops(device_name, ops_filter=None):
    results = []
    ops_to_test = ops_filter or list(OP_CONFIGS.keys())
    for op_name in ops_to_test:
        if op_name not in OP_CONFIGS or op_name not in BENCH_DISPATCH:
            print(f"  [SKIP] Unknown op: {op_name}")
            continue
        bench_fn = BENCH_DISPATCH[op_name]
        configs = OP_CONFIGS[op_name]["configs"]
        for cfg in configs:
            label = cfg.get("label", "")
            dtype = cfg["dtype"]
            full_name = f"{op_name}/{label}"
            print(f"  Running {full_name} ...", end=" ", flush=True)
            try:
                torch_time, llaisys_time = bench_fn(cfg, device_name)
                metrics = compute_metrics(op_name, cfg, torch_time, llaisys_time, dtype)
                result = {
                    "op": op_name, "label": label, "dtype": dtype,
                    "config": {k: v for k, v in cfg.items() if k not in ("label", "dtype")},
                    "torch_time_ms": torch_time * 1000,
                    "llaisys_time_ms": llaisys_time * 1000,
                    **metrics,
                }
                results.append(result)
                print(f"Torch={torch_time*1000:.4f}ms  LLAISYS={llaisys_time*1000:.4f}ms  "
                      f"Speedup={metrics['speedup']:.2f}x")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"op": op_name, "label": label, "dtype": dtype, "error": str(e)})
    return results

def print_summary_table(results):
    print()
    print("=" * 120)
    print(f"{'Op/Label':<35} {'Dtype':<6} {'Torch(ms)':<12} {'LLAISYS(ms)':<14} "
          f"{'Speedup':<10} {'TFLOPS':<10} {'BW(GB/s)':<12} {'AI(F/B)':<10}")
    print("=" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['op']+'/'+r['label']:<35} {r['dtype']:<6} {'ERROR':>12}  {r['error']}")
            continue
        print(f"{r['op']+'/'+r['label']:<35} {r['dtype']:<6} "
              f"{r['torch_time_ms']:>10.4f}  {r['llaisys_time_ms']:>12.4f}  "
              f"{r['speedup']:>8.2f}x  "
              f"{r['llaisys_tflops']:>8.4f}  "
              f"{r['llaisys_bandwidth_GBs']:>10.2f}  "
              f"{r['arithmetic_intensity']:>8.2f}")
    print("=" * 120)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLAISYS 算子基准测试")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print("=" * 80)
    print(f"LLAISYS 算子基准测试")
    print(f"设备: {args.device}  |  GPU: {GPU_SPECS['name']}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Warmup: {TIMING['warmup']}  Repeat: {TIMING['repeat']}")
    print("=" * 80)
    results = run_all_ops(args.device, args.ops)
    valid_results = [r for r in results if "error" not in r]
    print_summary_table(valid_results)
    output_path = args.output or f"benchmark_ops_{args.device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"device": args.device, "gpu": GPU_SPECS["name"],
                     "timestamp": datetime.now().isoformat(),
                     "warmup": TIMING["warmup"], "repeat": TIMING["repeat"]},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    return results

if __name__ == "__main__":
    main()
