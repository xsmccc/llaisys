#!/usr/bin/env python3
"""Roofline 模型分析"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GPU_SPECS

def classify_op(ai, dtype="f32"):
    bw = GPU_SPECS["mem_bandwidth_GBs"]
    peak = GPU_SPECS["fp32_tflops"] if dtype == "f32" else GPU_SPECS["fp16_tflops"]
    ridge_point = peak * 1e3 / bw
    return "memory-bound" if ai < ridge_point else "compute-bound"

def compute_roofline_data(results):
    points = []
    for r in results:
        if "error" in r:
            continue
        ai = r.get("arithmetic_intensity", 0)
        achieved_tflops = r.get("llaisys_tflops", 0)
        achieved_bw = r.get("llaisys_bandwidth_GBs", 0)
        dtype = r.get("dtype", "f32")
        bw = GPU_SPECS["mem_bandwidth_GBs"]
        peak = GPU_SPECS["fp32_tflops"] if dtype == "f32" else GPU_SPECS["fp16_tflops"]
        theoretical_tflops = min(peak, bw * ai / 1e3) if ai > 0 else 0
        efficiency = achieved_tflops / theoretical_tflops * 100 if theoretical_tflops > 0 else 0
        points.append({
            "name": f"{r['op']}/{r['label']}", "dtype": dtype,
            "arithmetic_intensity": ai, "achieved_tflops": achieved_tflops,
            "achieved_bandwidth_GBs": achieved_bw,
            "theoretical_tflops": theoretical_tflops,
            "efficiency_pct": efficiency, "bottleneck": classify_op(ai, dtype),
        })
    return points

def print_roofline_analysis(points):
    bw = GPU_SPECS["mem_bandwidth_GBs"]
    fp32_peak = GPU_SPECS["fp32_tflops"]
    ridge = fp32_peak * 1e3 / bw
    print()
    print("=" * 110)
    print(f"Roofline 模型分析  |  {GPU_SPECS['name']}")
    print(f"峰值带宽: {bw:.0f} GB/s  |  FP32 峰值: {fp32_peak:.1f} TFLOPS  |  Ridge Point: {ridge:.1f} FLOP/Byte")
    print("=" * 110)
    print(f"{'Op/Label':<35} {'AI(F/B)':<10} {'Achieved':<12} {'Theory':<12} "
          f"{'Efficiency':<12} {'Bottleneck':<15}")
    print("-" * 110)
    for p in sorted(points, key=lambda x: x["arithmetic_intensity"]):
        print(f"{p['name']:<35} {p['arithmetic_intensity']:<10.2f} "
              f"{p['achieved_tflops']:<12.4f} {p['theoretical_tflops']:<12.4f} "
              f"{p['efficiency_pct']:<12.1f}% {p['bottleneck']:<15}")
    print("=" * 110)
    mem_bound = [p for p in points if p["bottleneck"] == "memory-bound"]
    comp_bound = [p for p in points if p["bottleneck"] == "compute-bound"]
    print(f"\nMemory-bound 算子: {len(mem_bound)}")
    for p in mem_bound:
        print(f"  {p['name']}: BW={p['achieved_bandwidth_GBs']:.1f} GB/s "
              f"({p['achieved_bandwidth_GBs']/bw*100:.0f}% of peak)")
    print(f"\nCompute-bound 算子: {len(comp_bound)}")
    for p in comp_bound:
        peak = GPU_SPECS["fp32_tflops"] if p["dtype"] == "f32" else GPU_SPECS["fp16_tflops"]
        print(f"  {p['name']}: {p['achieved_tflops']:.4f} TFLOPS "
              f"({p['achieved_tflops']/peak*100:.0f}% of peak)")
    return points

def load_and_analyze(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    points = compute_roofline_data(data["results"])
    print_roofline_analysis(points)
    return points

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roofline 模型分析")
    parser.add_argument("input", help="benchmark_ops JSON 结果文件")
    args = parser.parse_args()
    load_and_analyze(args.input)
