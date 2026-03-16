#!/usr/bin/env python3
"""可视化报告生成"""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed")

from config import GPU_SPECS

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_roofline(roofline_points, output_path="report/roofline.png"):
    if not HAS_MPL:
        return
    _ensure_dir(output_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bw = GPU_SPECS["mem_bandwidth_GBs"]
    peak_f32 = GPU_SPECS["fp32_tflops"]
    ridge = peak_f32 * 1e3 / bw
    ai_range = np.logspace(-2, 4, 500)
    roofline = np.minimum(peak_f32, bw * ai_range / 1e3)
    ax.plot(ai_range, roofline, "k-", linewidth=2, label=f"Roofline (FP32 peak={peak_f32:.1f} TFLOPS)")
    ax.axvline(x=ridge, color="gray", linestyle="--", alpha=0.5, label=f"Ridge Point ({ridge:.1f} F/B)")
    ax.fill_between(ai_range, 0, roofline, alpha=0.05, color="blue")
    colors = {"memory-bound": "royalblue", "compute-bound": "crimson"}
    markers = {"f32": "o", "f16": "s", "bf16": "D"}
    for p in roofline_points:
        ai = p["arithmetic_intensity"]
        achieved = p["achieved_tflops"]
        if ai <= 0 or achieved <= 0:
            continue
        c = colors.get(p["bottleneck"], "gray")
        m = markers.get(p["dtype"], "o")
        ax.scatter(ai, achieved, c=c, marker=m, s=80, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(p["name"], (ai, achieved), fontsize=7, textcoords="offset points", xytext=(5, 5), rotation=15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Model - {GPU_SPECS['name']}", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 10000); ax.set_ylim(0.0001, peak_f32 * 2)
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Roofline saved: {output_path}")

def plot_op_speedup(op_results, output_path="report/op_speedup.png"):
    if not HAS_MPL:
        return
    _ensure_dir(output_path)
    valid = [r for r in op_results if "error" not in r and r.get("speedup", 0) > 0]
    if not valid:
        return
    names = [f"{r['op']}/{r['label']}\n({r['dtype']})" for r in valid]
    speedups = [r["speedup"] for r in valid]
    colors = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in speedups]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(names) * 0.8), 6))
    x = np.arange(len(names))
    bars = ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="LLAISYS = PyTorch")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{s:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Speedup (higher is better)", fontsize=11)
    ax.set_title("LLAISYS vs PyTorch - Operator Speedup", fontsize=13)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Speedup chart saved: {output_path}")

def plot_op_time_comparison(op_results, output_path="report/op_time_compare.png"):
    if not HAS_MPL:
        return
    _ensure_dir(output_path)
    valid = [r for r in op_results if "error" not in r]
    if not valid:
        return
    names = [f"{r['op']}/{r['label']}" for r in valid]
    torch_times = [r["torch_time_ms"] for r in valid]
    llaisys_times = [r["llaisys_time_ms"] for r in valid]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(names) * 0.8), 6))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, torch_times, w, label="PyTorch", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, llaisys_times, w, label="LLAISYS", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Operator Execution Time - PyTorch vs LLAISYS", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Time comparison saved: {output_path}")

def plot_bandwidth_utilization(op_results, output_path="report/bandwidth_util.png"):
    if not HAS_MPL:
        return
    _ensure_dir(output_path)
    valid = [r for r in op_results if "error" not in r and r.get("llaisys_bandwidth_GBs", 0) > 0]
    if not valid:
        return
    peak_bw = GPU_SPECS["mem_bandwidth_GBs"]
    names = [f"{r['op']}/{r['label']}" for r in valid]
    bws = [r["llaisys_bandwidth_GBs"] for r in valid]
    utils = [b / peak_bw * 100 for b in bws]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(names) * 0.7), 6))
    x = np.arange(len(names))
    colors = ["#2ecc71" if u >= 50 else "#f39c12" if u >= 25 else "#e74c3c" for u in utils]
    ax.bar(x, utils, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1, label=f"Peak ({peak_bw:.0f} GB/s)")
    ax.axhline(y=50, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="50%")
    for i, (u, b) in enumerate(zip(utils, bws)):
        ax.text(i, u + 1, f"{b:.0f}\nGB/s", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Bandwidth Utilization (%)", fontsize=11)
    ax.set_title(f"Memory BW Utilization - {GPU_SPECS['name']}", fontsize=13)
    ax.set_ylim(0, 120); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Bandwidth chart saved: {output_path}")

def plot_inference_comparison(infer_results, output_path="report/inference_compare.png"):
    if not HAS_MPL:
        return
    _ensure_dir(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels_set = set()
    for fw_key in ["hf", "llaisys", "llaisys_int8"]:
        for r in infer_results.get(fw_key, []):
            labels_set.add(r["label"])
    labels_list = sorted(labels_set)
    fw_display = {"hf": "HuggingFace", "llaisys": "LLAISYS-FP32", "llaisys_int8": "LLAISYS-INT8"}
    colors_fw = {"hf": "#3498db", "llaisys": "#e67e22", "llaisys_int8": "#2ecc71"}
    x = np.arange(len(labels_list))
    n_fw = sum(1 for k in ["hf", "llaisys", "llaisys_int8"] if infer_results.get(k))
    if n_fw == 0:
        return
    w = 0.8 / n_fw
    # Decode tokens/s
    ax = axes[0]; offset = 0
    for fw_key in ["hf", "llaisys", "llaisys_int8"]:
        data = infer_results.get(fw_key, [])
        if not data:
            continue
        m = {r["label"]: r["decode_tokens_per_s"] for r in data}
        vals = [m.get(l, 0) for l in labels_list]
        ax.bar(x + offset * w - (n_fw-1)*w/2, vals, w, label=fw_display[fw_key], color=colors_fw[fw_key], edgecolor="black", linewidth=0.5)
        offset += 1
    ax.set_xticks(x); ax.set_xticklabels(labels_list, fontsize=10)
    ax.set_ylabel("Decode Tokens/s"); ax.set_title("Decode Speed"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    # Prefill latency
    ax = axes[1]; offset = 0
    for fw_key in ["hf", "llaisys", "llaisys_int8"]:
        data = infer_results.get(fw_key, [])
        if not data:
            continue
        m = {r["label"]: r["prefill_time_s"] * 1000 for r in data}
        vals = [m.get(l, 0) for l in labels_list]
        ax.bar(x + offset * w - (n_fw-1)*w/2, vals, w, label=fw_display[fw_key], color=colors_fw[fw_key], edgecolor="black", linewidth=0.5)
        offset += 1
    ax.set_xticks(x); ax.set_xticklabels(labels_list, fontsize=10)
    ax.set_ylabel("Prefill Latency (ms)"); ax.set_title("Prefill Latency"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"End-to-End Inference - {GPU_SPECS['name']}", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Inference chart saved: {output_path}")

def generate_report(ops_json=None, infer_json=None, output_dir="report"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Generating visual report -> {output_dir}/")
    print(f"{'='*60}")
    if ops_json:
        with open(ops_json, "r") as f:
            ops_data = json.load(f)
        op_results = ops_data["results"]
        from roofline import compute_roofline_data
        rf_points = compute_roofline_data(op_results)
        plot_roofline(rf_points, os.path.join(output_dir, "roofline.png"))
        plot_op_speedup(op_results, os.path.join(output_dir, "op_speedup.png"))
        plot_op_time_comparison(op_results, os.path.join(output_dir, "op_time_compare.png"))
        plot_bandwidth_utilization(op_results, os.path.join(output_dir, "bandwidth_util.png"))
    if infer_json:
        with open(infer_json, "r") as f:
            infer_data = json.load(f)
        plot_inference_comparison(infer_data["results"], os.path.join(output_dir, "inference_compare.png"))
    print(f"\nReport generation complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual benchmark report")
    parser.add_argument("--ops", default=None, help="benchmark_ops JSON")
    parser.add_argument("--infer", default=None, help="benchmark_inference JSON")
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()
    generate_report(args.ops, args.infer, args.output_dir)
