#!/usr/bin/env python3
"""
LLAISYS 完整基准测试 - 一键运行入口

使用:
  cd scripts/benchmark
  python run_all.py --device nvidia                          # 仅算子
  python run_all.py --device nvidia --model /path/model      # 算子+推理
  python run_all.py --device nvidia --model /path/model --report  # 全部+可视化
"""

import sys, os, argparse, json
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../..")))

def main():
    parser = argparse.ArgumentParser(description="LLAISYS complete benchmark")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--skip-ops", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    ops_json_path = None
    infer_json_path = None

    # Phase 1: Op benchmarks
    if not args.skip_ops:
        print("\n" + "=" * 80)
        print("  Phase 1: Operator Benchmarks")
        print("=" * 80)
        from benchmark_ops import run_all_ops, print_summary_table
        from config import GPU_SPECS, TIMING
        results = run_all_ops(args.device, args.ops)
        valid = [r for r in results if "error" not in r]
        print_summary_table(valid)
        ops_json_path = os.path.join(args.output_dir, f"ops_{args.device}_{timestamp}.json")
        with open(ops_json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"device": args.device, "gpu": GPU_SPECS["name"],
                                "timestamp": datetime.now().isoformat(),
                                "warmup": TIMING["warmup"], "repeat": TIMING["repeat"]},
                       "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nOps results: {ops_json_path}")
        from roofline import compute_roofline_data, print_roofline_analysis
        rf_points = compute_roofline_data(valid)
        print_roofline_analysis(rf_points)

    # Phase 2: Inference benchmarks
    if not args.skip_infer and args.model:
        print("\n" + "=" * 80)
        print("  Phase 2: End-to-End Inference Benchmarks")
        print("=" * 80)
        from benchmark_inference import run_inference_benchmark, print_inference_summary
        from config import GPU_SPECS
        results = run_inference_benchmark(args.model, args.device, args.max_seq_len)
        print_inference_summary(results)
        infer_json_path = os.path.join(args.output_dir, f"infer_{args.device}_{timestamp}.json")
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if k != "text"}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, float):
                return round(obj, 6)
            return obj
        with open(infer_json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"device": args.device, "gpu": GPU_SPECS["name"],
                                "model": str(args.model), "max_seq_len": args.max_seq_len,
                                "timestamp": datetime.now().isoformat()},
                       "results": clean(results)}, f, indent=2, ensure_ascii=False)
        print(f"\nInference results: {infer_json_path}")
    elif not args.skip_infer and not args.model:
        print("\n[INFO] No --model specified, skipping inference benchmarks")

    # Phase 3: Visual report
    if args.report:
        print("\n" + "=" * 80)
        print("  Phase 3: Generating Visual Report")
        print("=" * 80)
        from visualize import generate_report
        generate_report(ops_json_path, infer_json_path, args.output_dir)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    if ops_json_path:
        print(f"  Ops: {ops_json_path}")
    if infer_json_path:
        print(f"  Inference: {infer_json_path}")
    if args.report:
        print(f"  Report: {args.output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
