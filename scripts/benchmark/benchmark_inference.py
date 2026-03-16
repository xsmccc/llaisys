#!/usr/bin/env python3
"""
端到端推理基准测试 (内存安全版本)
测量: prefill 延迟 | decode tokens/s | 总 tokens/s | 峰值显存 | vs PyTorch 对比

关键设计:
- 每个框架单独加载/测试/释放, 避免同时占用显存
- 默认生成 token 数控制在 16~32, 避免内存爆炸
- 自动监控 GPU 显存, 超限时中止
"""

import sys
import os
import time
import json
import gc
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

import torch
import llaisys

from config import GPU_SPECS, MODEL_PARAMS


# ============================================================
#  安全参数: 防止 OOM
# ============================================================
VRAM_LIMIT_MB = 7000          # GPU 显存安全上限 (8GB 卡留 1GB 余量)
MAX_SEQ_LEN_SAFE = 256        # KV cache 最大长度 (降到 256 节省显存)

DEFAULT_PROMPTS = [
    {"prompt": "Hello",                              "max_tokens": 16, "label": "short"},
    {"prompt": "Explain quantum computing briefly.",  "max_tokens": 24, "label": "medium"},
    {"prompt": "Write a Python sort function.",       "max_tokens": 32, "label": "long"},
]


def check_gpu_mem(limit_mb=VRAM_LIMIT_MB):
    """检查 GPU 显存是否超限"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e6
        if used > limit_mb:
            raise MemoryError(f"GPU 显存 {used:.0f}MB 超过安全限制 {limit_mb}MB, 中止")
    return True


def force_cleanup():
    """强制清理 GPU 和 CPU 内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
#  CUDA 显存监控
# ============================================================

class MemoryTracker:
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.reset()

    def reset(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def snapshot(self):
        if not self.enabled:
            return {"allocated_MB": 0, "reserved_MB": 0, "peak_MB": 0}
        torch.cuda.synchronize()
        return {
            "allocated_MB": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved_MB": round(torch.cuda.memory_reserved() / 1e6, 1),
            "peak_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        }


# ============================================================
#  HuggingFace 推理 (参考基线) — 单独加载/测试/释放
# ============================================================

def benchmark_hf(model_path, device_name, prompts, mem):
    """HuggingFace BF16 推理, 返回结果列表"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[1/3] HuggingFace (BF16) 推理基准...")
    results = []

    device = "cuda" if device_name == "nvidia" else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Warmup (1 token only)
    with torch.no_grad():
        ids = tokenizer("Hi", return_tensors="pt").input_ids.to(device)
        _ = model.generate(ids, max_new_tokens=1, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for p in prompts:
        mem.reset()
        check_gpu_mem()
        label = p["label"]
        max_tok = p["max_tokens"]

        messages = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        prefill_len = input_ids.shape[1]

        print(f"  [{label}] '{p['prompt'][:40]}' max_tokens={max_tok} input_len={prefill_len}")

        # 预填充
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        # 完整生成
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_tok, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        gen_ids = out[0][prefill_len:]
        n_gen = len(gen_ids)
        t_decode = t_total - t_prefill if n_gen > 1 else 0

        r = {
            "label": label,
            "num_input_tokens": prefill_len,
            "num_output_tokens": n_gen,
            "prefill_time_s": t_prefill,
            "decode_time_s": t_decode,
            "total_time_s": t_total,
            "prefill_tokens_per_s": prefill_len / t_prefill if t_prefill > 0 else 0,
            "decode_tokens_per_s": (n_gen - 1) / t_decode if t_decode > 0 else 0,
            "total_tokens_per_s": n_gen / t_total if t_total > 0 else 0,
            "memory": mem.snapshot(),
            "text": tokenizer.decode(gen_ids, skip_special_tokens=True),
        }
        results.append(r)
        print(f"    → {n_gen} tokens, prefill={t_prefill*1000:.1f}ms, "
              f"decode={r['decode_tokens_per_s']:.1f} tok/s, peak={r['memory']['peak_MB']:.0f}MB")

    # 彻底释放 HF 模型
    del model, tokenizer
    force_cleanup()
    print("  HF 模型已释放")
    return results


# ============================================================
#  LLAISYS 推理 — 单独加载/测试/释放
# ============================================================

def benchmark_llaisys(model_path, device_name, max_seq_len, prompts, mem,
                      quantized=False, label_prefix="LLAISYS-FP32"):
    """LLAISYS 推理, 返回结果列表"""
    from transformers import AutoTokenizer

    tag = "3/3" if quantized else "2/3"
    dtype_str = "INT8" if quantized else "FP32"
    print(f"\n[{tag}] {label_prefix} ({dtype_str}) 推理基准...")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU

    # 查找模型路径 (INT8 尝试 -INT8 后缀)
    actual_path = model_path
    if quantized:
        int8_path = str(model_path).rstrip("/") + "-INT8"
        if os.path.isdir(int8_path):
            actual_path = int8_path
        else:
            print(f"  INT8 模型不存在 ({int8_path}), 跳过")
            return results

    model = llaisys.models.Qwen2(
        actual_path, device=device, max_seq_len=max_seq_len, quantized=quantized,
    )

    # Warmup
    msgs = [{"role": "user", "content": "Hi"}]
    warmup_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    warmup_ids = tokenizer(warmup_text, return_tensors="pt").input_ids[0].tolist()
    _ = model.generate(warmup_ids, max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for p in prompts:
        mem.reset()
        check_gpu_mem()
        label = p["label"]
        max_tok = p["max_tokens"]

        messages = [{"role": "user", "content": p["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        prefill_len = len(input_ids)

        print(f"  [{label}] '{p['prompt'][:40]}' max_tokens={max_tok} input_len={prefill_len}")

        # 预填充 (1 token)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(input_ids, max_new_tokens=1, top_k=1, top_p=1.0, temperature=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        # 完整生成
        t0 = time.perf_counter()
        output_ids = model.generate(input_ids, max_new_tokens=max_tok,
                                    top_k=1, top_p=1.0, temperature=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        n_gen = len(output_ids) - prefill_len if len(output_ids) > prefill_len else len(output_ids)
        t_decode = t_total - t_prefill if n_gen > 1 else 0
        gen_ids = output_ids[-n_gen:] if n_gen > 0 else output_ids
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

        r = {
            "label": label,
            "num_input_tokens": prefill_len,
            "num_output_tokens": n_gen,
            "prefill_time_s": t_prefill,
            "decode_time_s": t_decode,
            "total_time_s": t_total,
            "prefill_tokens_per_s": prefill_len / t_prefill if t_prefill > 0 else 0,
            "decode_tokens_per_s": (n_gen - 1) / t_decode if t_decode > 0 else 0,
            "total_tokens_per_s": n_gen / t_total if t_total > 0 else 0,
            "memory": mem.snapshot(),
            "text": decoded,
        }
        results.append(r)
        print(f"    → {n_gen} tokens, prefill={t_prefill*1000:.1f}ms, "
              f"decode={r['decode_tokens_per_s']:.1f} tok/s, peak={r['memory']['peak_MB']:.0f}MB")

    del model, tokenizer
    force_cleanup()
    print(f"  {label_prefix} 模型已释放")
    return results


# ============================================================
#  综合基准测试
# ============================================================

def run_single_stage(model_path, device_name, max_seq_len, prompts, stage):
    """在当前进程中运行单个 benchmark 阶段"""
    mem = MemoryTracker(device_name == "nvidia")
    if stage == "hf":
        return benchmark_hf(model_path, device_name, prompts, mem)
    elif stage == "fp32":
        return benchmark_llaisys(model_path, device_name, max_seq_len, prompts, mem,
                                 quantized=False, label_prefix="LLAISYS-FP32")
    elif stage == "int8":
        return benchmark_llaisys(model_path, device_name, max_seq_len, prompts, mem,
                                 quantized=True, label_prefix="LLAISYS-INT8")
    return []


def run_inference_benchmark(model_path, device_name="nvidia",
                            max_seq_len=MAX_SEQ_LEN_SAFE,
                            prompts=None):
    """子进程隔离模式: 每个框架在独立进程中运行, 避免显存残留干扰"""
    import subprocess
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    results = {"hf": [], "llaisys": [], "llaisys_int8": []}
    script_path = os.path.abspath(__file__)

    for stage, key in [("hf", "hf"), ("fp32", "llaisys"), ("int8", "llaisys_int8")]:
        print(f"\n{'='*60}")
        print(f"  启动子进程: {stage} 阶段")
        print(f"{'='*60}")

        tmp_out = os.path.join(parent_dir, f"_tmp_bench_{stage}.json")
        cmd = [
            sys.executable, script_path,
            "--device", device_name,
            "--model", str(model_path),
            "--max-seq-len", str(max_seq_len),
            "--stage", stage,
            "--output", tmp_out,
        ]

        try:
            proc = subprocess.run(cmd, timeout=300, capture_output=False)
            if proc.returncode == 0 and os.path.exists(tmp_out):
                with open(tmp_out, "r") as f:
                    data = json.load(f)
                results[key] = data.get("results", {}).get(key, [])
                os.remove(tmp_out)
            else:
                print(f"  {stage} 阶段失败 (returncode={proc.returncode})")
        except subprocess.TimeoutExpired:
            print(f"  {stage} 阶段超时")
        except Exception as e:
            print(f"  {stage} 阶段异常: {e}")

    return results


def print_inference_summary(results):
    """打印推理基准汇总表"""
    print()
    print("=" * 110)
    print(f"{'Framework':<20} {'Label':<10} {'Input':<8} {'Output':<8} "
          f"{'Prefill(ms)':<14} {'Decode(tok/s)':<16} {'Total(tok/s)':<14} {'Peak(MB)':<10}")
    print("=" * 110)

    for fw_name in ["hf", "llaisys", "llaisys_int8"]:
        fw_label = {"hf": "HuggingFace", "llaisys": "LLAISYS-FP32", "llaisys_int8": "LLAISYS-INT8"}[fw_name]
        for r in results.get(fw_name, []):
            peak = r.get("memory", {}).get("peak_MB", 0)
            print(f"{fw_label:<20} {r['label']:<10} {r['num_input_tokens']:<8} {r['num_output_tokens']:<8} "
                  f"{r['prefill_time_s']*1000:<14.1f} {r['decode_tokens_per_s']:<16.1f} "
                  f"{r['total_tokens_per_s']:<14.1f} {peak:<10.0f}")
    print("=" * 110)


# ============================================================
#  主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLAISYS 端到端推理基准测试 ")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--model", default=None, help="模型路径")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN_SAFE)
    parser.add_argument("--output", default=None, help="JSON 输出路径")
    parser.add_argument("--stage", default=None,
                        choices=["hf", "fp32", "int8"],
                        help="只运行指定阶段 (子进程模式)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        candidates = [
            os.path.join(parent_dir, "models", "DeepSeek-R1-Distill-Qwen-1.5B"),
            os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                model_path = c
                break
        if model_path is None:
            print("错误: 未找到模型, 请通过 --model 指定路径")
            return

    print("=" * 80)
    print(f"LLAISYS 端到端推理基准测试 ")
    print(f"设备: {args.device}  |  GPU: {GPU_SPECS['name']}")
    print(f"模型: {model_path}")
    print(f"KV Cache: max_seq_len={args.max_seq_len}")
    print(f"安全限制: VRAM < {VRAM_LIMIT_MB}MB")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 单阶段子进程模式
    if args.stage:
        results_list = run_single_stage(model_path, args.device, args.max_seq_len,
                                        DEFAULT_PROMPTS, args.stage)
        key_map = {"hf": "hf", "fp32": "llaisys", "int8": "llaisys_int8"}
        results = {"hf": [], "llaisys": [], "llaisys_int8": []}
        results[key_map[args.stage]] = results_list

        output_path = args.output or os.path.join(
            parent_dir, f"_tmp_bench_{args.stage}.json")
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if k != "text"}
            if isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            if isinstance(obj, float):
                return round(obj, 6)
            return obj
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": clean_for_json(results)}, f, indent=2, ensure_ascii=False)
        return

    # 完整测试: 子进程隔离模式
    results = run_inference_benchmark(model_path, args.device, args.max_seq_len)
    print_inference_summary(results)

    output_path = args.output or os.path.join(
        parent_dir, f"benchmark_infer_{args.device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != "text"}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "device": args.device,
                "gpu": GPU_SPECS["name"],
                "model": str(model_path),
                "max_seq_len": args.max_seq_len,
                "timestamp": datetime.now().isoformat(),
            },
            "results": clean_for_json(results),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
