import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..","python"))
import torch, llaisys
from benchmark_inference import (load_hf_model, hf_generate, load_llaisys_model,
                                  llaisys_generate, MemoryTracker, print_inference_summary)
from transformers import AutoTokenizer

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompts = [{"prompt": "Hello", "max_tokens": 16, "label": "short"}]
results = {"hf": [], "llaisys": [], "llaisys_int8": []}
mem = MemoryTracker()

print("[1/3] HuggingFace BF16...")
hf_tok, hf_model = load_hf_model(model_path)
_ = hf_generate(hf_tok, hf_model, "Hi", max_new_tokens=3)
for p in prompts:
    mem.reset()
    r = hf_generate(hf_tok, hf_model, p["prompt"], p["max_tokens"])
    r["memory"] = mem.snapshot(); r["label"] = p["label"]
    results["hf"].append(r)
    print(f"  {p['label']}: {r['num_output_tokens']} tok, prefill={r['prefill_time_s']*1000:.0f}ms, decode={r['decode_tokens_per_s']:.1f} tok/s")
del hf_model; gc.collect(); torch.cuda.empty_cache()

print("[2/3] LLAISYS FP32...")
ll = load_llaisys_model(model_path, max_seq_len=128)
_ = llaisys_generate(tokenizer, ll, "Hi", max_new_tokens=3)
for p in prompts:
    mem.reset()
    r = llaisys_generate(tokenizer, ll, p["prompt"], p["max_tokens"])
    r["memory"] = mem.snapshot(); r["label"] = p["label"]
    results["llaisys"].append(r)
    print(f"  {p['label']}: {r['num_output_tokens']} tok, prefill={r['prefill_time_s']*1000:.0f}ms, decode={r['decode_tokens_per_s']:.1f} tok/s")
del ll; gc.collect(); torch.cuda.empty_cache()

int8_path = str(model_path).rstrip("/") + "-INT8"
if os.path.isdir(int8_path):
    print("[3/3] LLAISYS INT8...")
    ll_q = load_llaisys_model(int8_path, max_seq_len=128, quantized=True)
    _ = llaisys_generate(tokenizer, ll_q, "Hi", max_new_tokens=3)
    for p in prompts:
        mem.reset()
        r = llaisys_generate(tokenizer, ll_q, p["prompt"], p["max_tokens"])
        r["memory"] = mem.snapshot(); r["label"] = p["label"]
        results["llaisys_int8"].append(r)
        print(f"  {p['label']}: {r['num_output_tokens']} tok, prefill={r['prefill_time_s']*1000:.0f}ms, decode={r['decode_tokens_per_s']:.1f} tok/s")
    del ll_q; gc.collect()

print_inference_summary(results)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../benchmark_infer_final.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {out}")
