#!/usr/bin/env python3
"""
简历数据补充 benchmark（subprocess 隔离版本）
"""

import sys, os, time, json, subprocess, argparse, tempfile
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=5
        ).decode().strip().split("\n")[0]
    except Exception:
        return "unknown"


WORKER_SCRIPT = '''
import sys, os, time, json, subprocess
parent_dir = sys.argv[1]
sys.path.insert(0, os.path.join(parent_dir, "python"))
import llaisys
from llaisys import DeviceType

def gpu_mem():
    try:
        return float(subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
            timeout=5).decode().strip().split("\\n")[0])
    except: return -1.0

def ttft(model, toks, warmup=2):
    for _ in range(warmup):
        model.reset()
        for t in model.generate_stream(toks, max_new_tokens=1): break
    times=[]
    for _ in range(3):
        model.reset()
        t0=time.perf_counter()
        for t in model.generate_stream(toks, max_new_tokens=1):
            t1=time.perf_counter(); break
        times.append(t1-t0)
    times.sort()
    return times[1]

def decode_tps(model, toks, max_new=32, warmup=1):
    for _ in range(warmup):
        model.reset()
        model.generate(toks, max_new_tokens=8)
    model.reset()
    out=[]; tf=None; tl=None
    for i,tok in enumerate(model.generate_stream(toks, max_new_tokens=max_new)):
        if i==0: tf=time.perf_counter()
        else: tl=time.perf_counter()
        out.append(tok)
    if tl is None or tf is None or len(out)<2: return 0.0, len(out)
    return (len(out)-1)/(tl-tf), len(out)

cfg = json.loads(sys.argv[2])
path, dev_str, mode = cfg["path"], cfg["device"], cfg["mode"]
dev = DeviceType.NVIDIA if dev_str=="nvidia" else DeviceType.CPU

m0=gpu_mem()
t0=time.perf_counter()
if mode=="int4":
    model=llaisys.models.Qwen2(path, dev, max_seq_len=256, int4=True)
elif mode=="int8":
    model=llaisys.models.Qwen2(path, dev, max_seq_len=256, quantized=True)
else:
    model=llaisys.models.Qwen2(path, dev, max_seq_len=256)
tl=time.perf_counter()-t0
m1=gpu_mem()

p3=[151644,8948,198]
p9=[151644,8948,198,2610,525,264,10950,17847,13]

r={"load_s":round(tl,3), "vram_mb":m1, "vram_delta_mb":round(m1-m0,1)}
r["ttft_3tok_ms"]=round(ttft(model,p3)*1000,2)
r["ttft_9tok_ms"]=round(ttft(model,p9)*1000,2)
tps,n=decode_tps(model,p3,32)
r["decode_tps"]=round(tps,2)
r["n_tokens"]=n
r["vram_peak_mb"]=gpu_mem()
# output on last line only
print("RESULT:"+json.dumps(r))
'''


def run_worker(model_path, device_str, mode):
    cfg = json.dumps({"path": model_path, "device": device_str, "mode": mode})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(parent_dir, "python")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(WORKER_SCRIPT)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp, parent_dir, cfg],
            capture_output=True, text=True, env=env, timeout=300,
        )
        if result.returncode != 0:
            print(f"  [ERROR] {mode}: {result.stderr[-500:]}")
            return None
        for line in reversed(result.stdout.strip().split("\n")):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        print(f"  [ERROR] {mode}: no RESULT line. stdout={result.stdout[-200:]}")
        return None
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia")
    args = parser.parse_args()

    base = os.path.join(parent_dir, "models")
    configs = [
        ("fp32", os.path.join(base, "DeepSeek-R1-Distill-Qwen-1.5B")),
        ("int8", os.path.join(base, "DeepSeek-R1-Distill-Qwen-1.5B-INT8")),
        ("int4", os.path.join(base, "DeepSeek-R1-Distill-Qwen-1.5B-INT4")),
    ]

    all_results = {
        "meta": {"gpu": get_gpu_name(), "ts": datetime.now().isoformat(), "max_seq": 256},
        "results": {}
    }

    for mode, path in configs:
        if not os.path.exists(path):
            print(f"  [{mode}] 路径不存在，跳过")
            continue
        print(f"=== {mode.upper()} ===")
        data = run_worker(path, args.device, mode)
        if data:
            all_results["results"][mode] = data
            print(json.dumps(data, indent=2))
        print()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(parent_dir, f"benchmark_resume_data_{ts}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
