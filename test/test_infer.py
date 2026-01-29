import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys 
# 导入LLAISYS包
# 这会加载：
# - python/llaisys/__init__.py (主包初始化)
# - python/llaisys/models/__init__.py (models子包)
# - python/llaisys/models/qwen2.py (Qwen2类)
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 加载Pytorch模型
# 从HuggingFace或本地加载Qwen2模型
# tokenizer - 将文本转换为token IDs
# model - 实际的Transformer模型
# torch_dtype=torch.bfloat16 - 使用低精度浮点数（更快）
def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # 模型ID

    if model_path and os.path.isdir(model_path): # 本地加载
        print(f"Loading model from local path: {model_path}")
    else: # 云端加载
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # 分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # bfloat16 浮点类型
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path

# PyTorch推理
def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    # 格式化输入
    # 结果形如："<|User|>Who are you?<|Assistant|><think>\n"
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # 编码成token IDs
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    # 生成新tokens
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    # 输出包含输入tokens + 新生成的tokens

    # 解码回文
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 输出文本内容
    return outputs[0].tolist(), result

# llaisys模型加载
"""
    加载LLAISYS的Qwen2模型
    
    参数：
        model_path: 模型文件夹路径 (包含config.json和model.safetensors)
        device_name: 设备名称 ("cpu" 或 "nvidia")
    
    调用链：
        llaisys.models.Qwen2
        ↓
        python/llaisys/models/qwen2.py 中的 Qwen2 类
        ↓
        执行 Qwen2.__init__(model_path, device)
    
    返回：
        Qwen2 对象，包含 generate() 方法
    """
def load_llaisys_model(model_path, device_name):
    # 调用自己写的Qwen2类
    # 实现Qwen2类
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name)) 
    return model

# llaisys的推理
def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    
    # 调用llaisys的generate函数
    outputs = model.generate(
        inputs,                         # List[int]: token IDs
        max_new_tokens=max_new_tokens,  # 最多生成多少个新tokens
        top_k=top_k,                    # top-k采样参数（在测试时=1，即argmax）
        top_p=top_p,                    # top-p采样参数
        temperature=temperature,        # 温度参数
    )

    # 输出解码
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    
    # 测试模式下使用确定性参数
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    # 运行PyTorch模型
    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    # Example prompt
    start_time = time.time()
    # 推理答案
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps, # argmax采样
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    # 运行llaisys模型
    model = load_llaisys_model(model_path, args.device)
    start_time = time.time()
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    end_time = time.time()

    print("\n=== Your Result ===\n")
    print("Tokens:")
    print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    if args.test:
        assert llaisys_tokens == tokens
        print("\033[92mTest passed!\033[0m\n")
