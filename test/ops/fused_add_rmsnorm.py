"""
Fused Add + RMSNorm 算子正确性测试

验证 fused_add_rmsnorm(out, res, a, b, w, eps) 等价于:
  res = a + b
  out = RMSNorm(res, w, eps)

用法:
    source venv/bin/activate
    PYTHONPATH=python:test python3 test/ops/fused_add_rmsnorm.py --device nvidia
"""
import sys
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import random_tensor, check_equal


def torch_fused_add_rmsnorm(a, b, w, eps):
    """Reference: add + rmsnorm in PyTorch"""
    residual = a + b
    rms = torch.sqrt(torch.mean(residual ** 2, dim=-1, keepdim=True) + eps)
    out = w * residual / rms
    return out, residual


def test_fused_add_rmsnorm(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
):
    print(f"   shape {shape} dtype <{dtype_name}>")

    a_torch, a_ll = random_tensor(shape, dtype_name, device_name)
    b_torch, b_ll = random_tensor(shape, dtype_name, device_name)
    w_torch, w_ll = random_tensor((shape[-1],), dtype_name, device_name)
    eps = 1e-5

    # PyTorch reference
    ref_out, ref_res = torch_fused_add_rmsnorm(a_torch, b_torch, w_torch, eps)

    # LLAISYS output buffers
    _, out_ll = random_tensor(shape, dtype_name, device_name)
    _, res_ll = random_tensor(shape, dtype_name, device_name)

    llaisys.Ops.fused_add_rms_norm(out_ll, res_ll, a_ll, b_ll, w_ll, eps)

    assert check_equal(out_ll, ref_out, atol=atol, rtol=rtol), \
        f"fused_add_rmsnorm output mismatch (atol={atol})"
    assert check_equal(res_ll, ref_res, atol=atol, rtol=rtol), \
        f"fused_add_rmsnorm residual mismatch (atol={atol})"

    print(f"   ✅ PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"])
    args = parser.parse_args()

    device = args.device
    print(f"=== Fused Add + RMSNorm Test ({device}) ===\n")

    print("F32:")
    for shape in [(1, 1536), (4, 1536), (1, 4096), (16, 1536)]:
        test_fused_add_rmsnorm(shape, "f32", atol=1e-5, rtol=1e-5, device_name=device)

    if device == "nvidia":
        print("\nF16:")
        for shape in [(1, 1536), (4, 1536)]:
            test_fused_add_rmsnorm(shape, "f16", atol=1e-2, rtol=1e-2, device_name=device)

        print("\nBF16:")
        for shape in [(1, 1536), (4, 1536)]:
            test_fused_add_rmsnorm(shape, "bf16", atol=1e-2, rtol=1e-2, device_name=device)

    print("\n✅ All fused_add_rmsnorm tests passed!")
