"""Test for INT8 W8A32 quantized linear operator.

Tests that: out = dequant(W_int8, scales) @ x + bias
matches the PyTorch reference: out = (W_int8.float() * scales.unsqueeze(1)) @ x + bias
"""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
import numpy as np
from test_utils import check_equal


def make_llaisys_tensor(torch_tensor, device_name, device_id=0):
    """Copy a torch tensor into a llaisys tensor."""
    dt_map = {
        torch.float32: llaisys.DataType.F32,
        torch.int8: llaisys.DataType.I8,
    }
    dev_map = {
        "nvidia": llaisys.DeviceType.NVIDIA,
        "cpu": llaisys.DeviceType.CPU,
    }
    lt = llaisys.Tensor(
        tuple(torch_tensor.shape),
        dtype=dt_map[torch_tensor.dtype],
        device=dev_map[device_name],
        device_id=device_id,
    )
    api = llaisys.RuntimeAPI(dev_map[device_name])
    nbytes = torch_tensor.numel() * torch_tensor.element_size()
    api.memcpy_sync(lt.data_ptr(), torch_tensor.data_ptr(), nbytes, llaisys.MemcpyKind.D2D)
    return lt


def torch_linear_quantized_ref(x, w_int8, scales, bias):
    """Reference: dequant weight then matmul."""
    w_f32 = w_int8.float() * scales.unsqueeze(1)   # [N, K]
    out = torch.nn.functional.linear(x, w_f32, bias)
    return out


def test_linear_quantized(M, N, K, use_bias, device_name):
    print(f"  M={M}, N={N}, K={K}, bias={use_bias}, device={device_name}")
    dev = "cuda:0" if device_name == "nvidia" else "cpu"

    # Random FP32 input
    x = torch.randn(M, K, device=dev)

    # Simulate quantized weight: random INT8 in [-64, 63]
    w_int8 = torch.randint(-64, 64, (N, K), dtype=torch.int8, device=dev)

    # Per-channel scales
    scales = torch.rand(N, device=dev) * 0.01 + 0.001  # small positive scales

    bias = torch.randn(N, device=dev) if use_bias else None

    # Reference
    ref_out = torch_linear_quantized_ref(x, w_int8, scales, bias)

    # llaisys tensors
    x_ll = make_llaisys_tensor(x, device_name)
    w_ll = make_llaisys_tensor(w_int8, device_name)
    s_ll = make_llaisys_tensor(scales, device_name)

    # Output tensor
    out_ll = make_llaisys_tensor(torch.zeros(M, N, device=dev), device_name)

    # Bias tensor (None → nullptr if no bias)
    if use_bias:
        b_ll = make_llaisys_tensor(bias, device_name)
    else:
        b_ll = None

    llaisys.Ops.linear_quantized(out_ll, x_ll, w_ll, s_ll, b_ll)

    # INT8→FP16 dequant + FP16 TC GEMM means less precision than pure FP32
    assert check_equal(out_ll, ref_out, atol=5e-2, rtol=1e-2), \
        f"FAILED: M={M}, N={N}, K={K}, bias={use_bias}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia",
                        choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    print(f"Testing Ops.linear_quantized (INT8 W8A32) on {args.device}")

    test_cases = [
        # (M, N, K, use_bias)
        (1, 2048, 4096, True),     # decode shape (M=1)
        (1, 2048, 4096, False),    # decode without bias
        (32, 4096, 4096, True),    # prefill batch
        (2, 3, 4, True),           # small
        (7, 11, 13, False),        # non-power-of-2
    ]
    for M, N, K, bias in test_cases:
        llaisys.Ops.cleanup_quantized_weight_cache()
        test_linear_quantized(M, N, K, bias, args.device)

    print("\033[92mTest passed!\033[0m\n")
