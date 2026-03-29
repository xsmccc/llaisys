#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/fused_add_rmsnorm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/fused_add_rmsnorm_nvidia.hpp"
#endif

namespace llaisys::ops {

void fused_add_rmsnorm(tensor_t out, tensor_t residual_out,
                       tensor_t a, tensor_t b,
                       tensor_t weight, float eps) {
    // 维度检查
    size_t cols = a->shape().back();
    size_t rows = a->numel() / cols;

    ASSERT(out->numel() == a->numel(), "fused_add_rmsnorm: out shape mismatch");
    ASSERT(residual_out->numel() == a->numel(), "fused_add_rmsnorm: residual_out shape mismatch");
    ASSERT(b->numel() == a->numel(), "fused_add_rmsnorm: b shape mismatch");
    ASSERT(weight->numel() == cols, "fused_add_rmsnorm: weight shape mismatch");

    // 类型 / 设备检查
    CHECK_SAME_DEVICE(out, residual_out, a, b, weight);
    CHECK_SAME_DTYPE(out->dtype(), residual_out->dtype(), a->dtype(), b->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && residual_out->isContiguous() &&
           a->isContiguous() && b->isContiguous() && weight->isContiguous(),
           "fused_add_rmsnorm: all tensors must be contiguous");

    llaisys::core::context().setDevice(a->deviceType(), a->deviceId());

    switch (a->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::fused_add_rmsnorm(
            out->data(), residual_out->data(),
            a->data(), b->data(), weight->data(),
            a->dtype(), cols, rows, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::fused_add_rmsnorm(
            out->data(), residual_out->data(),
            a->data(), b->data(), weight->data(),
            a->dtype(), cols, rows, eps);
#endif
    default:
        ASSERT(false, "fused_add_rmsnorm: unsupported device");
    }
}

} // namespace llaisys::ops
