#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_quantized_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_quantized_nvidia.hpp"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_quantized_metax.hpp"
#endif
#ifdef ENABLE_TIANSHU_API
#include "tianshu/linear_quantized_tianshu.hpp"
#endif

/**
 * @file op.cpp
 * @brief W8A32 量化 Linear 算子 — 调度入口
 *
 * 约束:
 *   - out / in / scales / bias 必须为 F32
 *   - weight 必须为 I8
 *   - 所有张量必须在同一设备上且内存连续
 */
namespace llaisys::ops {

void linear_quantized(tensor_t out, tensor_t in, tensor_t weight,
                      tensor_t scales, tensor_t bias) {
    // ── 维度提取 ──
    size_t in_features  = weight->shape()[1]; // K
    size_t out_features = weight->shape()[0]; // N

    // ── 维度检查 ──
    ASSERT(in->shape().back() == in_features,
           "LinearQuantized: input feature dim mismatch");
    ASSERT(out->shape().back() == out_features,
           "LinearQuantized: output feature dim mismatch");
    size_t rows = in->numel() / in_features; // M
    ASSERT(out->numel() / out_features == rows,
           "LinearQuantized: input/output rows mismatch");

    // ── 类型检查 ──
    ASSERT(weight->dtype() == LLAISYS_DTYPE_I8,
           "LinearQuantized: weight must be INT8");
    ASSERT(in->dtype() == LLAISYS_DTYPE_F32,
           "LinearQuantized: input must be F32");
    ASSERT(out->dtype() == LLAISYS_DTYPE_F32,
           "LinearQuantized: output must be F32");
    ASSERT(scales->dtype() == LLAISYS_DTYPE_F32,
           "LinearQuantized: scales must be F32");
    ASSERT(scales->numel() == out_features,
           "LinearQuantized: scales shape mismatch (expected [N])");

    // ── Bias 检查 ──
    bool has_bias = (bias != nullptr) && (bias->numel() > 0);
    if (has_bias) {
        ASSERT(bias->dtype() == LLAISYS_DTYPE_F32,
               "LinearQuantized: bias must be F32");
        ASSERT(bias->numel() == out_features,
               "LinearQuantized: bias shape mismatch");
        CHECK_SAME_DEVICE(out, in, weight, scales, bias);
    } else {
        CHECK_SAME_DEVICE(out, in, weight, scales);
    }

    // ── 连续性检查 ──
    ASSERT(out->isContiguous() && in->isContiguous() &&
           weight->isContiguous() && scales->isContiguous(),
           "LinearQuantized: all inputs must be contiguous");
    if (has_bias) {
        ASSERT(bias->isContiguous(),
               "LinearQuantized: bias must be contiguous");
    }

    // ── 切换设备上下文 ──
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    // ── 调度到设备后端 ──
    switch (in->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            return cpu::linear_quantized(
                out->data(),
                in->data(),
                weight->data(),
                scales->data(),
                has_bias ? bias->data() : nullptr,
                in_features,    // K
                out_features,   // N
                rows            // M
            );
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::linear_quantized(
                out->data(),
                in->data(),
                weight->data(),
                scales->data(),
                has_bias ? bias->data() : nullptr,
                in_features,
                out_features,
                rows
            );
    #endif
    #ifdef ENABLE_METAX_API
        case LLAISYS_DEVICE_METAX:
            return metax::linear_quantized(
                out->data(),
                in->data(),
                weight->data(),
                scales->data(),
                has_bias ? bias->data() : nullptr,
                in_features,
                out_features,
                rows
            );
    #endif
    #ifdef ENABLE_TIANSHU_API
        case LLAISYS_DEVICE_TIANSHU:
            return tianshu::linear_quantized(
                out->data(),
                in->data(),
                weight->data(),
                scales->data(),
                has_bias ? bias->data() : nullptr,
                in_features,
                out_features,
                rows
            );
    #endif
        default:
            ASSERT(false, "Unsupported device type for linear_quantized");
    }
}

} // namespace llaisys::ops
