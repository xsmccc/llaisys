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

/**
 * @file op.cpp
 * @brief W8A16 量化 Linear 算子 — 调度入口
 *
 * 支持 FP16 和 FP32 activation:
 *   - W8A16: in(F16) -> dequant(INT8->F16) -> TC GEMM -> out(F16)
 *   - 兼容 FP32: in(F32) -> convert(F16) -> TC GEMM -> out(F32)
 *   - 混合: in(F16) -> TC GEMM -> out(F32) (如 lm_head 需要 F32 logits)
 *
 * 约束:
 *   - weight 必须为 I8
 *   - in/out 必须为 F32 或 F16
 *   - scales 必须为 F32
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

    // ── 类型检查 (W8A16: 支持 F16 和 F32 activation) ──
    ASSERT(weight->dtype() == LLAISYS_DTYPE_I8,
           "LinearQuantized: weight must be INT8");
    ASSERT(in->dtype() == LLAISYS_DTYPE_F32 || in->dtype() == LLAISYS_DTYPE_F16,
           "LinearQuantized: input must be F32 or F16");
    ASSERT(out->dtype() == LLAISYS_DTYPE_F32 || out->dtype() == LLAISYS_DTYPE_F16,
           "LinearQuantized: output must be F32 or F16");
    ASSERT(scales->dtype() == LLAISYS_DTYPE_F32,
           "LinearQuantized: scales must be F32");
    ASSERT(scales->numel() == out_features,
           "LinearQuantized: scales shape mismatch (expected [N])");

    // ── Bias 检查 ──
    bool has_bias = (bias != nullptr) && (bias->numel() > 0);
    llaisysDataType_t bias_dtype = LLAISYS_DTYPE_F32;
    if (has_bias) {
        ASSERT(bias->dtype() == LLAISYS_DTYPE_F32 || bias->dtype() == LLAISYS_DTYPE_F16,
               "LinearQuantized: bias must be F32 or F16");
        bias_dtype = bias->dtype();
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
                rows,
                in->dtype(),
                out->dtype(),
                bias_dtype
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
        default:
            ASSERT(false, "Unsupported device type for linear_quantized");
    }
}


void linear_quantized_int4(tensor_t out, tensor_t in, tensor_t weight_packed,
                           tensor_t scales, tensor_t bias,
                           size_t group_size, size_t K_orig) {
    // ── 维度提取 ──
    size_t N = weight_packed->shape()[0];       // out_features
    size_t K_packed = weight_packed->shape()[1]; // K_orig / 2
    size_t num_groups = scales->shape()[1];      // K_orig / group_size

    // ── 维度检查 ──
    ASSERT(K_orig == K_packed * 2,
           "LinearQuantizedINT4: K_orig must be 2 * K_packed");
    ASSERT(in->shape().back() == K_orig,
           "LinearQuantizedINT4: input feature dim mismatch");
    ASSERT(out->shape().back() == N,
           "LinearQuantizedINT4: output feature dim mismatch");
    size_t rows = in->numel() / K_orig;
    ASSERT(out->numel() / N == rows,
           "LinearQuantizedINT4: input/output rows mismatch");

    // ── 类型检查 (W4A16: 支持 F16 和 F32 activation) ──
    ASSERT(weight_packed->dtype() == LLAISYS_DTYPE_U8,
           "LinearQuantizedINT4: weight must be U8 (packed INT4)");
    ASSERT(in->dtype() == LLAISYS_DTYPE_F32 || in->dtype() == LLAISYS_DTYPE_F16,
           "LinearQuantizedINT4: input must be F32 or F16");
    ASSERT(out->dtype() == LLAISYS_DTYPE_F32 || out->dtype() == LLAISYS_DTYPE_F16,
           "LinearQuantizedINT4: output must be F32 or F16");
    ASSERT(scales->dtype() == LLAISYS_DTYPE_F16,
           "LinearQuantizedINT4: scales must be F16");
    ASSERT(scales->shape()[0] == N,
           "LinearQuantizedINT4: scales shape[0] must match N");
    ASSERT(num_groups == K_orig / group_size,
           "LinearQuantizedINT4: num_groups mismatch");

    // ── Bias 检查 ──
    bool has_bias = (bias != nullptr) && (bias->numel() > 0);
    llaisysDataType_t bias_dtype = LLAISYS_DTYPE_F32;
    if (has_bias) {
        ASSERT(bias->dtype() == LLAISYS_DTYPE_F32 || bias->dtype() == LLAISYS_DTYPE_F16,
               "LinearQuantizedINT4: bias must be F32 or F16");
        bias_dtype = bias->dtype();
        ASSERT(bias->numel() == N,
               "LinearQuantizedINT4: bias shape mismatch");
        CHECK_SAME_DEVICE(out, in, weight_packed, scales, bias);
    } else {
        CHECK_SAME_DEVICE(out, in, weight_packed, scales);
    }

    // ── 连续性检查 ──
    ASSERT(out->isContiguous() && in->isContiguous() &&
           weight_packed->isContiguous() && scales->isContiguous(),
           "LinearQuantizedINT4: all inputs must be contiguous");

    // ── 设备上下文 ──
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    // ── 调度到设备后端 ──
    switch (in->deviceType()) {
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::linear_quantized_int4(
                out->data(),
                in->data(),
                weight_packed->data(),
                scales->data(),
                has_bias ? bias->data() : nullptr,
                K_orig,         // in_features
                N,              // out_features
                rows,           // M
                num_groups,
                group_size,
                in->dtype(),
                out->dtype(),
                bias_dtype
            );
    #endif
        default:
            ASSERT(false, "LinearQuantizedINT4: only NVIDIA backend supported");
    }
}

} // namespace llaisys::ops
