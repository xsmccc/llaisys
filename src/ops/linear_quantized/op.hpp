#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/**
 * @brief W8A32 量化 Linear 算子
 *
 * 公式: out(F32) = dequant(weight_int8, scales) @ in(F32)^T + bias(F32)
 *
 * 权重为 INT8 per-channel 量化（absmax 对称量化）:
 *   scale[n] = max(|W_fp32[n, :]|) / 127.0
 *   W_int8[n, k] = round(W_fp32[n, k] / scale[n])
 *   还原: W_fp32[n, k] ≈ W_int8[n, k] * scale[n]
 *
 * @param out       输出张量 [M, N], F32
 * @param in        输入激活 [M, K], F32
 * @param weight    INT8 量化权重 [N, K], I8
 * @param scales    per-channel 缩放因子 [N], F32
 * @param bias      偏置 [N], F32 (可选, nullptr 表示无偏置)
 */
void linear_quantized(tensor_t out, tensor_t in, tensor_t weight,
                      tensor_t scales, tensor_t bias);


/**
 * @brief W4A32 量化 Linear 算子 (INT4 group quantization)
 *
 * 公式: out(F32) = dequant(weight_int4_packed, scales_group) @ in(F32)^T + bias(F32)
 *
 * INT4 per-group 对称量化 (absmax):
 *   group_size = 128, num_groups = K / group_size
 *   scale[n, g] = max(|W_fp32[n, g*gs:(g+1)*gs]|) / 7.0
 *   W_int4[n, k] = round(W_fp32[n, k] / scale[n, k/gs])  ∈ [-8, 7]
 *   Pack: 2×int4 → 1×uint8 (low nibble + high nibble)
 *
 * @param out             输出张量 [M, N], F32
 * @param in              输入激活 [M, K_orig], F32
 * @param weight_packed   INT4 packed 权重 [N, K_orig/2], U8
 * @param scales          group-wise 缩放因子 [N, num_groups], F16
 * @param bias            偏置 [N], F32 (可选)
 * @param group_size      量化组大小 (128)
 * @param K_orig          原始 input features (= K_packed * 2)
 */
void linear_quantized_int4(tensor_t out, tensor_t in, tensor_t weight_packed,
                           tensor_t scales, tensor_t bias,
                           size_t group_size, size_t K_orig);

} // namespace llaisys::ops

