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

} // namespace llaisys::ops

