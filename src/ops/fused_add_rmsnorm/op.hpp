#pragma once
#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/**
 * @brief Fused Add + RMSNorm
 *
 * 将残差连接 (c = a + b) 和 RMSNorm (out = w * c / rms(c)) 融合为单个 kernel。
 * 相比分别调用 add() + rms_norm()，省去 1 次 hidden_size 的 global memory 读写。
 *
 * @param out          [rows, cols] 归一化输出
 * @param residual_out [rows, cols] 残差连接输出 (a + b)，可与 a 或 b 相同（in-place）
 * @param a            [rows, cols] 输入 a
 * @param b            [rows, cols] 输入 b
 * @param weight       [cols] RMSNorm 权重
 * @param eps          RMSNorm epsilon
 */
void fused_add_rmsnorm(tensor_t out, tensor_t residual_out,
                       tensor_t a, tensor_t b,
                       tensor_t weight, float eps);

} // namespace llaisys::ops
