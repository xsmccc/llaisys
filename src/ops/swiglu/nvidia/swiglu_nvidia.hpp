#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

/**
 * @brief CUDA 实现的 SwiGLU 算子
 * 
 * 计算: out = up * SiLU(gate)
 * 其中 SiLU(x) = x / (1 + exp(-x))
 * 
 * @param out_ptr  输出数据指针
 * @param dtype    数据类型 (F32/F16/BF16)
 * @param gate_ptr gate 输入数据指针
 * @param up_ptr   up 输入数据指针
 * @param numel    元素总数
 */
void swiglu(
    std::byte* out_ptr,
    llaisysDataType_t dtype,
    const std::byte* gate_ptr,
    const std::byte* up_ptr,
    size_t numel
);

} // namespace llaisys::ops::nvidia
