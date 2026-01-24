// src/ops/argmax/cpu/argmax_cpu.cpp
#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <limits> // 用于获取 float 的最小值

// T: 输入值 vals 的类型 (float, half 等)
// IndexT: 输出索引 max_idx 的类型 (int32_t, int64_t)
// Double Dispatch
template <typename T, typename IndexT>
void argmax_kernel(IndexT* max_idx_ptr, T* max_val_ptr, const T* vals, size_t numel) {
    if (numel == 0) return;

    // 初始化第0个是最大的
    size_t best_idx = 0;

    float max_val_f = llaisys::utils::cast<float>(vals[0]); //直接强转

    // 遍历寻找最大值
    for (size_t i = 1; i < numel; i++) {
        float curr_val_f = llaisys::utils::cast<float>(vals[i]);
        if (curr_val_f > max_val_f) {
            max_val_f = curr_val_f;
            best_idx = i;
        }
    }
    *max_val_ptr = llaisys::utils::cast<T>(max_val_f);
    *max_idx_ptr = static_cast<IndexT>(best_idx);
}

// 用于类型强转
template <typename T>
void dispatch_idx_dtype(std::byte* max_idx, llaisysDataType_t idx_dtype,
                        std::byte* max_val, const std::byte* vals, size_t numel) {
    // 强转输入输出指针
    T* max_val_ptr = reinterpret_cast<T*>(max_val);
    const T* vals_ptr = reinterpret_cast<const T*>(vals);

    // 根据索引类型 (Int32 或 Int64) 调用核心 kernel
    if (idx_dtype == LLAISYS_DTYPE_I32) {
        argmax_kernel<T, int32_t>(reinterpret_cast<int32_t*>(max_idx), max_val_ptr, vals_ptr, numel);
    } else if (idx_dtype == LLAISYS_DTYPE_I64) {
        argmax_kernel<T, int64_t>(reinterpret_cast<int64_t*>(max_idx), max_val_ptr, vals_ptr, numel);
    } else {
        // 报错或处理其他情况
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte* max_idx, llaisysDataType_t idx_dtype,
            std::byte* max_val,
            const std::byte* vals, llaisysDataType_t val_dtype,
            size_t numel) {
    // 根据输入值类型 (Float32, BF16, FP16) 分发
    switch (val_dtype) {
    case LLAISYS_DTYPE_F32:
        dispatch_idx_dtype<float>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        dispatch_idx_dtype<llaisys::bf16_t>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    case LLAISYS_DTYPE_F16:
        dispatch_idx_dtype<llaisys::fp16_t>(max_idx, idx_dtype, max_val, vals, numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype);
    }
}
} // namespace llaisys::ops::cpu

