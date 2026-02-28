#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <omp.h>  // OpenMP 头文件

template <typename T>
void rms_norm_kernel(
    T* out,
    const T* in,
    const T* weight,
    size_t rows,
    size_t cols,
    float eps

){
    // OpenMP 并行化：每个线程处理不同的行
    // 行之间没有数据依赖，可以安全并行
    #pragma omp parallel for schedule(static)
    for (size_t i = 0;i<rows;i++)
    {
        const T* in_ptr = in + i * cols;
        T* out_ptr = out + i * cols;
        float sum_seq = 0.0f;
        // 内层循环计算 RMS（单线程执行，因为需要累加） 后续可进行SIMD优化
        for (size_t j = 0;j < cols;j++)
        {
            float val = llaisys::utils::cast<float>(in_ptr[j]);
            sum_seq += val * val;
        }
        float rms = std::sqrt(sum_seq / cols + eps);
        float inv_rms = 1.0f / rms;

        // 归一化和缩放（单线程执行当前行） 后续可进行SIMD优化
        for (size_t j = 0;j < cols;j++)
        {
            float val = llaisys::utils::cast<float>(in_ptr[j]);
            float w   = llaisys::utils::cast<float>(weight[j]);
            out_ptr[j] =  llaisys::utils::cast<T>(val * w * inv_rms);
        }
    }
}

namespace llaisys::ops::cpu{
    void rms_norm(
        std::byte* out_ptr,
        llaisysDataType_t dtype,
        const std::byte* in_ptr,
        const std::byte* weight_ptr,
        size_t cols,
        size_t rows,
        float eps
    ){
        switch(dtype){
            case LLAISYS_DTYPE_F32:
            return rms_norm_kernel(
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<const float*>(in_ptr),
            reinterpret_cast<const float*>(weight_ptr),
            rows,
            cols,
            eps
            );
            case LLAISYS_DTYPE_BF16:
            return rms_norm_kernel(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(in_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(weight_ptr),
            rows,
            cols,
            eps
            );
            case LLAISYS_DTYPE_F16:
            return rms_norm_kernel(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(in_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(weight_ptr),
            rows,
            cols,
            eps
            );
            default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
}