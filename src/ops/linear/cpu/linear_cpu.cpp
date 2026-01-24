// 矩阵乘法
/*
X形状为[rows,in_features]——[M,K]
偏移地址：m * in_features + k
W形状为[out_features,in_features]——[N,K]
偏移地址：n * in_features + k
*/
// 。。。算的太慢了
#include "linear_cpu.hpp"
#include "../../../utils.hpp"
template <typename T>
void linear_kernel(
    T* out,
    const T* in,
    const T* weight,
    const T* bias,
    size_t in_features,
    size_t out_features,
    size_t rows
){
    for (size_t m = 0;m < rows;m++)
    {
        for (size_t n = 0;n < out_features;n++){
            float sum = 0.0f;
            const T* in_ptr = in + m * in_features;
            const T* w_ptr  = weight + n * in_features; 
            for (size_t k = 0;k < in_features;k++)
            {
                //进行类型转换，防止精度损失
                float x_val = llaisys::utils::cast<float>(in_ptr[k]);
                float w_val = llaisys::utils::cast<float>(w_ptr[k]);
                sum += x_val * w_val;
            }
            if (bias != nullptr){
                sum += llaisys::utils::cast<float>(bias[n]);
            }

            out[m * out_features + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu{
    void linear(
        std::byte* out,
        llaisysDataType_t dtype,
        const std::byte* in,
        const std::byte* weight,
        const std::byte* bias,
        size_t in_features,
        size_t out_features,
        size_t rows){
        switch (dtype){
            case LLAISYS_DTYPE_F32:
            return linear_kernel(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(bias),
            in_features, out_features, rows
            );
            case LLAISYS_DTYPE_BF16:
            return linear_kernel(
            reinterpret_cast<llaisys::bf16_t*>(out),
            reinterpret_cast<const llaisys::bf16_t*>(in),
            reinterpret_cast<const llaisys::bf16_t*>(weight),
            reinterpret_cast<const llaisys::bf16_t*>(bias),
            in_features, out_features, rows
            );
            case LLAISYS_DTYPE_F16:
            return linear_kernel(
            reinterpret_cast<llaisys::fp16_t*>(out),
            reinterpret_cast<const llaisys::fp16_t*>(in),
            reinterpret_cast<const llaisys::fp16_t*>(weight),
            reinterpret_cast<const llaisys::fp16_t*>(bias),
            in_features, out_features, rows
            );
            default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
}