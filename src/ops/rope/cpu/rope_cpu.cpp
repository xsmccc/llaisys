#include "rope_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
//float32精度不够...
template <typename T>
void rope_kernel(
    T* out,
    const T* in,
    const int64_t* pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta
){
    size_t half_dim = head_dim / 2;
    
    //遍历seq
    for (size_t i = 0;i < seq_len;i++)
    {
        int64_t p = pos_ids[i];

        //遍历head
        for (size_t h=0;h < n_heads;h++)
        {
            size_t offset = i * n_heads * head_dim + h * head_dim;

            for (size_t j = 0;j < half_dim;j++)
            {
                //计算角度
                double freq = 1.0f / std::pow(static_cast<double>(theta),2.0f * j / head_dim);
                double angle = static_cast<double>(p) * freq;
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);
                
                //分组
                double a = static_cast<double>(llaisys::utils::cast<float>(in[offset + j]));
                double b = static_cast<double>(llaisys::utils::cast<float>(in[offset + j + half_dim]));
                //旋转公式
                double out_a = a * cos_val - b * sin_val;
                double out_b = b * cos_val + a * sin_val;

                out[offset + j] = llaisys::utils::cast<T>(out_a);
                out[offset + j + half_dim] = llaisys::utils::cast<T>(out_b);

            }
        }
    }
}

namespace llaisys::ops::cpu{
    void rope(
        std::byte* out_ptr,
        llaisysDataType_t dtype,
        const std::byte* in_ptr,
        const std::byte* pos_ids,
        size_t seq_len,
        size_t n_heads,
        size_t head_dim,
        float theta
    ){
        const int64_t* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_ids);
        switch(dtype){
            case LLAISYS_DTYPE_F32:
            return rope_kernel(
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<const float*>(in_ptr),
            pos_ids_ptr,
            seq_len,
            n_heads,
            head_dim,
            theta
            );
            case LLAISYS_DTYPE_BF16:
            return rope_kernel(
            reinterpret_cast<llaisys::bf16_t*>(out_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(in_ptr),
            pos_ids_ptr,
            seq_len,
            n_heads,
            head_dim,
            theta
            );
            case LLAISYS_DTYPE_F16:
            return rope_kernel(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(in_ptr),
            pos_ids_ptr,
            seq_len,
            n_heads,
            head_dim,
            theta
            );
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }    
    }
}