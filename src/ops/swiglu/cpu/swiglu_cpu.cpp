#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

template<typename T>
void swiglu_kernel(
    T* out_ptr,
    const T* gate_ptr,
    const T* up_ptr,
    size_t numel
){
    for (size_t i = 0;i < numel;i++)
    {
        float g = llaisys::utils::cast<float>(gate_ptr[i]);
        float u = llaisys::utils::cast<float>(up_ptr[i]);

        float silu_val = g / (1.0 + std::exp(-g));
        float res  = u * silu_val;
        out_ptr[i] = llaisys::utils::cast<T>(res);
    }
}

namespace llaisys::ops::cpu {
    void swiglu(
        std::byte* out_ptr, llaisysDataType_t dtype,
        const std::byte* gate_ptr, const std::byte* up_ptr,
        size_t numel
    ) {
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return swiglu_kernel(
                reinterpret_cast<float*>(out_ptr),
                reinterpret_cast<const float*>(gate_ptr),
                reinterpret_cast<const float*>(up_ptr),
                numel
            );
        case LLAISYS_DTYPE_BF16:
            return swiglu_kernel(
                reinterpret_cast<llaisys::bf16_t*>(out_ptr),
                reinterpret_cast<const llaisys::bf16_t*>(gate_ptr),
                reinterpret_cast<const llaisys::bf16_t*>(up_ptr),
                numel
            );
        case LLAISYS_DTYPE_F16:
        return swiglu_kernel(
            reinterpret_cast<llaisys::fp16_t*>(out_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(gate_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(up_ptr),
            numel
        );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }
}