#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    //设备类型检查
    CHECK_SAME_DEVICE(out,in,weight);
    //维度检查
    size_t last_dim = in->shape().back();
    ASSERT(weight->shape().size() == 1,"RMSNorm weight must be 1D");
    ASSERT(weight->shape()[0] == last_dim,"RMSNorm weight shape mismatch");   
    //数据类型检查
    CHECK_SAME_DTYPE(out->dtype(),in->dtype(),weight->dtype());
    //连续性检查
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),"RMSNorm input must be contiguous");
    //数据提取
    size_t cols = last_dim;
    size_t rows = in->numel() / cols;
    //切换上下文
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    //cpu调用
    switch (in->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),out->dtype(),
            in->data(),  
            weight->data(), 
            cols,rows,eps
        );
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            TO_BE_IMPLEMENTED();
            return;
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops

/*

*/