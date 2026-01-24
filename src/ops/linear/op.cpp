#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

//默认输入为2D张量
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    //参数提取
    size_t in_features  = weight->shape()[1]; // 输入特征维度
    size_t out_features = weight->shape()[0]; //输出特征维度

    // 维度检查
    ASSERT(in->shape().back() == in_features, "Linear: input feature dim mismatch");
    // 检查输出的最后一维是否等于权重的输出维度
    ASSERT(out->shape().back() == out_features, "Linear: output feature dim mismatch");
    // 检查总行数是否匹配
    size_t rows = in->numel() / in_features;
    ASSERT(out->numel() / out_features == rows, "Linear: input/output rows mismatch");

    //是否有bias
    bool has_bias = (bias != nullptr) && (bias->numel() > 0);

    //根据是否有bias来检查对应的设备，参数类型等
    if (has_bias){
       CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        // Bias 必须是 1D 且长度等于 Out_Features
        ASSERT(bias->numel() == out_features, "Linear: bias shape mismatch");
    }else{
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(),);
    }

    //连续性检查
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), 
           "Linear inputs must be contiguous");
    if (has_bias) {
        ASSERT(bias->isContiguous(), "Linear bias must be contiguous");
    }

    //切换上下文
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    //调用CPU
    switch (in->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),out->dtype(),
            in->data(),  
            weight->data(), 
            has_bias ? bias->data() : nullptr,//判断是否有bias
            in_features,    // K
            out_features,   // N
            rows            // M
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
