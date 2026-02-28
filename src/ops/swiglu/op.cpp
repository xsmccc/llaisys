#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

// CUDA 实现（条件编译）
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    //设备检查
    CHECK_SAME_DEVICE(out,gate,up);
    //连续性检查
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),"must be contiguous");
    //维度检查
    CHECK_SAME_SHAPE(out->shape(),gate->shape(),up->shape());
    ASSERT(out->shape().size() == 2,"ensure 2D");

    //数据提取
    size_t numel = gate->numel();

    //上下文切换
    llaisys::core::context().setDevice(gate->deviceType(), gate->deviceId());
    //cpu调用
    switch (gate->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),out->dtype(),
            gate->data(),
            up->data(),
            numel
        );
        #ifdef ENABLE_NVIDIA_API
            case LLAISYS_DEVICE_NVIDIA:
                return nvidia::swiglu(
                    out->data(),
                    out->dtype(),
                    gate->data(),
                    up->data(),
                    numel
                );
        #endif
            default:
                EXCEPTION_UNSUPPORTED_DEVICE;
    }
    }
} // namespace llaisys::ops
