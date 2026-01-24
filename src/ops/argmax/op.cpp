#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp" // 引用 CPU 实现头文件

//传入max_idx max_val是对象，然后将结果传入参数接口
// max_idx：输出索引 
// max_val：输出最大值
// vals：输入数据
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx,max_val,vals); //首先要检查设备类型是否正确，CPU还是GPU

    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());//随后保证max_val和vals的数据类型是一样的。
    //确保连续性，不连续会导致内部裸指针不能安全遍历
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: input tensor must be contiguous.");
    // 索引必须是整形
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I32 || max_idx->dtype() == LLAISYS_DTYPE_I64,"Argmax: max_idx must be INT32 or INT64");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(), max_idx->dtype(),
            max_val->data(),
            vals->data(), vals->dtype(), vals->numel()
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
