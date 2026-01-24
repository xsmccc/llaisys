#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    //设备一致性检查
    CHECK_SAME_DEVICE(out,in,pos_ids);

    //连续性检查
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(),"RoPE paremeter must be contiguous");

    //先检查输入维度
    ASSERT(in->shape().size() == 3,"input shape must be 3");//输入维度为3,对应[seqlen,nhead,d]
    CHECK_SAME_SHAPE(in->shape(),out->shape());

    //数据提取
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    //逻辑检查
    ASSERT(pos_ids->numel() == seq_len,"pos_ids length must match Seq_Len");
    ASSERT(head_dim % 2 == 0,"Head dimension must be even for RoPE");
    
    //数据类型检查
    CHECK_SAME_DTYPE(in->dtype(),out->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,"token id data type must be INT64");
    //上下文切换
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    //cpu调用
    switch (in->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),out->dtype(),
            in->data(),  
            pos_ids->data(), 
            seq_len,n_heads,head_dim,
            theta
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
需要的参数

*/