#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
//Embedding层实现
/*
weight——shape[Vocab_Size,Hidden_Dim]
index-一串ID，查找对应张量
out-将weight的index对应部分提取出来为新张量
*/

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查设备类型
    CHECK_SAME_DEVICE(out,index,weight);
    // 检查连续性
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding inputs must be contiguous");
    // 检查weight out的数据类型
    CHECK_SAME_DTYPE(out->dtype(),weight->dtype());
    ASSERT(index->dtype() == LLAISYS_DTYPE_I32 || index->dtype() == LLAISYS_DTYPE_I64,"Embedding parameter index data type must be INT32 or INT64");
    // 提取参数
    size_t num_indices = index->numel();//index有多大就要找多少个embedding向量
    size_t vocab_size = weight->shape()[0];//词表维度
    size_t embedding_dim = weight->shape()[1];//隐藏层维度

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),out->dtype(),
            index->data(),num_indices,index->dtype(),
            weight->data(),vocab_size,embedding_dim
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
