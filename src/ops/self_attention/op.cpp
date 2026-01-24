#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
/*
维度检查
连续性检查
数据类型检查
数据提取
atte_val-[seqlen,nhead,dv]
q-[seqlen,nhead,d]
k-[total_len,nkvhead.d]
v-[total_len,nkvhead,dv]
scale
*/
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    //假设张量是连续的，先进行连续性判断
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),"Tensor shoule be contiguous");

    //首先检查是不是3D-Tensor
    ASSERT(q->shape().size() == 3, "Q shape must be 3D");
    ASSERT(k->shape().size() == 3, "K shape must be 3D");
    ASSERT(v->shape().size() == 3, "V shape must be 3D");
    ASSERT(attn_val->shape().size() == 3, "Attn_Val shape must be 3D");

    //数据提取
    size_t seq_len   = q->shape()[0];
    size_t nhead     = q->shape()[1];
    size_t head_dim  = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t kv_head   = k->shape()[1];
    size_t v_head_dim= v->shape()[2];

    //维度检查
    ASSERT(seq_len == attn_val->shape()[0],"attn_val seq_len mismatch");
    ASSERT(nhead == attn_val->shape()[1],"attn_val nhead mismatch");
    ASSERT(total_len == v->shape()[0],"K/V total_len mismatch");
    ASSERT(head_dim == k->shape()[2],"K head_dim mismatch");
    ASSERT(kv_head == v->shape()[1], "K/V head mismatch");
    //GQA检查
    ASSERT(nhead % kv_head == 0,"nhead must be divisible by kv_head (GQA)");
    //上下文切换
    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());

    //cpu调用
    switch (q->deviceType()){
        case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),attn_val->dtype(),
            q->data(),
            k->data(),
            v->data(),
            seq_len,total_len,
            nhead,kv_head,
            head_dim,v_head_dim,
            scale
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
