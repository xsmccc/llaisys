#pragma once
#include "components.hpp"
#include "qwen2_impl.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys {

class Qwen2MLP {
public:
    Qwen2MLP() = default;

    // 初始化预分配的工作空间
    void init_workspace(const Qwen2Config& config) {
        size_t di = config.intermediate_size;
        size_t hs = config.hidden_size;
        ws_gate_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32, config.device_type, config.device_id);
        ws_up_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32, config.device_type, config.device_id);
        ws_swiglu_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32, config.device_type, config.device_id);
        ws_down_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config.device_type, config.device_id);
    }

    void set_params(void* gate_handle, void* up_handle, void* down_handle) {
        gate_proj_.set_params(gate_handle);
        up_proj_.set_params(up_handle);
        down_proj_.set_params(down_handle);
    }

    tensor_t forward(tensor_t input) {
        // 使用预分配张量
        if (ws_gate_) {
            gate_proj_.forward(ws_gate_, input);
            up_proj_.forward(ws_up_, input);
            ops::swiglu(ws_swiglu_, ws_gate_, ws_up_);
            down_proj_.forward(ws_down_, ws_swiglu_);
            return ws_down_;
        }
        // Fallback: 原始路径
        auto gate_out = gate_proj_.forward(input);
        auto up_out = up_proj_.forward(input);
        auto swiglu_out = Tensor::create(gate_out->shape(), gate_out->dtype(), gate_out->deviceType(), gate_out->deviceId());
        ops::swiglu(swiglu_out, gate_out, up_out);
        return down_proj_.forward(swiglu_out);
    }

private:
    Linear gate_proj_, up_proj_, down_proj_;
    tensor_t ws_gate_, ws_up_, ws_swiglu_, ws_down_;
};

} // namespace llaisys
/*
这个扩展比例是 Transformer 的常见设计：
- 通常是 3-4 倍（BERT/GPT-2 用 4x）
- Qwen2 用 5.38x，给更多的表达能力
- 再压缩回原大小，进行非线性变换
MLP 比例越大，模型表现越好
*/
