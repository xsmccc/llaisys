#pragma once
#include "components.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys {

// 前向传播网络层
class Qwen2MLP {
public:
    Qwen2MLP() = default;

    void set_params(void* gate_handle, void* up_handle, void* down_handle) {
        gate_proj_.set_params(gate_handle);
        up_proj_.set_params(up_handle);
        down_proj_.set_params(down_handle);
    }

    tensor_t forward(tensor_t input) {
        // Gate投影 多少信息应该通过
        auto gate_out = gate_proj_.forward(input);

        //Up投影  什么信息应该通过
        auto up_out = up_proj_.forward(input);
        
        // SwiGLU 激活（Gate × Up）
        auto swiglu_out = Tensor::create(
            gate_out->shape(),
            gate_out->dtype(),
            gate_out->deviceType(),
            gate_out->deviceId()
        );

        ops::swiglu(swiglu_out, gate_out, up_out);

        // Down 投影（投影回隐层维度）
        auto output = down_proj_.forward(swiglu_out);

        return output;
    }

private:
    Linear gate_proj_;   // 第1个投影：门控投影
    Linear up_proj_;     // 第2个投影：上投影
    Linear down_proj_;   // 第3个投影：下投影
};

} // namespace llaisys
/*
这个扩展比例是 Transformer 的常见设计：
- 通常是 3-4 倍（BERT/GPT-2 用 4x）
- Qwen2 用 5.38x，给更多的表达能力
- 再压缩回原大小，进行非线性变换
MLP 比例越大，模型表现越好
*/
