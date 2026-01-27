#pragma once
#include "components.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys {

class Qwen2MLP {
public:
    Qwen2MLP() = default;

    void set_params(void* gate_handle, void* up_handle, void* down_handle) {
        gate_proj_.set_params(gate_handle);
        up_proj_.set_params(up_handle);
        down_proj_.set_params(down_handle);
    }

    tensor_t forward(tensor_t input) {
        auto gate_out = gate_proj_.forward(input);
        auto up_out = up_proj_.forward(input);

        auto swiglu_out = Tensor::create(
            gate_out->shape(),
            gate_out->dtype(),
            gate_out->deviceType(),
            gate_out->deviceId()
        );

        ops::swiglu(swiglu_out, gate_out, up_out);
        auto output = down_proj_.forward(swiglu_out);

        return output;
    }

private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
};

} // namespace llaisys
