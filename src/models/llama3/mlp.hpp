#pragma once
#include "../qwen2/components.hpp"
#include "llama3_impl.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys {

class Llama3MLP {
public:
    Llama3MLP() = default;

    void init_workspace(const Llama3Config& config) {
        size_t di = config.intermediate_size;
        size_t hs = config.hidden_size;
        ws_gate_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32,
                                  config.device_type, config.device_id);
        ws_up_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32,
                                config.device_type, config.device_id);
        ws_swiglu_ = Tensor::create({1, di}, LLAISYS_DTYPE_F32,
                                    config.device_type, config.device_id);
        ws_down_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                  config.device_type, config.device_id);
    }

    // LLaMA3: 无 bias
    void set_params(void* gate_handle, void* up_handle, void* down_handle) {
        gate_proj_.set_params(gate_handle);
        up_proj_.set_params(up_handle);
        down_proj_.set_params(down_handle);
    }

    tensor_t forward(tensor_t input) {
        gate_proj_.forward(ws_gate_, input);
        up_proj_.forward(ws_up_, input);
        ops::swiglu(ws_swiglu_, ws_gate_, ws_up_);
        down_proj_.forward(ws_down_, ws_swiglu_);
        return ws_down_;
    }

private:
    Linear gate_proj_, up_proj_, down_proj_;
    tensor_t ws_gate_, ws_up_, ws_swiglu_, ws_down_;
};

} // namespace llaisys
