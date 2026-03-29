#pragma once
#include "components.hpp"
#include "qwen2_impl.hpp"
#include "../../ops/swiglu/op.hpp"

namespace llaisys {

class Qwen2MLP {
public:
    Qwen2MLP() = default;

    void init_workspace(const Qwen2Config& config) {
        config_ = &config;
        alloc_workspace(1);
    }

    void prepare_for_seq_len(size_t seq_len) {
        if (seq_len == current_seq_len_) return;
        alloc_workspace(seq_len);
    }

    void set_params(void* gate_handle, void* up_handle, void* down_handle) {
        gate_proj_.set_params(gate_handle);
        up_proj_.set_params(up_handle);
        down_proj_.set_params(down_handle);
    }

    void set_params_quantized(void* gate_handle, void* up_handle, void* down_handle,
                              void* gate_scales, void* up_scales, void* down_scales) {
        gate_proj_.set_params_quantized(gate_handle, gate_scales);
        up_proj_.set_params_quantized(up_handle, up_scales);
        down_proj_.set_params_quantized(down_handle, down_scales);
    }

    void set_params_int4(void* gate_handle, void* up_handle, void* down_handle,
                         void* gate_scales, void* up_scales, void* down_scales,
                         size_t gs, size_t gate_K, size_t up_K, size_t down_K) {
        gate_proj_.set_params_int4(gate_handle, gate_scales, gs, gate_K);
        up_proj_.set_params_int4(up_handle, up_scales, gs, up_K);
        down_proj_.set_params_int4(down_handle, down_scales, gs, down_K);
    }

    tensor_t forward(tensor_t input) {
        if (ws_gate_) {
            gate_proj_.forward(ws_gate_, input);
            up_proj_.forward(ws_up_, input);
            ops::swiglu(ws_swiglu_, ws_gate_, ws_up_);
            down_proj_.forward(ws_down_, ws_swiglu_);
            return ws_down_;
        }
        auto gate_out = gate_proj_.forward(input);
        auto up_out = up_proj_.forward(input);
        auto swiglu_out = Tensor::create(gate_out->shape(), gate_out->dtype(),
            gate_out->deviceType(), gate_out->deviceId());
        ops::swiglu(swiglu_out, gate_out, up_out);
        return down_proj_.forward(swiglu_out);
    }

private:
    const Qwen2Config* config_ = nullptr;
    size_t current_seq_len_ = 0;
    Linear gate_proj_, up_proj_, down_proj_;
    tensor_t ws_gate_, ws_up_, ws_swiglu_, ws_down_;

    void alloc_workspace(size_t seq_len) {
        current_seq_len_ = seq_len;
        size_t di = config_->intermediate_size;
        size_t hs = config_->hidden_size;
        auto dt = config_->compute_dtype;
        ws_gate_ = Tensor::create({seq_len, di}, dt, config_->device_type, config_->device_id);
        ws_up_ = Tensor::create({seq_len, di}, dt, config_->device_type, config_->device_id);
        ws_swiglu_ = Tensor::create({seq_len, di}, dt, config_->device_type, config_->device_id);
        ws_down_ = Tensor::create({seq_len, hs}, dt, config_->device_type, config_->device_id);
    }
};

} // namespace llaisys
