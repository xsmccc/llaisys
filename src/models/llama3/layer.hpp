#pragma once
#include "attention.hpp"
#include "mlp.hpp"
#include "../../ops/add/op.hpp"

namespace llaisys {

class Llama3DecoderLayer {
public:
    Llama3DecoderLayer(const Llama3Config& config)
        : config_(config),
          attn_(config),
          input_norm_(config.rms_norm_eps),
          post_attn_norm_(config.rms_norm_eps) {
        mlp_.init_workspace(config);
        init_workspace();
    }

    // LLaMA3: 无 bias, 简化的 set_params
    void set_params(const LlaisysLlama3Weights* w, size_t layer_idx) {
        attn_.set_params(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx]
        );
        mlp_.set_params(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx]
        );
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    tensor_t forward(tensor_t x, size_t pos, tensor_t pos_tensor) {
        auto residual = x;

        input_norm_.forward(ws_norm1_, x);
        auto attn_out = attn_.forward(ws_norm1_, pos, pos_tensor);
        ops::add(ws_add1_, attn_out, residual);
        residual = ws_add1_;
        post_attn_norm_.forward(ws_norm2_, ws_add1_);
        auto mlp_out = mlp_.forward(ws_norm2_);
        ops::add(ws_add2_, mlp_out, residual);
        return ws_add2_;
    }

private:
    Llama3Config config_;
    Llama3Attention attn_;
    Llama3MLP mlp_;
    RMSNorm input_norm_, post_attn_norm_;
    tensor_t ws_norm1_, ws_norm2_, ws_add1_, ws_add2_;

    void init_workspace() {
        size_t hs = config_.hidden_size;
        ws_norm1_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                   config_.device_type, config_.device_id);
        ws_norm2_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                   config_.device_type, config_.device_id);
        ws_add1_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
        ws_add2_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
    }
};

} // namespace llaisys
