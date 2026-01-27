#pragma once
#include "attention.hpp"
#include "mlp.hpp"
#include "../../ops/add/op.hpp"

namespace llaisys {

class Qwen2DecoderLayer {
public:
    Qwen2DecoderLayer(const Qwen2Config& config)
        : attn_(config),
          input_norm_(config.rms_norm_eps),
          post_attn_norm_(config.rms_norm_eps) {}

    void set_params(const LlaisysQwen2Weights* w, size_t layer_idx) {
        // Set attention parameters
        attn_.set_params(
            w->attn_q_w[layer_idx],
            w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx],
            w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx],
            w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx]
        );

        // Set MLP parameters
        mlp_.set_params(
            w->mlp_gate_w[layer_idx],
            w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx]
        );

        // Set norm parameters
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    tensor_t forward(tensor_t x, size_t pos) {
        // Attention block with residual
        auto residual = x;
        auto norm_x = input_norm_.forward(x);
        auto attn_out = attn_.forward(norm_x, pos);
        
        auto add_out = Tensor::create(
            attn_out->shape(),
            attn_out->dtype(),
            attn_out->deviceType(),
            attn_out->deviceId()
        );
        ops::add(add_out, attn_out, residual);

        // MLP block with residual
        residual = add_out;
        norm_x = post_attn_norm_.forward(add_out);
        auto mlp_out = mlp_.forward(norm_x);

        auto final_out = Tensor::create(
            mlp_out->shape(),
            mlp_out->dtype(),
            mlp_out->deviceType(),
            mlp_out->deviceId()
        );
        ops::add(final_out, mlp_out, residual);

        return final_out;
    }

private:
    Qwen2Attention attn_;
    Qwen2MLP mlp_;
    RMSNorm input_norm_;
    RMSNorm post_attn_norm_;
};

} // namespace llaisys
