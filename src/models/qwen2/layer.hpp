#pragma once
#include "attention.hpp"
#include "mlp.hpp"
#include "../../ops/add/op.hpp"

namespace llaisys {

// decoder解码器层
/*
x → Norm → Attention → Add(残差) → Norm → MLP → Add(残差) → output
Pre-Norm 优势：
  - 更稳定，梯度流动更好
  - 无需 warmup
  - 特别是在深网络中表现更好
*/
class Qwen2DecoderLayer {
public:
    Qwen2DecoderLayer(const Qwen2Config& config)
        : attn_(config),
          input_norm_(config.rms_norm_eps),
          post_attn_norm_(config.rms_norm_eps) {}

    void set_params(const LlaisysQwen2Weights* w, size_t layer_idx) {
        // Set attention parameters
        attn_.set_params(
            w->attn_q_w[layer_idx],   // Q投影权重 [hidden_size, hidden_size]
            w->attn_k_w[layer_idx],   // K投影权重 [hidden_size, hidden_size]
            w->attn_v_w[layer_idx],   // V投影权重 [hidden_size, hidden_size]
            w->attn_o_w[layer_idx],   // Output投影权重 [hidden_size, hidden_size]
            w->attn_q_b[layer_idx],   // Q偏置 [hidden_size]
            w->attn_k_b[layer_idx],   // K偏置 [hidden_size]
            w->attn_v_b[layer_idx]    // V偏置 [hidden_size]
        );

        // Set MLP parameters
        mlp_.set_params(
            w->mlp_gate_w[layer_idx],  // Gate投影权重 [intermediate_size, hidden_size]
            w->mlp_up_w[layer_idx],    // Up投影权重 [intermediate_size, hidden_size]
            w->mlp_down_w[layer_idx]   // Down投影权重 [hidden_size, intermediate_size]
        );

        // 设置层归一化权重
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    tensor_t forward(tensor_t x, size_t pos) {
        // 保存输入用于残差连接
        auto residual = x;
        // Layer Norm（归一化）
        auto norm_x = input_norm_.forward(x);
        // 多头自注意力
        auto attn_out = attn_.forward(norm_x, pos);
        
        // 残差连接：Add
        // 残差连接的目的：
        //   1. 梯度直通（backprop 更有效）
        //   2. 信息保留（attention不能摧毁原始信息）
        //   3. 网络更深时尤其重要
        auto add_out = Tensor::create(
            attn_out->shape(),
            attn_out->dtype(),
            attn_out->deviceType(),
            attn_out->deviceId()
        );
        ops::add(add_out, attn_out, residual);

        // 保存注意力块的输出用于残差
        residual = add_out;
        // MLP前Layer norm（RMSNorm）
        norm_x = post_attn_norm_.forward(add_out);
        // 前向传播网络（MLP）
        auto mlp_out = mlp_.forward(norm_x);
        // MLP 后的残差连接
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
    Qwen2Attention attn_;   //多头自注意力
    Qwen2MLP mlp_;          //前向传播网络
    RMSNorm input_norm_;    //注意力前的层归一化
    RMSNorm post_attn_norm_;//MLP前的层归一化
};

} // namespace llaisys
