#pragma once
#include "attention.hpp"
#include "mlp.hpp"
#include "../../ops/add/op.hpp"

namespace llaisys {

class Qwen2DecoderLayer {
public:
    Qwen2DecoderLayer(const Qwen2Config& config)
        : config_(config),
          attn_(config),
          input_norm_(config.rms_norm_eps),
          post_attn_norm_(config.rms_norm_eps) {
        // 初始化 MLP 工作空间
        mlp_.init_workspace(config);
        // 预分配 DecoderLayer 的工作空间张量
        init_workspace();
    }

    void set_params(const LlaisysQwen2Weights* w, size_t layer_idx) {
        attn_.set_params(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx]
        );
        mlp_.set_params(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx]
        );
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    // 量化版本: 权重为 INT8 + per-channel scales
    void set_params_quantized(const LlaisysQwen2Weights* w, size_t layer_idx) {
        attn_.set_params_quantized(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx],
            w->attn_q_w_scales[layer_idx], w->attn_k_w_scales[layer_idx],
            w->attn_v_w_scales[layer_idx], w->attn_o_w_scales[layer_idx]
        );
        mlp_.set_params_quantized(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx],
            w->mlp_gate_w_scales[layer_idx], w->mlp_up_w_scales[layer_idx],
            w->mlp_down_w_scales[layer_idx]
        );
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    // INT4 量化版本: packed U8 权重 + group F16 scales
    void set_params_int4(const LlaisysQwen2Weights* w, size_t layer_idx) {
        size_t gs = w->int4_group_size;
        // 从 int4_K_orig 数组获取 K_orig
        // layout: [nlayer*7+1], 层内顺序: q,k,v,o, gate,up,down
        size_t base = layer_idx * 7;
        size_t q_K = w->int4_K_orig[base + 0];
        size_t k_K = w->int4_K_orig[base + 1];
        size_t v_K = w->int4_K_orig[base + 2];
        size_t o_K = w->int4_K_orig[base + 3];
        size_t gate_K = w->int4_K_orig[base + 4];
        size_t up_K = w->int4_K_orig[base + 5];
        size_t down_K = w->int4_K_orig[base + 6];

        attn_.set_params_int4(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx],
            w->attn_q_w_scales[layer_idx], w->attn_k_w_scales[layer_idx],
            w->attn_v_w_scales[layer_idx], w->attn_o_w_scales[layer_idx],
            gs, q_K, k_K, v_K, o_K
        );
        mlp_.set_params_int4(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx],
            w->mlp_gate_w_scales[layer_idx], w->mlp_up_w_scales[layer_idx],
            w->mlp_down_w_scales[layer_idx],
            gs, gate_K, up_K, down_K
        );
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    // 优化版：使用预分配张量 + 外部 pos_tensor
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

    // 兼容旧接口
    tensor_t forward(tensor_t x, size_t pos) {
        std::vector<size_t> ps = {1};
        std::vector<int64_t> ph = {static_cast<int64_t>(pos)};
        auto pt = Tensor::create(ps, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
        pt->load(ph.data());
        return forward(x, pos, pt);
    }

private:
    Qwen2Config config_;
    Qwen2Attention attn_;
    Qwen2MLP mlp_;
    RMSNorm input_norm_, post_attn_norm_;
    // 预分配的工作空间
    tensor_t ws_norm1_, ws_norm2_, ws_add1_, ws_add2_;

    void init_workspace() {
        size_t hs = config_.hidden_size;
        ws_norm1_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_norm2_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_add1_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_add2_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
    }
};

} // namespace llaisys
