#pragma once
#include "attention.hpp"
#include "mlp.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/fused_add_rmsnorm/op.hpp"

namespace llaisys {

class Qwen2DecoderLayer {
public:
    Qwen2DecoderLayer(const Qwen2Config& config)
        : config_(config),
          attn_(config),
          input_norm_(config.rms_norm_eps),
          post_attn_norm_(config.rms_norm_eps) {
        mlp_.init_workspace(config);
        init_workspace(1);
    }

    void set_params(const LlaisysQwen2Weights* w, size_t layer_idx) {
        attn_.set_params(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx]);
        mlp_.set_params(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx]);
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    void set_params_quantized(const LlaisysQwen2Weights* w, size_t layer_idx) {
        attn_.set_params_quantized(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx],
            w->attn_q_w_scales[layer_idx], w->attn_k_w_scales[layer_idx],
            w->attn_v_w_scales[layer_idx], w->attn_o_w_scales[layer_idx]);
        mlp_.set_params_quantized(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx],
            w->mlp_gate_w_scales[layer_idx], w->mlp_up_w_scales[layer_idx],
            w->mlp_down_w_scales[layer_idx]);
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    void set_params_int4(const LlaisysQwen2Weights* w, size_t layer_idx) {
        size_t gs = w->int4_group_size;
        size_t base = layer_idx * 7;
        attn_.set_params_int4(
            w->attn_q_w[layer_idx], w->attn_k_w[layer_idx],
            w->attn_v_w[layer_idx], w->attn_o_w[layer_idx],
            w->attn_q_b[layer_idx], w->attn_k_b[layer_idx],
            w->attn_v_b[layer_idx],
            w->attn_q_w_scales[layer_idx], w->attn_k_w_scales[layer_idx],
            w->attn_v_w_scales[layer_idx], w->attn_o_w_scales[layer_idx],
            gs, w->int4_K_orig[base+0], w->int4_K_orig[base+1],
            w->int4_K_orig[base+2], w->int4_K_orig[base+3]);
        mlp_.set_params_int4(
            w->mlp_gate_w[layer_idx], w->mlp_up_w[layer_idx],
            w->mlp_down_w[layer_idx],
            w->mlp_gate_w_scales[layer_idx], w->mlp_up_w_scales[layer_idx],
            w->mlp_down_w_scales[layer_idx],
            gs, w->int4_K_orig[base+4], w->int4_K_orig[base+5],
            w->int4_K_orig[base+6]);
        input_norm_.set_weight(w->attn_norm_w[layer_idx]);
        post_attn_norm_.set_weight(w->mlp_norm_w[layer_idx]);
    }

    void set_norm_tensors(tensor_t attn_norm, tensor_t mlp_norm) {
        attn_norm_tensor_ = attn_norm;
        mlp_norm_tensor_ = mlp_norm;
    }

    void prepare_for_seq_len(size_t seq_len) {
        if (seq_len == current_seq_len_) return;
        init_workspace(seq_len);
        attn_.prepare_for_seq_len(seq_len);
        mlp_.prepare_for_seq_len(seq_len);
    }

    tensor_t forward(tensor_t x, size_t pos, tensor_t pos_tensor) {
        auto residual = x;
        input_norm_.forward(ws_norm1_, x);
        auto attn_out = attn_.forward(ws_norm1_, pos, pos_tensor);

        if (mlp_norm_tensor_) {
            ops::fused_add_rmsnorm(ws_norm2_, ws_add1_, attn_out, residual,
                                   mlp_norm_tensor_, config_.rms_norm_eps);
        } else {
            ops::add(ws_add1_, attn_out, residual);
            post_attn_norm_.forward(ws_norm2_, ws_add1_);
        }

        residual = ws_add1_;
        auto mlp_out = mlp_.forward(ws_norm2_);
        ops::add(ws_add2_, mlp_out, residual);
        return ws_add2_;
    }

    tensor_t forward(tensor_t x, size_t pos) {
        std::vector<size_t> ps = {1};
        std::vector<int64_t> ph = {static_cast<int64_t>(pos)};
        auto pt = Tensor::create(ps, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
        pt->load(ph.data());
        return forward(x, pos, pt);
    }

private:
    Qwen2Config config_;
    size_t current_seq_len_ = 0;
    Qwen2Attention attn_;
    Qwen2MLP mlp_;
    RMSNorm input_norm_, post_attn_norm_;
    tensor_t attn_norm_tensor_;
    tensor_t mlp_norm_tensor_;
    tensor_t ws_norm1_, ws_norm2_, ws_add1_, ws_add2_;

    void init_workspace(size_t seq_len) {
        current_seq_len_ = seq_len;
        size_t hs = config_.hidden_size;
        auto dt = config_.compute_dtype;
        ws_norm1_ = Tensor::create({seq_len, hs}, dt, config_.device_type, config_.device_id);
        ws_norm2_ = Tensor::create({seq_len, hs}, dt, config_.device_type, config_.device_id);
        ws_add1_ = Tensor::create({seq_len, hs}, dt, config_.device_type, config_.device_id);
        ws_add2_ = Tensor::create({seq_len, hs}, dt, config_.device_type, config_.device_id);
    }
};

} // namespace llaisys
