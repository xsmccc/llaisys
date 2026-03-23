#pragma once
#include "../qwen2/components.hpp"
#include "llama3_impl.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../device/runtime_api.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath>

namespace llaisys {

class Llama3Attention {
public:
    Llama3Attention(const Llama3Config& config) : config_(config) {
        init_kv_cache();
        init_workspace();
    }

    // LLaMA3: 无 bias, 所有参数都是 weight-only
    void set_params(void* q_w, void* k_w, void* v_w, void* o_w) {
        q_proj_.set_params(q_w);
        k_proj_.set_params(k_w);
        v_proj_.set_params(v_w);
        o_proj_.set_params(o_w);
    }

    tensor_t forward(tensor_t x, size_t pos, tensor_t pos_tensor) {
        q_proj_.forward(ws_q_2d_, x);
        k_proj_.forward(ws_k_2d_, x);
        v_proj_.forward(ws_v_2d_, x);

        auto q_3d = ws_q_2d_->view(q_shape_);
        auto k_3d = ws_k_2d_->view(kv_shape_);
        auto v_3d = ws_v_2d_->view(kv_shape_);

        ops::rope(ws_q_rope_, q_3d, pos_tensor, config_.rope_theta);
        ops::rope(ws_k_rope_, k_3d, pos_tensor, config_.rope_theta);

        update_cache(k_cache_, ws_k_rope_, pos);
        update_cache(v_cache_, v_3d, pos);

        float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
        auto k_valid = k_cache_->slice(0, 0, pos + 1);
        auto v_valid = v_cache_->slice(0, 0, pos + 1);
        ops::self_attention(ws_attn_3d_, ws_q_rope_, k_valid, v_valid, scale);

        auto attn_2d = ws_attn_3d_->view(out_2d_shape_);
        o_proj_.forward(ws_o_out_, attn_2d);
        return ws_o_out_;
    }

    void reset_cache() { cache_pos_ = 0; }

private:
    Llama3Config config_;
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
    tensor_t k_cache_, v_cache_;
    size_t cache_pos_ = 0;

    tensor_t ws_q_2d_, ws_k_2d_, ws_v_2d_;
    tensor_t ws_q_rope_, ws_k_rope_;
    tensor_t ws_attn_3d_;
    tensor_t ws_o_out_;
    std::vector<size_t> q_shape_, kv_shape_, out_2d_shape_;

    void init_kv_cache() {
        std::vector<size_t> shape = {
            config_.max_position_embeddings,
            config_.num_key_value_heads,
            config_.head_dim
        };
        k_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
        v_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
    }

    void init_workspace() {
        size_t hs = config_.hidden_size;
        size_t kv_dim = config_.kv_dim();
        q_shape_ = {1, config_.num_attention_heads, config_.head_dim};
        kv_shape_ = {1, config_.num_key_value_heads, config_.head_dim};
        out_2d_shape_ = {1, hs};

        ws_q_2d_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
        ws_k_2d_ = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
        ws_v_2d_ = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32,
                                  config_.device_type, config_.device_id);
        ws_q_rope_ = Tensor::create(q_shape_, LLAISYS_DTYPE_F32,
                                    config_.device_type, config_.device_id);
        ws_k_rope_ = Tensor::create(kv_shape_, LLAISYS_DTYPE_F32,
                                    config_.device_type, config_.device_id);
        ws_attn_3d_ = Tensor::create(q_shape_, LLAISYS_DTYPE_F32,
                                     config_.device_type, config_.device_id);
        ws_o_out_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32,
                                   config_.device_type, config_.device_id);
    }

    void update_cache(tensor_t cache, tensor_t update, size_t pos) {
        size_t bytes_per_elem = 4;
        size_t row_size = config_.kv_dim() * bytes_per_elem;
        uint8_t* dst = reinterpret_cast<uint8_t*>(cache->data()) + row_size * pos;
        uint8_t* src = reinterpret_cast<uint8_t*>(update->data());

        if (config_.device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, row_size);
        } else {
            const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
            llaisysStream_t stream = core::context().runtime().stream();
            api->memcpy_async(dst, src, row_size, LLAISYS_MEMCPY_D2D, stream);
        }
    }
};

} // namespace llaisys
