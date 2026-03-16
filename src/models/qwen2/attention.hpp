#pragma once
#include "components.hpp"
#include "qwen2_impl.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../device/runtime_api.hpp"
#include "../../core/llaisys_core.hpp"
#include <cmath>

namespace llaisys {

class Qwen2Attention {
public:
    Qwen2Attention(const Qwen2Config& config) : config_(config) {
        init_kv_cache();
        init_workspace();
    }

    void set_params(void* q_w, void* k_w, void* v_w, void* o_w,
                    void* q_b, void* k_b, void* v_b) {
        q_proj_.set_params(q_w, q_b);
        k_proj_.set_params(k_w, k_b);
        v_proj_.set_params(v_w, v_b);
        o_proj_.set_params(o_w);
    }

    // 量化版本: 权重为 INT8 + per-channel scales
    void set_params_quantized(void* q_w, void* k_w, void* v_w, void* o_w,
                              void* q_b, void* k_b, void* v_b,
                              void* q_s, void* k_s, void* v_s, void* o_s) {
        q_proj_.set_params_quantized(q_w, q_s, q_b);
        k_proj_.set_params_quantized(k_w, k_s, k_b);
        v_proj_.set_params_quantized(v_w, v_s, v_b);
        o_proj_.set_params_quantized(o_w, o_s);
    }

    // INT4 量化版本: packed U8 权重 + group F16 scales
    void set_params_int4(void* q_w, void* k_w, void* v_w, void* o_w,
                         void* q_b, void* k_b, void* v_b,
                         void* q_s, void* k_s, void* v_s, void* o_s,
                         size_t gs, size_t q_K, size_t k_K, size_t v_K, size_t o_K) {
        q_proj_.set_params_int4(q_w, q_s, gs, q_K, q_b);
        k_proj_.set_params_int4(k_w, k_s, gs, k_K, k_b);
        v_proj_.set_params_int4(v_w, v_s, gs, v_K, v_b);
        o_proj_.set_params_int4(o_w, o_s, gs, o_K);
    }

    // 优化版 forward：接受外部传入的 pos_tensor（每 token 只创建一次）
    tensor_t forward(tensor_t x, size_t pos, tensor_t pos_tensor) {
        // Q, K, V 投影 — 使用预分配输出张量
        q_proj_.forward(ws_q_2d_, x);
        k_proj_.forward(ws_k_2d_, x);
        v_proj_.forward(ws_v_2d_, x);

        // Reshape 为多头格式（view 不分配新内存）
        auto q_3d = ws_q_2d_->view(q_shape_);
        auto k_3d = ws_k_2d_->view(kv_shape_);
        auto v_3d = ws_v_2d_->view(kv_shape_);

        // 应用RoPE — 使用预分配输出张量
        ops::rope(ws_q_rope_, q_3d, pos_tensor, config_.rope_theta);
        ops::rope(ws_k_rope_, k_3d, pos_tensor, config_.rope_theta);

        // 更新KV缓存（GPU 用异步 D2D，CPU 用 memcpy）
        update_cache(k_cache_, ws_k_rope_, pos);
        update_cache(v_cache_, v_3d, pos);

        // 自注意力计算
        float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
        auto k_valid = k_cache_->slice(0, 0, pos + 1);
        auto v_valid = v_cache_->slice(0, 0, pos + 1);
        ops::self_attention(ws_attn_3d_, ws_q_rope_, k_valid, v_valid, scale);

        // Reshape 回 2D 并通过 O 投影
        auto attn_2d = ws_attn_3d_->view(out_2d_shape_);
        o_proj_.forward(ws_o_out_, attn_2d);
        return ws_o_out_;
    }

    // 兼容旧接口
    tensor_t forward(tensor_t x, size_t pos) {
        std::vector<size_t> pos_shape = {x->shape()[0]};
        std::vector<int64_t> pos_host(pos_shape[0]);
        for (size_t i = 0; i < pos_shape[0]; i++)
            pos_host[i] = static_cast<int64_t>(pos + i);
        auto pos_tensor = Tensor::create(pos_shape, LLAISYS_DTYPE_I64, config_.device_type, config_.device_id);
        pos_tensor->load(pos_host.data());
        return forward(x, pos, pos_tensor);
    }

    void reset_cache() { cache_pos_ = 0; }

private:
    Qwen2Config config_;
    Linear q_proj_, k_proj_, v_proj_, o_proj_;
    tensor_t k_cache_, v_cache_;
    size_t cache_pos_ = 0;

    // 预分配的工作空间张量（decode 阶段 seq_len=1）
    tensor_t ws_q_2d_, ws_k_2d_, ws_v_2d_;     // 线性投影输出
    tensor_t ws_q_rope_, ws_k_rope_;            // RoPE 输出
    tensor_t ws_attn_3d_;                       // 注意力输出
    tensor_t ws_o_out_;                         // O 投影输出
    std::vector<size_t> q_shape_, kv_shape_, out_2d_shape_;

    void init_kv_cache() {
        std::vector<size_t> shape = {
            config_.max_position_embeddings,
            config_.num_key_value_heads,
            config_.head_dim
        };
        k_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        v_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
    }

    void init_workspace() {
        // 预分配 decode 阶段（seq_len=1）的所有中间张量
        size_t hs = config_.hidden_size;
        size_t kv_dim = config_.kv_dim();
        q_shape_ = {1, config_.num_attention_heads, config_.head_dim};
        kv_shape_ = {1, config_.num_key_value_heads, config_.head_dim};
        out_2d_shape_ = {1, hs};

        ws_q_2d_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_k_2d_ = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_v_2d_ = Tensor::create({1, kv_dim}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_q_rope_ = Tensor::create(q_shape_, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_k_rope_ = Tensor::create(kv_shape_, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_attn_3d_ = Tensor::create(q_shape_, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
        ws_o_out_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, config_.device_type, config_.device_id);
    }

    void update_cache(tensor_t cache, tensor_t update, size_t pos) {
        size_t bytes_per_elem = 4;
        if (update->dtype() == LLAISYS_DTYPE_F16) bytes_per_elem = 2;
        if (update->dtype() == LLAISYS_DTYPE_BF16) bytes_per_elem = 2;
        size_t row_size = config_.kv_dim() * bytes_per_elem;
        uint8_t* dst = reinterpret_cast<uint8_t*>(cache->data()) + row_size * pos;
        uint8_t* src = reinterpret_cast<uint8_t*>(update->data());

        if (config_.device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, row_size);
        } else {
            // 使用异步 D2D 拷贝，避免同步阻塞
            const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
            llaisysStream_t stream = core::context().runtime().stream();
            api->memcpy_async(dst, src, row_size, LLAISYS_MEMCPY_D2D, stream);
        }
    }
};

} // namespace llaisys
