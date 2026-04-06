#pragma once
#include "components.hpp"
#include "qwen2_impl.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../device/runtime_api.hpp"
#include "../../core/llaisys_core.hpp"

#ifdef ENABLE_NVIDIA_API
#include "../../ops/kv_cache_quant/nvidia/kv_cache_quant_nvidia.hpp"
#include "../../ops/self_attention/nvidia/self_attention_nvidia.hpp"
#endif

#include <cmath>
#include <cstring>

namespace llaisys {

class Qwen2Attention {
public:
    Qwen2Attention(const Qwen2Config& config) : config_(config) {
        init_kv_cache();
        init_workspace(1);
    }

    // === FP32 path: separate QKV ===
    void set_params(void* q_w, void* k_w, void* v_w, void* o_w,
                    void* q_b, void* k_b, void* v_b) {
        q_proj_.set_params(q_w, q_b);
        k_proj_.set_params(k_w, k_b);
        v_proj_.set_params(v_w, v_b);
        o_proj_.set_params(o_w);
        qkv_merged_ = false;
    }

    // === INT8 W8A16 path: merged QKV ===
    void set_params_quantized(void* q_w, void* k_w, void* v_w, void* o_w,
                              void* q_b, void* k_b, void* v_b,
                              void* q_s, void* k_s, void* v_s, void* o_s) {
        merge_qkv_quantized(
            cast_handle(q_w), cast_handle(k_w), cast_handle(v_w),
            cast_handle(q_b), cast_handle(k_b), cast_handle(v_b),
            cast_handle(q_s), cast_handle(k_s), cast_handle(v_s));
        o_proj_.set_params_quantized(o_w, o_s);
        qkv_merged_ = true;
    }

    // === INT4 W4A16 path: merged QKV ===
    void set_params_int4(void* q_w, void* k_w, void* v_w, void* o_w,
                         void* q_b, void* k_b, void* v_b,
                         void* q_s, void* k_s, void* v_s, void* o_s,
                         size_t gs, size_t q_K, size_t k_K, size_t v_K, size_t o_K) {
        merge_qkv_int4(
            cast_handle(q_w), cast_handle(k_w), cast_handle(v_w),
            cast_handle(q_b), cast_handle(k_b), cast_handle(v_b),
            cast_handle(q_s), cast_handle(k_s), cast_handle(v_s),
            gs, q_K);
        o_proj_.set_params_int4(o_w, o_s, gs, o_K);
        qkv_merged_ = true;
    }

    void prepare_for_seq_len(size_t seq_len) {
        if (seq_len == current_seq_len_) return;
        init_workspace(seq_len);
    }

    tensor_t forward(tensor_t x, size_t start_pos, tensor_t pos_tensor) {
        size_t seq_len = x->shape()[0];
        tensor_t q, k, v;

        if (qkv_merged_) {
            // === Merged path: single QKV GEMM ===
            qkv_proj_.forward(ws_qkv_, x);

            size_t nq = config_.hidden_size;
            size_t nkv = config_.kv_dim();

            if (seq_len == 1) {
                q = ws_qkv_->slice(1, 0, nq)->view(q_shape_);
                k = ws_qkv_->slice(1, nq, nq + nkv)->view(kv_shape_);
                v = ws_qkv_->slice(1, nq + nkv, nq + 2 * nkv)->view(kv_shape_);
            } else {
                split_qkv_batch(ws_qkv_, ws_q_2d_, ws_k_2d_, ws_v_2d_, seq_len);
                q = ws_q_2d_->view(q_shape_);
                k = ws_k_2d_->view(kv_shape_);
                v = ws_v_2d_->view(kv_shape_);
            }
        } else {
            // === Separate path: 3 GEMMs ===
            q_proj_.forward(ws_q_2d_, x);
            k_proj_.forward(ws_k_2d_, x);
            v_proj_.forward(ws_v_2d_, x);
            q = ws_q_2d_->view(q_shape_);
            k = ws_k_2d_->view(kv_shape_);
            v = ws_v_2d_->view(kv_shape_);
        }

        // RoPE
        ops::rope(ws_q_rope_, q, pos_tensor, config_.rope_theta);
        ops::rope(ws_k_rope_, k, pos_tensor, config_.rope_theta);

        // KV Cache update
        if (config_.kv_cache_int8) {
            update_cache_int8(k_cache_int8_, k_scales_, ws_k_rope_, start_pos, seq_len);
            update_cache_int8(v_cache_int8_, v_scales_, v, start_pos, seq_len);

            // Dequantize valid portion for attention
            size_t valid_len = start_pos + seq_len;
            dequant_cache(ws_k_dequant_, k_cache_int8_, k_scales_, valid_len);
            dequant_cache(ws_v_dequant_, v_cache_int8_, v_scales_, valid_len);

            auto k_valid = ws_k_dequant_->slice(0, 0, valid_len);
            auto v_valid = ws_v_dequant_->slice(0, 0, valid_len);

            float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
            ops::self_attention(ws_attn_3d_, ws_q_rope_, k_valid, v_valid, scale);
        } else {
            update_cache(k_cache_, ws_k_rope_, start_pos, seq_len);
            update_cache(v_cache_, v, start_pos, seq_len);

            float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
            auto k_valid = k_cache_->slice(0, 0, start_pos + seq_len);
            auto v_valid = v_cache_->slice(0, 0, start_pos + seq_len);
            ops::self_attention(ws_attn_3d_, ws_q_rope_, k_valid, v_valid, scale);
        }

        // O Projection
        auto attn_2d = ws_attn_3d_->view(out_2d_shape_);
        o_proj_.forward(ws_o_out_, attn_2d);
        return ws_o_out_;
    }

    // === CUDA Graph static capture path (decode only, seq_len=1) ===
    // Uses device pointers for start_pos/total_len → graph captures pointer
    // addresses (constant). Values at pointers updated via H2D before launch.
    tensor_t forward_graph(tensor_t x, const size_t* d_start_pos,
                           size_t total_len_hint, const size_t* d_total_len,
                           tensor_t pos_tensor) {
        tensor_t q, k, v;

        if (qkv_merged_) {
            qkv_proj_.forward(ws_qkv_, x); // 一个 GEMM 算出 Q/K/V
            size_t nq = config_.hidden_size;
            size_t nkv = config_.kv_dim();
            q = ws_qkv_->slice(1, 0, nq)->view(q_shape_);
            k = ws_qkv_->slice(1, nq, nq + nkv)->view(kv_shape_);
            v = ws_qkv_->slice(1, nq + nkv, nq + 2 * nkv)->view(kv_shape_);
        } else {
            q_proj_.forward(ws_q_2d_, x);
            k_proj_.forward(ws_k_2d_, x);
            v_proj_.forward(ws_v_2d_, x);
            q = ws_q_2d_->view(q_shape_);
            k = ws_k_2d_->view(kv_shape_);
            v = ws_v_2d_->view(kv_shape_);
        }
        // RoPE 位置编码
        ops::rope(ws_q_rope_, q, pos_tensor, config_.rope_theta);
        ops::rope(ws_k_rope_, k, pos_tensor, config_.rope_theta);

        float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

#ifdef ENABLE_NVIDIA_API
        if (config_.kv_cache_int8) {
            // Quantize new token → cache at d_start_pos (grid constant)
            ops::nvidia::kv_quantize_to_cache(
                reinterpret_cast<int8_t*>(k_cache_int8_->data()),
                reinterpret_cast<float*>(k_scales_->data()),
                ws_k_rope_->data(), ws_k_rope_->dtype(),
                0, 1, config_.num_key_value_heads, config_.head_dim,
                config_.max_position_embeddings, d_start_pos);
            ops::nvidia::kv_quantize_to_cache(
                reinterpret_cast<int8_t*>(v_cache_int8_->data()),
                reinterpret_cast<float*>(v_scales_->data()),
                v->data(), v->dtype(),
                0, 1, config_.num_key_value_heads, config_.head_dim,
                config_.max_position_embeddings, d_start_pos);

            // Incremental dequant: only position *d_start_pos (grid=(1,nkvh) constant)
            ops::nvidia::kv_dequantize_from_cache(
                ws_k_dequant_->data(), ws_k_dequant_->dtype(),
                reinterpret_cast<const int8_t*>(k_cache_int8_->data()),
                reinterpret_cast<const float*>(k_scales_->data()),
                1, config_.num_key_value_heads, config_.head_dim, d_start_pos);
            ops::nvidia::kv_dequantize_from_cache(
                ws_v_dequant_->data(), ws_v_dequant_->dtype(),
                reinterpret_cast<const int8_t*>(v_cache_int8_->data()),
                reinterpret_cast<const float*>(v_scales_->data()),
                1, config_.num_key_value_heads, config_.head_dim, d_start_pos);

            // Self attention: full dequant workspace, d_total_len for loops
            ops::nvidia::self_attention(
                ws_attn_3d_->data(), ws_attn_3d_->dtype(),
                ws_q_rope_->data(),
                ws_k_dequant_->data(),
                ws_v_dequant_->data(),
                1, total_len_hint,
                config_.num_attention_heads, config_.num_key_value_heads,
                config_.head_dim, config_.head_dim, scale,
                d_total_len);
        } else {
            // FP32: copy to cache via device kernel (d_start_pos)
            ops::nvidia::kv_cache_copy(
                k_cache_->data(), ws_k_rope_->data(), ws_k_rope_->dtype(),
                d_start_pos, config_.num_key_value_heads, config_.head_dim);
            ops::nvidia::kv_cache_copy(
                v_cache_->data(), v->data(), v->dtype(),
                d_start_pos, config_.num_key_value_heads, config_.head_dim);

            ops::nvidia::self_attention(
                ws_attn_3d_->data(), ws_attn_3d_->dtype(),
                ws_q_rope_->data(),
                k_cache_->data(),
                v_cache_->data(),
                1, total_len_hint,
                config_.num_attention_heads, config_.num_key_value_heads,
                config_.head_dim, config_.head_dim, scale,
                d_total_len);
        }
#else
        ASSERT(false, "forward_graph requires NVIDIA backend");
#endif

        auto attn_2d = ws_attn_3d_->view(out_2d_shape_);
        o_proj_.forward(ws_o_out_, attn_2d);
        return ws_o_out_;
    }

    // Legacy interface
    tensor_t forward(tensor_t x, size_t pos) {
        std::vector<size_t> pos_shape = {x->shape()[0]};
        std::vector<int64_t> pos_host(pos_shape[0]);
        for (size_t i = 0; i < pos_shape[0]; i++)
            pos_host[i] = static_cast<int64_t>(pos + i);
        auto pos_tensor = Tensor::create(pos_shape, LLAISYS_DTYPE_I64,
                                         config_.device_type, config_.device_id);
        pos_tensor->load(pos_host.data());
        return forward(x, pos, pos_tensor);
    }

    void reset_cache() { cache_pos_ = 0; }

private:
    Qwen2Config config_;
    bool qkv_merged_ = false;
    size_t current_seq_len_ = 0;

    // Merged path
    Linear qkv_proj_, o_proj_;
    tensor_t qkv_weight_, qkv_bias_, qkv_scales_;

    // Separate path (FP32)
    Linear q_proj_, k_proj_, v_proj_;

    // FP32 KV Cache (original)
    tensor_t k_cache_, v_cache_;
    size_t cache_pos_ = 0;

    // INT8 KV Cache
    tensor_t k_cache_int8_, v_cache_int8_;   // [max_seq_len, num_kv_heads, head_dim] INT8
    tensor_t k_scales_, v_scales_;           // [max_seq_len, num_kv_heads] F32
    tensor_t ws_k_dequant_, ws_v_dequant_;   // [max_seq_len, num_kv_heads, head_dim] compute_dtype

    // Workspace
    tensor_t ws_qkv_;
    tensor_t ws_q_2d_, ws_k_2d_, ws_v_2d_;
    tensor_t ws_q_rope_, ws_k_rope_;
    tensor_t ws_attn_3d_, ws_o_out_;
    std::vector<size_t> q_shape_, kv_shape_, out_2d_shape_;

    // ====================== Init ======================

    void init_kv_cache() {
        std::vector<size_t> shape = {
            config_.max_position_embeddings,
            config_.num_key_value_heads,
            config_.head_dim
        };

        if (config_.kv_cache_int8) {
            // INT8 cache + F32 scales
            k_cache_int8_ = Tensor::create(shape, LLAISYS_DTYPE_I8,
                                           config_.device_type, config_.device_id);
            v_cache_int8_ = Tensor::create(shape, LLAISYS_DTYPE_I8,
                                           config_.device_type, config_.device_id);

            std::vector<size_t> scale_shape = {
                config_.max_position_embeddings,
                config_.num_key_value_heads
            };
            k_scales_ = Tensor::create(scale_shape, LLAISYS_DTYPE_F32,
                                       config_.device_type, config_.device_id);
            v_scales_ = Tensor::create(scale_shape, LLAISYS_DTYPE_F32,
                                       config_.device_type, config_.device_id);

            // Dequantization workspace (reused across calls)
            auto dt = config_.compute_dtype;
            ws_k_dequant_ = Tensor::create(shape, dt, config_.device_type, config_.device_id);
            ws_v_dequant_ = Tensor::create(shape, dt, config_.device_type, config_.device_id);
        } else {
            auto dt = config_.compute_dtype;
            k_cache_ = Tensor::create(shape, dt, config_.device_type, config_.device_id);
            v_cache_ = Tensor::create(shape, dt, config_.device_type, config_.device_id);
        }
    }

    void init_workspace(size_t seq_len) {
        current_seq_len_ = seq_len;
        size_t hs = config_.hidden_size;
        size_t kv = config_.kv_dim();
        size_t n_total = hs + 2 * kv;

        q_shape_ = {seq_len, config_.num_attention_heads, config_.head_dim};
        kv_shape_ = {seq_len, config_.num_key_value_heads, config_.head_dim};
        out_2d_shape_ = {seq_len, hs};

        auto dt = config_.compute_dtype;
        auto dev = config_.device_type;
        auto did = config_.device_id;

        ws_qkv_ = Tensor::create({seq_len, n_total}, dt, dev, did);
        ws_q_2d_ = Tensor::create({seq_len, hs}, dt, dev, did);
        ws_k_2d_ = Tensor::create({seq_len, kv}, dt, dev, did);
        ws_v_2d_ = Tensor::create({seq_len, kv}, dt, dev, did);

        ws_q_rope_  = Tensor::create(q_shape_, dt, dev, did);
        ws_k_rope_  = Tensor::create(kv_shape_, dt, dev, did);
        ws_attn_3d_ = Tensor::create(q_shape_, dt, dev, did);
        ws_o_out_   = Tensor::create({seq_len, hs}, dt, dev, did);
    }

    // ====================== QKV Batch Split ======================

    void split_qkv_batch(tensor_t qkv, tensor_t q_out, tensor_t k_out, tensor_t v_out,
                                     size_t seq_len) {
        size_t nq = config_.hidden_size;
        size_t nkv = config_.kv_dim();
        size_t n_total = nq + 2 * nkv;
        size_t elem = qkv->elementSize();

        const LlaisysRuntimeAPI* api = nullptr;
        llaisysStream_t stream = nullptr;
        if (config_.device_type != LLAISYS_DEVICE_CPU) {
            api = llaisysGetRuntimeAPI(config_.device_type);
            stream = core::context().runtime().stream();
        }

        for (size_t i = 0; i < seq_len; ++i) {
            uint8_t* src = reinterpret_cast<uint8_t*>(qkv->data()) + i * n_total * elem;
            uint8_t* q_dst = reinterpret_cast<uint8_t*>(q_out->data()) + i * nq * elem;
            uint8_t* k_dst = reinterpret_cast<uint8_t*>(k_out->data()) + i * nkv * elem;
            uint8_t* v_dst = reinterpret_cast<uint8_t*>(v_out->data()) + i * nkv * elem;

            if (api) {
                api->memcpy_async(q_dst, src, nq * elem, LLAISYS_MEMCPY_D2D, stream);
                api->memcpy_async(k_dst, src + nq * elem, nkv * elem, LLAISYS_MEMCPY_D2D, stream);
                api->memcpy_async(v_dst, src + (nq + nkv) * elem, nkv * elem, LLAISYS_MEMCPY_D2D, stream);
            } else {
                std::memcpy(q_dst, src, nq * elem);
                std::memcpy(k_dst, src + nq * elem, nkv * elem);
                std::memcpy(v_dst, src + (nq + nkv) * elem, nkv * elem);
            }
        }
    }

    // ====================== QKV Weight Merge ======================

    void d2d_copy(void* dst, const void* src, size_t bytes) {
        if (config_.device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, bytes);
        } else {
            const auto* api = llaisysGetRuntimeAPI(config_.device_type);
            api->memcpy_sync(dst, src, bytes, LLAISYS_MEMCPY_D2D);
        }
    }

    tensor_t concat_dim0(tensor_t a, tensor_t b, tensor_t c) {
        size_t na = a->shape()[0], nb = b->shape()[0], nc = c->shape()[0];
        size_t cols = a->numel() / na;
        size_t elem = a->elementSize();
        auto merged = Tensor::create({na + nb + nc, cols}, a->dtype(),
                                     config_.device_type, config_.device_id);
        auto* dst = merged->data();
        d2d_copy(dst, a->data(), na * cols * elem);
        d2d_copy(dst + na * cols * elem, b->data(), nb * cols * elem);
        d2d_copy(dst + (na + nb) * cols * elem, c->data(), nc * cols * elem);
        return merged;
    }

    tensor_t concat_1d(tensor_t a, tensor_t b, tensor_t c) {
        size_t na = a->numel(), nb = b->numel(), nc = c->numel();
        size_t elem = a->elementSize();
        auto merged = Tensor::create({na + nb + nc}, a->dtype(),
                                     config_.device_type, config_.device_id);
        auto* dst = merged->data();
        d2d_copy(dst, a->data(), na * elem);
        d2d_copy(dst + na * elem, b->data(), nb * elem);
        d2d_copy(dst + (na + nb) * elem, c->data(), nc * elem);
        return merged;
    }

    void merge_qkv_quantized(tensor_t qw, tensor_t kw, tensor_t vw,
                             tensor_t qb, tensor_t kb, tensor_t vb,
                             tensor_t qs, tensor_t ks, tensor_t vs) {
        qkv_weight_ = concat_dim0(qw, kw, vw);
        qkv_scales_ = concat_1d(qs, ks, vs);
        if (qb) qkv_bias_ = concat_1d(qb, kb, vb);
        qkv_proj_.set_params_quantized_direct(qkv_weight_, qkv_scales_, qkv_bias_);
    }

    void merge_qkv_int4(tensor_t qw, tensor_t kw, tensor_t vw,
                        tensor_t qb, tensor_t kb, tensor_t vb,
                        tensor_t qs, tensor_t ks, tensor_t vs,
                        size_t gs, size_t K_orig) {
        qkv_weight_ = concat_dim0(qw, kw, vw);
        qkv_scales_ = concat_dim0(qs, ks, vs);
        if (qb) qkv_bias_ = concat_1d(qb, kb, vb);
        qkv_proj_.set_params_int4_direct(qkv_weight_, qkv_scales_, gs, K_orig, qkv_bias_);
    }

    // ====================== KV Cache (FP32 Original) ======================

    void update_cache(tensor_t cache, tensor_t update, size_t start_pos, size_t seq_len) {
        size_t row_size = config_.kv_dim() * update->elementSize();
        uint8_t* dst = reinterpret_cast<uint8_t*>(cache->data()) + row_size * start_pos;
        uint8_t* src = reinterpret_cast<uint8_t*>(update->data());
        size_t total_bytes = row_size * seq_len;
        if (config_.device_type == LLAISYS_DEVICE_CPU) {
            std::memcpy(dst, src, total_bytes);
        } else {
            const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
            llaisysStream_t stream = core::context().runtime().stream();
            api->memcpy_async(dst, src, total_bytes, LLAISYS_MEMCPY_D2D, stream);
        }
    }

    // ====================== KV Cache INT8 ======================

    void update_cache_int8(tensor_t cache_int8, tensor_t scales,
                           tensor_t update, size_t start_pos, size_t seq_len) {
#ifdef ENABLE_NVIDIA_API
        ops::nvidia::kv_quantize_to_cache(
            reinterpret_cast<int8_t*>(cache_int8->data()),
            reinterpret_cast<float*>(scales->data()),
            update->data(),
            update->dtype(),
            start_pos,
            seq_len,
            config_.num_key_value_heads,
            config_.head_dim,
            config_.max_position_embeddings
        );
#else
        ASSERT(false, "KV Cache INT8 requires NVIDIA backend");
#endif
    }

    void dequant_cache(tensor_t dst, tensor_t cache_int8, tensor_t scales, size_t valid_len) {
#ifdef ENABLE_NVIDIA_API
        ops::nvidia::kv_dequantize_from_cache(
            dst->data(),
            dst->dtype(),
            reinterpret_cast<const int8_t*>(cache_int8->data()),
            reinterpret_cast<const float*>(scales->data()),
            valid_len,
            config_.num_key_value_heads,
            config_.head_dim
        );
#else
        ASSERT(false, "KV Cache INT8 requires NVIDIA backend");
#endif
    }
};

} // namespace llaisys
