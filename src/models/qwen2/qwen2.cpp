#include "llaisys/models/qwen2.h"
#include "qwen2_impl.hpp"
#include "layer.hpp"
#include "components.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../device/runtime_api.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../ops/linear_quantized/nvidia/linear_quantized_nvidia.hpp"
#include "../../ops/self_attention/nvidia/self_attention_nvidia.hpp"
#include "cuda_graph_manager.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <random>
#include <cmath>

using namespace llaisys;

// ============ Random Sampling ============

static int64_t sample_token(float* logits, size_t vocab_size,
                            float temperature, int top_k, float top_p,
                            uint64_t seed) {
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (size_t i = 0; i < vocab_size; ++i) logits[i] *= inv_temp;
    }
    float max_logit = *std::max_element(logits, logits + vocab_size);
    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_logit);
        sum_exp += logits[i];
    }
    float inv_sum = static_cast<float>(1.0 / sum_exp);
    for (size_t i = 0; i < vocab_size; ++i) logits[i] *= inv_sum;

    std::vector<int64_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        return logits[a] > logits[b];
    });

    size_t cutoff = vocab_size;
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size)
        cutoff = static_cast<size_t>(top_k);
    if (top_p > 0.0f && top_p < 1.0f) {
        double cumsum = 0.0;
        for (size_t i = 0; i < cutoff; ++i) {
            cumsum += logits[indices[i]];
            if (cumsum >= static_cast<double>(top_p)) { cutoff = i + 1; break; }
        }
    }
    for (size_t i = cutoff; i < vocab_size; ++i) logits[indices[i]] = 0.0f;
    double new_sum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) new_sum += logits[indices[i]];
    if (new_sum > 0.0) {
        float inv_new_sum = static_cast<float>(1.0 / new_sum);
        for (size_t i = 0; i < cutoff; ++i) logits[indices[i]] *= inv_new_sum;
    }

    std::mt19937_64 rng(seed != 0 ? seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        cumulative += logits[indices[i]];
        if (r <= cumulative) return indices[i];
    }
    return indices[0];
}

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int device_id = 0)
        : config_(*meta, device, device_id),
          embed_(), final_norm_(config_.rms_norm_eps), lm_head_()
    {
        init_weight_arrays(config_.num_hidden_layers);
        std::cerr << "[Qwen2] Model created with " << config_.num_hidden_layers
                  << " layers, hidden_size=" << config_.hidden_size << std::endl;

#ifdef ENABLE_NVIDIA_API
        if (config_.device_type == LLAISYS_DEVICE_NVIDIA) {
            const char* graph_env = std::getenv("LLAISYS_CUDA_GRAPH");
            if (graph_env && std::string(graph_env) == "1") {
                decode_graph_ = std::make_unique<llaisys::models::qwen2::CudaGraphManager>();
                cuda_graph_enabled_ = true;
                std::cerr << "[Qwen2] CUDA Graph enabled for decode" << std::endl;
            }
        }
#endif
    }

    ~Qwen2Model() {
#ifdef ENABLE_NVIDIA_API
        if (decode_graph_) { decode_graph_->printStats(); decode_graph_.reset(); }
#endif
        free_weight_arrays();
        // 清理 FP16 权重缓存和 FlashDecoding workspace
#ifdef ENABLE_NVIDIA_API
        if (config_.device_type == LLAISYS_DEVICE_NVIDIA) {
            llaisys::ops::nvidia::cleanup_quantized_weight_cache();
            llaisys::ops::nvidia::cleanup_self_attention_workspace();
        }
#endif
    }

    LlaisysQwen2Weights* get_weights_struct() { return &weights_; }

    /**
     * Batch-aware inference.
     * ntoken=N: 所有 token 一次 forward pass（batch prefill）
     * ntoken=1: 单 token decode
     */
    int64_t infer(int64_t* token_ids, size_t ntoken) {
        if (!weights_distributed_) { distribute_weights(); weights_distributed_ = true; }
        if (!weights_.in_embed) { std::cerr << "[ERROR] Weights not loaded!" << std::endl; return 0; }

        prepare_for_seq_len(ntoken);

        // ── H2D loads (sync, outside graph capture) ──
        ws_token_->load(token_ids);

        std::vector<int64_t> positions(ntoken);
        for (size_t i = 0; i < ntoken; ++i)
            positions[i] = static_cast<int64_t>(current_pos_ + i);
        ws_pos_->load(positions.data());

        // ── Forward pass ──
        size_t saved_pos = current_pos_;
#ifdef ENABLE_NVIDIA_API
        if (cuda_graph_enabled_ && ntoken == 1) {
            cudaStream_t stream = (cudaStream_t)core::context().runtime().stream();
            decode_graph_->captureAndLaunch(stream, [&]() {
                embed_.forward(ws_hidden_, ws_token_);
                tensor_t current = ws_hidden_;
                for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx)
                    current = layers_[layer_idx].forward(current, saved_pos, ws_pos_);
                final_norm_.forward(ws_final_norm_, current);
                lm_head_.forward(ws_logits_, ws_final_norm_);
                ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);
            });
        } else
#endif
        {
            embed_.forward(ws_hidden_, ws_token_);
            tensor_t current = ws_hidden_;
            for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx)
                current = layers_[layer_idx].forward(current, saved_pos, ws_pos_);
            final_norm_.forward(ws_final_norm_, current);
            tensor_t last_hidden = (ntoken > 1)
                ? ws_final_norm_->slice(0, ntoken - 1, ntoken)
                : ws_final_norm_;
            lm_head_.forward(ws_logits_, last_hidden);
            ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);
        }
        current_pos_ += ntoken;

        // ── D2H output (sync, outside graph) ──
        int64_t output_token = 0;
        if (config_.device_type != LLAISYS_DEVICE_CPU) {
            const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
            core::context().runtime().synchronize();
            api->memcpy_sync(&output_token, ws_out_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        } else {
            output_token = *reinterpret_cast<int64_t*>(ws_out_idx_->data());
        }

        if (ntoken > 1) prepare_for_seq_len(1);

        std::cerr << std::endl;
        return output_token;
    }

    int64_t infer_ex(int64_t* token_ids, size_t ntoken,
                     float temperature, int top_k, float top_p, uint64_t seed) {
        if (!weights_distributed_) { distribute_weights(); weights_distributed_ = true; }
        if (!weights_.in_embed) { std::cerr << "[ERROR] Weights not loaded!" << std::endl; return 0; }

        prepare_for_seq_len(ntoken);

        // ── H2D loads (sync, outside graph capture) ──
        ws_token_->load(token_ids);

        std::vector<int64_t> positions(ntoken);
        for (size_t i = 0; i < ntoken; ++i)
            positions[i] = static_cast<int64_t>(current_pos_ + i);
        ws_pos_->load(positions.data());

        // ── Forward pass ──
        size_t saved_pos = current_pos_;
        bool is_greedy = (temperature <= 0.0f) || (top_k == 1) ||
                         (temperature == 1.0f && top_k <= 0 && top_p >= 1.0f);

#ifdef ENABLE_NVIDIA_API
        if (cuda_graph_enabled_ && ntoken == 1) {
            cudaStream_t stream = (cudaStream_t)core::context().runtime().stream();
            decode_graph_->captureAndLaunch(stream, [&]() {
                embed_.forward(ws_hidden_, ws_token_);
                tensor_t current = ws_hidden_;
                for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx)
                    current = layers_[layer_idx].forward(current, saved_pos, ws_pos_);
                final_norm_.forward(ws_final_norm_, current);
                lm_head_.forward(ws_logits_, ws_final_norm_);
                ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);
            });
            current_pos_ += ntoken;

            int64_t output_token = 0;
            const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
            core::context().runtime().synchronize();
            if (is_greedy) {
                api->memcpy_sync(&output_token, ws_out_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
            } else {
                size_t vocab_size = config_.vocab_size;
                std::vector<float> logits(vocab_size);
                api->memcpy_sync(logits.data(), ws_logits_->data(),
                                 vocab_size * sizeof(float), LLAISYS_MEMCPY_D2H);
                output_token = sample_token(logits.data(), vocab_size,
                                            temperature, top_k, top_p, seed);
            }
            return output_token;
        }
#endif

        embed_.forward(ws_hidden_, ws_token_);

        tensor_t current = ws_hidden_;
        for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx)
            current = layers_[layer_idx].forward(current, saved_pos, ws_pos_);

        final_norm_.forward(ws_final_norm_, current);
        current_pos_ += ntoken;

        tensor_t last_hidden = (ntoken > 1)
            ? ws_final_norm_->slice(0, ntoken - 1, ntoken)
            : ws_final_norm_;
        lm_head_.forward(ws_logits_, last_hidden);

        int64_t output_token = 0;

        if (is_greedy && top_k == 1) {
            ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);
            if (config_.device_type != LLAISYS_DEVICE_CPU) {
                const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
                core::context().runtime().synchronize();
                api->memcpy_sync(&output_token, ws_out_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
            } else {
                output_token = *reinterpret_cast<int64_t*>(ws_out_idx_->data());
            }
        } else {
            size_t vocab_size = config_.vocab_size;
            std::vector<float> logits(vocab_size);
            if (config_.device_type != LLAISYS_DEVICE_CPU) {
                const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
                core::context().runtime().synchronize();
                api->memcpy_sync(logits.data(), ws_logits_->data(),
                                 vocab_size * sizeof(float), LLAISYS_MEMCPY_D2H);
            } else {
                std::memcpy(logits.data(), ws_logits_->data(), vocab_size * sizeof(float));
            }
            output_token = sample_token(logits.data(), vocab_size,
                                        temperature, top_k, top_p, seed);
        }

        if (ntoken > 1) prepare_for_seq_len(1);

        std::cerr << std::endl;
        return output_token;
    }

    void reset() {
        current_pos_ = 0;
#ifdef ENABLE_NVIDIA_API
        if (decode_graph_) decode_graph_->reset();
#endif
    }

private:
    Qwen2Config config_;
    LlaisysQwen2Weights weights_;

    Embedding embed_;
    std::vector<Qwen2DecoderLayer> layers_;
    RMSNorm final_norm_;
    Linear lm_head_;

    size_t current_pos_ = 0;
    size_t current_ws_seq_len_ = 0;
    bool weights_distributed_ = false;
    bool cuda_graph_enabled_ = false;

    tensor_t ws_token_, ws_pos_, ws_hidden_, ws_final_norm_;
    tensor_t ws_logits_, ws_out_idx_, ws_out_val_;

#ifdef ENABLE_NVIDIA_API
    std::unique_ptr<llaisys::models::qwen2::CudaGraphManager> decode_graph_;
#endif

    void prepare_for_seq_len(size_t seq_len) {
        if (seq_len == current_ws_seq_len_) return;
        current_ws_seq_len_ = seq_len;

        size_t hs = config_.hidden_size;
        auto dev = config_.device_type;
        auto di = config_.device_id;
        auto cdt = config_.compute_dtype;

        ws_token_ = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, di);
        ws_pos_ = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, dev, di);
        ws_hidden_ = Tensor::create({seq_len, hs}, cdt, dev, di);
        ws_final_norm_ = Tensor::create({seq_len, hs}, cdt, dev, di);

        for (auto& layer : layers_)
            layer.prepare_for_seq_len(seq_len);
    }

    void init_inference_workspace() {
        size_t vs = config_.vocab_size;
        auto dev = config_.device_type;
        auto di = config_.device_id;

        ws_logits_ = Tensor::create({1, vs}, LLAISYS_DTYPE_F32, dev, di);
        ws_out_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, di);
        ws_out_val_ = Tensor::create({1}, LLAISYS_DTYPE_F32, dev, di);

        prepare_for_seq_len(1);
    }

    void init_weight_arrays(size_t nlayers) {
        weights_.attn_q_w = new llaisysTensor_t[nlayers];
        weights_.attn_k_w = new llaisysTensor_t[nlayers];
        weights_.attn_v_w = new llaisysTensor_t[nlayers];
        weights_.attn_o_w = new llaisysTensor_t[nlayers];
        weights_.attn_q_b = new llaisysTensor_t[nlayers];
        weights_.attn_k_b = new llaisysTensor_t[nlayers];
        weights_.attn_v_b = new llaisysTensor_t[nlayers];
        weights_.attn_norm_w = new llaisysTensor_t[nlayers];
        weights_.mlp_norm_w = new llaisysTensor_t[nlayers];
        weights_.mlp_gate_w = new llaisysTensor_t[nlayers];
        weights_.mlp_up_w = new llaisysTensor_t[nlayers];
        weights_.mlp_down_w = new llaisysTensor_t[nlayers];
        weights_.quantized = 0;
        weights_.attn_q_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_k_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_v_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_o_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_gate_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_up_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_down_w_scales = new llaisysTensor_t[nlayers];
        weights_.out_embed_scales = nullptr;
        weights_.int4_group_size = 128;
        weights_.int4_K_orig = new size_t[nlayers * 7 + 1];
        std::memset(weights_.int4_K_orig, 0, (nlayers * 7 + 1) * sizeof(size_t));

        std::memset(weights_.attn_q_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_k_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_v_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_o_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_q_b, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_k_b, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_v_b, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_norm_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_norm_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_gate_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_up_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_down_w, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_q_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_k_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_v_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.attn_o_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_gate_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_up_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
        std::memset(weights_.mlp_down_w_scales, 0, nlayers * sizeof(llaisysTensor_t));
    }

    void free_weight_arrays() {
        delete[] weights_.attn_q_w; delete[] weights_.attn_k_w;
        delete[] weights_.attn_v_w; delete[] weights_.attn_o_w;
        delete[] weights_.attn_q_b; delete[] weights_.attn_k_b;
        delete[] weights_.attn_v_b; delete[] weights_.attn_norm_w;
        delete[] weights_.mlp_norm_w; delete[] weights_.mlp_gate_w;
        delete[] weights_.mlp_up_w; delete[] weights_.mlp_down_w;
        delete[] weights_.attn_q_w_scales; delete[] weights_.attn_k_w_scales;
        delete[] weights_.attn_v_w_scales; delete[] weights_.attn_o_w_scales;
        delete[] weights_.mlp_gate_w_scales; delete[] weights_.mlp_up_w_scales;
        delete[] weights_.mlp_down_w_scales; delete[] weights_.int4_K_orig;
    }

    static tensor_t handle_to_tensor(llaisysTensor_t h) {
        if (!h) return nullptr;
        return *reinterpret_cast<tensor_t*>(h);
    }

    void distribute_norm_tensors() {
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto attn_norm = handle_to_tensor(weights_.attn_norm_w[i]);
            auto mlp_norm = handle_to_tensor(weights_.mlp_norm_w[i]);
            layers_[i].set_norm_tensors(attn_norm, mlp_norm);
        }
    }

    void distribute_weights() {
        bool is_quantized = (weights_.quantized != 0);
        bool is_int4 = (weights_.quantized == 2);

        if (is_quantized) {
            config_.compute_dtype = LLAISYS_DTYPE_F16;
            std::cerr << "[Qwen2] Quantized mode detected, using FP16 compute pipeline (W"
                      << (is_int4 ? "4" : "8") << "A16)" << std::endl;
        }

        // KV Cache INT8: enable via environment variable
        const char* kv_int8_env = std::getenv("LLAISYS_KV_CACHE_INT8");
        if (kv_int8_env && std::strcmp(kv_int8_env, "1") == 0) {
            config_.kv_cache_int8 = true;
            std::cerr << "[Qwen2] KV Cache INT8 quantization enabled" << std::endl;
        }

        layers_.reserve(config_.num_hidden_layers);
        for (size_t i = 0; i < config_.num_hidden_layers; ++i)
            layers_.emplace_back(config_);

        init_inference_workspace();

        embed_.set_weight(weights_.in_embed);
        final_norm_.set_weight(weights_.out_norm_w);

        if (is_int4) {
            if (weights_.out_embed_scales) {
                size_t nlayer = layers_.size();
                size_t lm_K = weights_.int4_K_orig[nlayer * 7];
                lm_head_.set_params_int4(weights_.out_embed, weights_.out_embed_scales,
                                         weights_.int4_group_size, lm_K);
            } else {
                lm_head_.set_params(weights_.out_embed);
            }
            for (size_t i = 0; i < layers_.size(); ++i)
                layers_[i].set_params_int4(&weights_, i);
            std::cerr << "[Qwen2] Weights distributed (W4A16 quantized mode, group_size="
                      << weights_.int4_group_size << ")" << std::endl;
        } else if (is_quantized) {
            if (weights_.out_embed_scales)
                lm_head_.set_params_quantized(weights_.out_embed, weights_.out_embed_scales);
            else
                lm_head_.set_params(weights_.out_embed);
            for (size_t i = 0; i < layers_.size(); ++i)
                layers_[i].set_params_quantized(&weights_, i);
            std::cerr << "[Qwen2] Weights distributed (W8A16 quantized mode)" << std::endl;
        } else {
            lm_head_.set_params(weights_.out_embed);
            for (size_t i = 0; i < layers_.size(); ++i)
                layers_[i].set_params(&weights_, i);
        }

        distribute_norm_tensors();
    }
};

struct LlaisysQwen2Model { std::unique_ptr<Qwen2Model> impl; };

extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
    int *device_ids, int ndevice) {
    auto model = new LlaisysQwen2Model();
    model->impl = std::make_unique<Qwen2Model>(meta, device, device_ids ? device_ids[0] : 0);
    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) delete model;
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->impl) return nullptr;
    return model->impl->get_weights_struct();
}

__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    if (!model || !model->impl) return 0;
    return model->impl->infer(token_ids, ntoken);
}

__export int64_t llaisysQwen2ModelInferEx(
    struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken,
    float temperature, int top_k, float top_p, uint64_t seed) {
    if (!model || !model->impl) return 0;
    return model->impl->infer_ex(token_ids, ntoken, temperature, top_k, top_p, seed);
}

__export void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    if (model && model->impl) model->impl->reset();
}

} // extern "C"
