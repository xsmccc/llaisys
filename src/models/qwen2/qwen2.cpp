#include "llaisys/models/qwen2.h"
#include "qwen2_impl.hpp"
#include "layer.hpp"
#include "components.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../device/runtime_api.hpp"
#include "../../core/llaisys_core.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

using namespace llaisys;

// ============ Random Sampling Implementation ============

/**
 * Sample a token from logits using temperature, top-k, top-p.
 *
 * Pipeline: temperature scaling → softmax → top-k filter → top-p filter → multinomial
 *
 * @param logits     CPU buffer of logits [vocab_size], will be modified in-place
 * @param vocab_size Number of vocabulary entries
 * @param temperature Temperature for scaling (>1 = more random, <1 = more deterministic)
 * @param top_k     Keep only top-k tokens (0 = disabled)
 * @param top_p     Keep tokens with cumulative probability <= top_p (1.0 = disabled)
 * @param seed      Random seed (0 = use random_device)
 */
static int64_t sample_token(float* logits, size_t vocab_size,
                            float temperature, int top_k, float top_p,
                            uint64_t seed) {
    // 1. Temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (size_t i = 0; i < vocab_size; ++i) {
            logits[i] *= inv_temp;
        }
    }

    // 2. Softmax (numerically stable: subtract max first)
    float max_logit = *std::max_element(logits, logits + vocab_size);
    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_logit);
        sum_exp += logits[i];
    }
    float inv_sum = static_cast<float>(1.0 / sum_exp);
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_sum;
    }
    // logits[] now contains probabilities

    // 3. Build sorted index array for top-k / top-p filtering
    std::vector<int64_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int64_t a, int64_t b) {
        return logits[a] > logits[b];
    });

    // 4. Top-K: keep only top_k tokens
    size_t cutoff = vocab_size;
    if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
        cutoff = static_cast<size_t>(top_k);
    }

    // 5. Top-P (nucleus): keep smallest set with cumulative prob >= top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        double cumsum = 0.0;
        for (size_t i = 0; i < cutoff; ++i) {
            cumsum += logits[indices[i]];
            if (cumsum >= static_cast<double>(top_p)) {
                cutoff = i + 1;
                break;
            }
        }
    }

    // 6. Zero out tokens beyond cutoff
    for (size_t i = cutoff; i < vocab_size; ++i) {
        logits[indices[i]] = 0.0f;
    }

    // 7. Renormalize
    double new_sum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        new_sum += logits[indices[i]];
    }
    if (new_sum > 0.0) {
        float inv_new_sum = static_cast<float>(1.0 / new_sum);
        for (size_t i = 0; i < cutoff; ++i) {
            logits[indices[i]] *= inv_new_sum;
        }
    }

    // 8. Multinomial sampling
    std::mt19937_64 rng(seed != 0 ? seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    double cumulative = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        cumulative += logits[indices[i]];
        if (r <= cumulative) {
            return indices[i];
        }
    }

    // Fallback: return the most probable token
    return indices[0];
}

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int device_id = 0)
        : config_(*meta, device, device_id),
          embed_(),
          final_norm_(config_.rms_norm_eps),
          lm_head_()
    {
        // Initialize weight arrays
        init_weight_arrays(config_.num_hidden_layers);

        // Create layers
        for (size_t i = 0; i < config_.num_hidden_layers; ++i) {
            layers_.emplace_back(config_);
        }

        // 预分配推理所需的工作空间张量
        init_inference_workspace();

        std::cerr << "[Qwen2] Model created with " << config_.num_hidden_layers 
                  << " layers, hidden_size=" << config_.hidden_size << std::endl;
    }

    ~Qwen2Model() {
        free_weight_arrays();
    }

    LlaisysQwen2Weights* get_weights_struct() {
        return &weights_;
    }

    int64_t infer(int64_t* token_ids, size_t ntoken) {
        // Distribute weights on first call only
        if (!weights_distributed_) {
            distribute_weights();
            weights_distributed_ = true;
        }

        if (!weights_.in_embed) {
            std::cerr << "[ERROR] Weights not loaded!" << std::endl;
            return 0;
        }

        int64_t output_token = 0;

        for (size_t i = 0; i < ntoken; ++i) {
            if (i % 10 == 0) {
                std::cerr << "\r[Qwen2] Token " << i << "/" << ntoken << std::flush;
            }

            // 1. 加载 token 到预分配张量
            int64_t token_val = token_ids[i];
            ws_token_->load(&token_val);

            // 2. Embedding
            embed_.forward(ws_hidden_, ws_token_);

            // 3. 加载 pos_tensor（每 token 只加载一次，所有层共享）
            int64_t pos_val = static_cast<int64_t>(current_pos_);
            ws_pos_->load(&pos_val);

            // 4. Forward through all layers（使用预分配工作空间 + 共享 pos_tensor）
            tensor_t current = ws_hidden_;
            for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
                current = layers_[layer_idx].forward(current, current_pos_, ws_pos_);
            }

            // 5. Final norm
            final_norm_.forward(ws_final_norm_, current);

            current_pos_++;

            // 6. LM Head prediction (only for last token)
            if (i == ntoken - 1) {
                lm_head_.forward(ws_logits_, ws_final_norm_);

                // Argmax
                ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);

                // 将结果从设备拷回主机（这里需要同步）
                if (config_.device_type != LLAISYS_DEVICE_CPU) {
                    const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
                    core::context().runtime().synchronize();
                    api->memcpy_sync(&output_token, ws_out_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
                } else {
                    output_token = *reinterpret_cast<int64_t*>(ws_out_idx_->data());
                }
            }
        }

        std::cerr << std::endl;
        return output_token;
    }

    /**
     * Extended inference with random sampling support.
     * Implements: temperature → softmax → top-k → top-p → multinomial
     * Sampling is performed on CPU for cross-platform compatibility.
     */
    int64_t infer_ex(int64_t* token_ids, size_t ntoken,
                     float temperature, int top_k, float top_p, uint64_t seed) {
        // Distribute weights on first call only
        if (!weights_distributed_) {
            distribute_weights();
            weights_distributed_ = true;
        }

        if (!weights_.in_embed) {
            std::cerr << "[ERROR] Weights not loaded!" << std::endl;
            return 0;
        }

        int64_t output_token = 0;

        for (size_t i = 0; i < ntoken; ++i) {
            if (i % 10 == 0) {
                std::cerr << "\r[Qwen2] Token " << i << "/" << ntoken << std::flush;
            }

            int64_t token_val = token_ids[i];
            ws_token_->load(&token_val);
            embed_.forward(ws_hidden_, ws_token_);

            int64_t pos_val = static_cast<int64_t>(current_pos_);
            ws_pos_->load(&pos_val);

            tensor_t current = ws_hidden_;
            for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
                current = layers_[layer_idx].forward(current, current_pos_, ws_pos_);
            }

            final_norm_.forward(ws_final_norm_, current);
            current_pos_++;

            if (i == ntoken - 1) {
                lm_head_.forward(ws_logits_, ws_final_norm_);

                // Check if greedy decoding (argmax) is sufficient
                bool is_greedy = (temperature <= 0.0f) ||
                                 (top_k == 1) ||
                                 (temperature == 1.0f && top_k <= 0 && top_p >= 1.0f);

                if (is_greedy && top_k == 1) {
                    // Use existing argmax for greedy decoding
                    ops::argmax(ws_out_idx_, ws_out_val_, ws_logits_);
                    if (config_.device_type != LLAISYS_DEVICE_CPU) {
                        const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
                        core::context().runtime().synchronize();
                        api->memcpy_sync(&output_token, ws_out_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
                    } else {
                        output_token = *reinterpret_cast<int64_t*>(ws_out_idx_->data());
                    }
                } else {
                    // Random sampling: copy logits to CPU, then sample
                    size_t vocab_size = config_.vocab_size;
                    std::vector<float> logits(vocab_size);

                    if (config_.device_type != LLAISYS_DEVICE_CPU) {
                        const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(config_.device_type);
                        core::context().runtime().synchronize();
                        api->memcpy_sync(logits.data(), ws_logits_->data(),
                                         vocab_size * sizeof(float), LLAISYS_MEMCPY_D2H);
                    } else {
                        std::memcpy(logits.data(), ws_logits_->data(),
                                    vocab_size * sizeof(float));
                    }

                    output_token = sample_token(logits.data(), vocab_size,
                                                temperature, top_k, top_p, seed);
                }
            }
        }

        std::cerr << std::endl;
        return output_token;
    }

    void reset() {
        current_pos_ = 0;
    }

private:
    Qwen2Config config_;
    LlaisysQwen2Weights weights_;

    Embedding embed_;
    std::vector<Qwen2DecoderLayer> layers_;
    RMSNorm final_norm_;
    Linear lm_head_;

    size_t current_pos_ = 0;
    bool weights_distributed_ = false;

    // 预分配的推理工作空间张量（decode 阶段 seq_len=1）
    tensor_t ws_token_;       // [1] I64
    tensor_t ws_pos_;         // [1] I64
    tensor_t ws_hidden_;      // [1, hidden_size] F32
    tensor_t ws_final_norm_;  // [1, hidden_size] F32
    tensor_t ws_logits_;      // [1, vocab_size] F32
    tensor_t ws_out_idx_;     // [1] I64
    tensor_t ws_out_val_;     // [1] F32

    void init_inference_workspace() {
        size_t hs = config_.hidden_size;
        size_t vs = config_.vocab_size;
        auto dt = config_.device_type;
        auto di = config_.device_id;

        ws_token_ = Tensor::create({1}, LLAISYS_DTYPE_I64, dt, di);
        ws_pos_ = Tensor::create({1}, LLAISYS_DTYPE_I64, dt, di);
        ws_hidden_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, dt, di);
        ws_final_norm_ = Tensor::create({1, hs}, LLAISYS_DTYPE_F32, dt, di);
        ws_logits_ = Tensor::create({1, vs}, LLAISYS_DTYPE_F32, dt, di);
        ws_out_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, dt, di);
        ws_out_val_ = Tensor::create({1}, LLAISYS_DTYPE_F32, dt, di);
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

        // INT8 量化 scales 数组
        weights_.quantized = 0;
        weights_.attn_q_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_k_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_v_w_scales = new llaisysTensor_t[nlayers];
        weights_.attn_o_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_gate_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_up_w_scales = new llaisysTensor_t[nlayers];
        weights_.mlp_down_w_scales = new llaisysTensor_t[nlayers];
        weights_.out_embed_scales = nullptr;

        // Initialize to null
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
        delete[] weights_.attn_q_w;
        delete[] weights_.attn_k_w;
        delete[] weights_.attn_v_w;
        delete[] weights_.attn_o_w;
        delete[] weights_.attn_q_b;
        delete[] weights_.attn_k_b;
        delete[] weights_.attn_v_b;
        delete[] weights_.attn_norm_w;
        delete[] weights_.mlp_norm_w;
        delete[] weights_.mlp_gate_w;
        delete[] weights_.mlp_up_w;
        delete[] weights_.mlp_down_w;
        // 量化 scales 数组
        delete[] weights_.attn_q_w_scales;
        delete[] weights_.attn_k_w_scales;
        delete[] weights_.attn_v_w_scales;
        delete[] weights_.attn_o_w_scales;
        delete[] weights_.mlp_gate_w_scales;
        delete[] weights_.mlp_up_w_scales;
        delete[] weights_.mlp_down_w_scales;
    }

    void distribute_weights() {
        embed_.set_weight(weights_.in_embed);
        final_norm_.set_weight(weights_.out_norm_w);

        bool is_quantized = (weights_.quantized != 0);

        if (is_quantized) {
            // LM head 量化路径
            if (weights_.out_embed_scales) {
                lm_head_.set_params_quantized(weights_.out_embed, weights_.out_embed_scales);
            } else {
                lm_head_.set_params(weights_.out_embed);
            }
            // 每层使用量化 set_params
            for (size_t i = 0; i < layers_.size(); ++i) {
                layers_[i].set_params_quantized(&weights_, i);
            }
            std::cerr << "[Qwen2] Weights distributed (INT8 quantized mode)" << std::endl;
        } else {
            // 原有 F32 路径
            lm_head_.set_params(weights_.out_embed);
            for (size_t i = 0; i < layers_.size(); ++i) {
                layers_[i].set_params(&weights_, i);
            }
        }
    }
};

struct LlaisysQwen2Model {
    std::unique_ptr<Qwen2Model> impl;
};


extern "C" {

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice)
{
    auto model = new LlaisysQwen2Model();
    model->impl = std::make_unique<Qwen2Model>(meta, device, device_ids ? device_ids[0] : 0);
    return model;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete model;
    }
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->impl) return nullptr;
    return model->impl->get_weights_struct();
}

__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken)
{
    if (!model || !model->impl) return 0;
    return model->impl->infer(token_ids, ntoken);
}

__export int64_t llaisysQwen2ModelInferEx(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken,
    float temperature,
    int top_k,
    float top_p,
    uint64_t seed)
{
    if (!model || !model->impl) return 0;
    return model->impl->infer_ex(token_ids, ntoken, temperature, top_k, top_p, seed);
}

__export void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    if (model && model->impl) {
        model->impl->reset();
    }
}

} // extern "C"
