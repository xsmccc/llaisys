#include "llaisys/models/qwen2.h"
#include "qwen2_impl.hpp"
#include "layer.hpp"
#include "components.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>

using namespace llaisys;

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta* meta)
        : config_(*meta),
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
        // Distribute weights on first call
        distribute_weights();

        if (!weights_.in_embed) {
            std::cerr << "[ERROR] Weights not loaded!" << std::endl;
            return 0;
        }

        int64_t output_token = 0;

        for (size_t i = 0; i < ntoken; ++i) {
            if (i % 10 == 0) {
                std::cerr << "\r[Qwen2] Token " << i << "/" << ntoken << std::flush;
            }

            // 1. Embedding
            std::vector<size_t> token_shape = {1};
            auto token_tensor = Tensor::create(token_shape, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            *reinterpret_cast<int64_t*>(token_tensor->data()) = token_ids[i];

            auto hidden_state = embed_.forward(token_tensor);

            // 2. Forward through all layers
            for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
                hidden_state = layers_[layer_idx].forward(hidden_state, current_pos_);
            }

            // 3. Final norm
            hidden_state = final_norm_.forward(hidden_state);

            current_pos_++;

            // 4. LM Head prediction (only for last token)
            if (i == ntoken - 1) {
                auto logits = lm_head_.forward(hidden_state);

                // Argmax to get next token
                auto out_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
                auto out_val = Tensor::create({1}, logits->dtype(), LLAISYS_DEVICE_CPU, 0);

                ops::argmax(out_idx, out_val, logits);

                output_token = *reinterpret_cast<int64_t*>(out_idx->data());
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
    }

    void distribute_weights() {
        embed_.set_weight(weights_.in_embed);
        final_norm_.set_weight(weights_.out_norm_w);
        lm_head_.set_params(weights_.out_embed);

        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_[i].set_params(&weights_, i);
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
    model->impl = std::make_unique<Qwen2Model>(meta);
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

} // extern "C"
