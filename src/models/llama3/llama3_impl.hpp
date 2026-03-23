#pragma once
#include <cstddef>
#include <cmath>
#include <llaisys/models/llama3.h>

namespace llaisys {

struct Llama3Config {
    size_t vocab_size;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t max_position_embeddings;
    size_t head_dim;
    float rms_norm_eps;
    float rope_theta;
    int64_t end_token_id;
    bool tie_embeddings;
    llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    Llama3Config() = default;

    Llama3Config(const LlaisysLlama3Meta& meta,
                 llaisysDeviceType_t dev = LLAISYS_DEVICE_CPU,
                 int dev_id = 0) {
        vocab_size = meta.voc;
        hidden_size = meta.hs;
        intermediate_size = meta.di;
        num_hidden_layers = meta.nlayer;
        num_attention_heads = meta.nh;
        num_key_value_heads = meta.nkvh;
        max_position_embeddings = meta.maxseq;
        head_dim = meta.dh;
        rms_norm_eps = meta.epsilon;
        rope_theta = meta.theta;
        end_token_id = meta.end_token;
        tie_embeddings = (meta.tie_embeddings != 0);
        device_type = dev;
        device_id = dev_id;
    }

    size_t kv_dim() const {
        return num_key_value_heads * head_dim;
    }
};

} // namespace llaisys
