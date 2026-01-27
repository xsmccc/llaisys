#pragma once
#include <cstddef>
#include <cmath>
#include <llaisys/models/qwen2.h>

namespace llaisys {

struct Qwen2Config {
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

    Qwen2Config() = default;

    Qwen2Config(const LlaisysQwen2Meta& meta) {
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
    }

    size_t kv_dim() const {
        return num_key_value_heads * head_dim;
    }
};

} // namespace llaisys
