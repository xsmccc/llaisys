#pragma once
#include <cstddef>
#include <cmath>
#include <llaisys/models/qwen2.h>

// 适配器模式-设计模式
namespace llaisys {

struct Qwen2Config {
    size_t vocab_size;                  //词表大小 151936
    size_t hidden_size;                 //隐藏层大小 1536
    size_t intermediate_size;           //MLP中间维度 8960
    size_t num_hidden_layers;           //层数 28层
    size_t num_attention_heads;         //注意力头数 12
    size_t num_key_value_heads;         //KV头数
    size_t max_position_embeddings;
    size_t head_dim;
    float rms_norm_eps;
    float rope_theta;
    int64_t end_token_id;

    Qwen2Config() = default;

    // 将C结构体转换为C++结构体
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
