#ifndef LLAISYS_MODELS_LLAMA3_H
#define LLAISYS_MODELS_LLAMA3_H

#include "../tensor.h"

__C {
    struct LlaisysLlama3Meta {
        llaisysDataType_t dtype;
        size_t nlayer;     // 层数 (16 for 1B)
        size_t hs;         // hidden_size (2048)
        size_t nh;         // num_attention_heads (32)
        size_t nkvh;       // num_key_value_heads (8, GQA 4:1)
        size_t dh;         // head_dim (64)
        size_t di;         // intermediate_size (8192)
        size_t maxseq;     // max KV cache length
        size_t voc;        // vocab_size (128256)
        float epsilon;     // rms_norm_eps (1e-5)
        float theta;       // rope_theta (500000.0)
        int64_t end_token; // eos_token_id
        int tie_embeddings; // 1=tied (LLaMA3), 0=separate
    };

    struct LlaisysLlama3Weights {
        llaisysTensor_t in_embed;      // model.embed_tokens.weight [vocab, hs]
        llaisysTensor_t out_embed;     // lm_head (=in_embed when tied)
        llaisysTensor_t out_norm_w;    // model.norm.weight [hs]

        // Per-layer attention (no bias in LLaMA3!)
        llaisysTensor_t *attn_norm_w;  // input_layernorm.weight
        llaisysTensor_t *attn_q_w;     // self_attn.q_proj.weight [hs, hs]
        llaisysTensor_t *attn_k_w;     // self_attn.k_proj.weight [kv_dim, hs]
        llaisysTensor_t *attn_v_w;     // self_attn.v_proj.weight [kv_dim, hs]
        llaisysTensor_t *attn_o_w;     // self_attn.o_proj.weight [hs, hs]

        // Per-layer MLP (no bias!)
        llaisysTensor_t *mlp_norm_w;   // post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;   // mlp.gate_proj.weight [di, hs]
        llaisysTensor_t *mlp_up_w;     // mlp.up_proj.weight [di, hs]
        llaisysTensor_t *mlp_down_w;   // mlp.down_proj.weight [hs, di]
    };

    struct LlaisysLlama3Model;

    __export struct LlaisysLlama3Model *llaisysLlama3ModelCreate(
        const LlaisysLlama3Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice);
    __export void llaisysLlama3ModelDestroy(struct LlaisysLlama3Model *model);
    __export struct LlaisysLlama3Weights *llaisysLlama3ModelWeights(
        struct LlaisysLlama3Model *model);
    __export int64_t llaisysLlama3ModelInfer(
        struct LlaisysLlama3Model *model,
        int64_t *token_ids,
        size_t ntoken);
    __export int64_t llaisysLlama3ModelInferEx(
        struct LlaisysLlama3Model *model,
        int64_t *token_ids,
        size_t ntoken,
        float temperature,
        int top_k,
        float top_p,
        uint64_t seed);
    __export void llaisysLlama3ModelReset(struct LlaisysLlama3Model *model);
}

#endif // LLAISYS_MODELS_LLAMA3_H
