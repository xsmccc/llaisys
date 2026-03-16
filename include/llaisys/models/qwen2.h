#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
        // ── W8A32 量化 per-channel scales (对称 absmax) ──
        // 每个指针数组长度 = nlayer, 元素为 F32 张量 [out_features]
        // 当 quantized=0 时这些指针全为 NULL，不影响原有逻辑
        int quantized;                     // 0=F32, 1=INT8 量化模式
        llaisysTensor_t *attn_q_w_scales;  // [nlayer] Q 投影 scales
        llaisysTensor_t *attn_k_w_scales;  // [nlayer] K 投影 scales
        llaisysTensor_t *attn_v_w_scales;  // [nlayer] V 投影 scales
        llaisysTensor_t *attn_o_w_scales;  // [nlayer] O 投影 scales
        llaisysTensor_t *mlp_gate_w_scales;// [nlayer] Gate 投影 scales
        llaisysTensor_t *mlp_up_w_scales;  // [nlayer] Up 投影 scales
        llaisysTensor_t *mlp_down_w_scales;// [nlayer] Down 投影 scales
        llaisysTensor_t out_embed_scales;  // LM head scales (单个张量)

        // ── W4A32 INT4 group量化 ──
        // quantized=2 表示 INT4 模式
        // 权重为 U8 packed (2×int4 per byte), scales 为 F16 [N, num_groups]
        size_t int4_group_size;                // 量化组大小 (128)
        // K_orig arrays: 每层 7 个权重 (q,k,v,o, gate,up,down) + lm_head
        size_t *int4_K_orig;                   // [nlayer*7+1] 原始 K 维度
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    // Extended infer with sampling parameters
    __export int64_t llaisysQwen2ModelInferEx(
        struct LlaisysQwen2Model * model,
        int64_t * token_ids,
        size_t ntoken,
        float temperature,
        int top_k,
        float top_p,
        uint64_t seed
    );

    // Reset model state (KV cache position) for new conversation
    __export void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model);
}
#endif // LLAISYS_MODELS_QWEN2_H
