#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

template<typename T>
void self_attention_kernel(
    T* attn_val,
    const T* q,
    const T* k,
    const T* v,
    size_t seq_len,
    size_t total_len,
    size_t nhead,
    size_t kv_head,
    size_t head_dim,
    size_t v_head_dim,
    float scale
){
    // 一个KV head负责group_szie个Qhead
    size_t group_size = nhead / kv_head;

    //之前一共有total_len个数据
    std::vector<float> scores(total_len);

    //遍历Q头
    for (size_t h = 0;h < nhead;h++)
    {
        // 当前Q头对应的kvhead
        size_t kv_h = h / group_size;

        // 遍历Q的每一个token
        for (size_t i = 0;i < seq_len;i++)
        {
            // 计算QK^T

            //当前Q的数据起始位置
            const T* q_ptr = q + i * nhead * head_dim + h * head_dim;

            // 遍历所有K 来得到当前Q与KVCache中过往的K的关系
            for (size_t t = 0;t < total_len;t++)
            {
                // 当前K的数据起始位置
                const T* k_ptr = k + t * kv_head * head_dim + kv_h * head_dim;
                
                //点积
                float dot = 0.0f;
                for (size_t d = 0;d < head_dim;d++)
                {
                    float q_val = llaisys::utils::cast<float>(q_ptr[d]);
                    float k_val = llaisys::utils::cast<float>(k_ptr[d]);
                    dot += q_val * k_val;
                }

                //获得注意力分数
                scores[t] = dot * scale;
            }
            //casual mask 掩码机制
            size_t current_pos = total_len - seq_len + i;
            
            //当前最大值初值赋值为负无穷
            float max_val = -std::numeric_limits<float>::infinity();

            // 开始对未知的token进行处理（因果掩码）
            for(size_t t = 0;t < total_len;t++){
                if (t > current_pos){
                    scores[t] = -std::numeric_limits<float>::infinity();
                }
                if (scores[t] > max_val)
                    max_val = scores[t];
            }
            float sum_exp = 0.0f;
            // 负无穷的为未来的token，此时其注意力分数为0
            for (size_t t = 0;t < total_len;t++){
                if (scores[t] == -std::numeric_limits<float>::infinity()){
                    scores[t] = 0.0f;
                }else{
                    scores[t] = std::exp(scores[t] - max_val);//safesoftmax
                }
                sum_exp += scores[t];
            }
            // 获得当前token对之前每一个token的关系的注意力分数
            for (size_t t = 0;t < total_len;t++){
                scores[t] /= sum_exp;
            }

            //输出坐标索引
            T* out_ptr = attn_val + i * nhead * v_head_dim + h * v_head_dim;

            // 遍历 V 的特征维度
            for (size_t dv = 0; dv < v_head_dim; dv++) {
                float val = 0.0f;
                
                // 遍历所有历史token
                for (size_t t = 0; t < total_len; t++) {
                    // 获取V的读取位置
                    const T* v_ptr = v + t * kv_head * v_head_dim + kv_h * v_head_dim;
                    
                    // 累加 (概率 * 值)得到新的val
                    val += scores[t] * llaisys::utils::cast<float>(v_ptr[dv]);
                }
                
                // 存入结果
                out_ptr[dv] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu{
    void self_attention(
        std::byte* attn_val_ptr,
        llaisysDataType_t attn_val_type,
        const std::byte* q,
        const std::byte* k,
        const std::byte* v,
        size_t seq_len,
        size_t total_len,
        size_t nhead,
        size_t kv_head,
        size_t head_dim,
        size_t v_head_dim,
        float scale
    ){
        switch(attn_val_type){
            case LLAISYS_DTYPE_F32:
            return self_attention_kernel(
            reinterpret_cast<float*>(attn_val_ptr),
            reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v),
            seq_len,
            total_len,
            nhead,
            kv_head,
            head_dim,
            v_head_dim,
            scale
            );
            case LLAISYS_DTYPE_BF16:
            return self_attention_kernel(
            reinterpret_cast<llaisys::bf16_t*>(attn_val_ptr),
            reinterpret_cast<const llaisys::bf16_t*>(q),
            reinterpret_cast<const llaisys::bf16_t*>(k),
            reinterpret_cast<const llaisys::bf16_t*>(v),
            seq_len,
            total_len,
            nhead,
            kv_head,
            head_dim,
            v_head_dim,
            scale
            );
            case LLAISYS_DTYPE_F16:
            return self_attention_kernel(
            reinterpret_cast<llaisys::fp16_t*>(attn_val_ptr),
            reinterpret_cast<const llaisys::fp16_t*>(q),
            reinterpret_cast<const llaisys::fp16_t*>(k),
            reinterpret_cast<const llaisys::fp16_t*>(v),
            seq_len,
            total_len,
            nhead,
            kv_head,
            head_dim,
            v_head_dim,
            scale
            );
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(attn_val_type);
        }    
    }
}
