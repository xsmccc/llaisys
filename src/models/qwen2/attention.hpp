#pragma once
#include "components.hpp"
#include "qwen2_impl.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include <cmath>

namespace llaisys {

class Qwen2Attention {
public:
    Qwen2Attention(const Qwen2Config& config) : config_(config) {
        init_kv_cache();
    }

    void set_params(void* q_w, void* k_w, void* v_w, void* o_w,
                    void* q_b, void* k_b, void* v_b) {
        q_proj_.set_params(q_w, q_b);
        k_proj_.set_params(k_w, k_b);
        v_proj_.set_params(v_w, v_b);
        o_proj_.set_params(o_w);    //无偏置输出投影
    }

    tensor_t forward(tensor_t x, size_t pos) {
        // Q, K, V 投影
        // 将输入线性变换到Q,K,V空间
        auto q_2d = q_proj_.forward(x);
        auto k_2d = k_proj_.forward(x);
        auto v_2d = v_proj_.forward(x);

        // Reshape 为多头格式
        // 采用多头注意力可以从不同的角度看问题，得到更完善的表示
        std::vector<size_t> q_shape = {x->shape()[0], config_.num_attention_heads, config_.head_dim};
        std::vector<size_t> kv_shape = {x->shape()[0], config_.num_key_value_heads, config_.head_dim};

        auto q_3d = q_2d->view(q_shape);
        auto k_3d = k_2d->view(kv_shape);
        auto v_3d = v_2d->view(kv_shape);

        // RoPE 位置编码 创建位置张量
        std::vector<size_t> pos_shape = {x->shape()[0]};
        auto pos_tensor = Tensor::create(pos_shape, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        int64_t* pos_ptr = reinterpret_cast<int64_t*>(pos_tensor->data());
        // pos-表示当前token是序列中的第几个
        for (size_t i = 0; i < pos_shape[0]; i++) {
            pos_ptr[i] = static_cast<int64_t>(pos + i);
        }

        // 应用RoPE
        // 在高纬空间中对向量进行旋转，旋转角度取决于token的位置，从而将位置信息编码到向量中
        auto q_rope = Tensor::create(q_shape, q_3d->dtype(), q_3d->deviceType(), q_3d->deviceId());
        auto k_rope = Tensor::create(kv_shape, k_3d->dtype(), k_3d->deviceType(), k_3d->deviceId());

        ops::rope(q_rope, q_3d, pos_tensor, config_.rope_theta);
        ops::rope(k_rope, k_3d, pos_tensor, config_.rope_theta);

        // 更新KV缓存
        update_cache(k_cache_, k_rope, pos);
        update_cache(v_cache_, v_3d, pos);

        // 自注意力计算
        // 创建输出张量
        auto attn_out_3d = Tensor::create(
            q_shape, q_3d->dtype(), q_3d->deviceType(), q_3d->deviceId()
        );

        float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

        // 获取有效的缓存部分
        auto k_valid = k_cache_->slice(0, 0, pos + 1);
        auto v_valid = v_cache_->slice(0, 0, pos + 1);

        ops::self_attention(attn_out_3d, q_rope, k_valid, v_valid, scale);

        // Reshape 回2D
        std::vector<size_t> out_shape = {x->shape()[0], config_.hidden_size};
        auto attn_out_2d = attn_out_3d->view(out_shape);

        // 将注意力输出映射回隐层维度
        return o_proj_.forward(attn_out_2d);
    }

    // 重置缓存——对于需要生成第二个序列的时候用的
    void reset_cache() {
        // Reset cache for new sequence
        // Called at the start of generation
        cache_pos_ = 0;
    }

private:
    Qwen2Config config_;    //配置信息
    Linear q_proj_, k_proj_, v_proj_, o_proj_;  //4个投影层
    tensor_t k_cache_;  //KEY cache
    tensor_t v_cache_;  //Value Cache
    size_t cache_pos_ = 0; //缓存位置-未使用

    void init_kv_cache() {
        std::vector<size_t> shape = {
            config_.max_position_embeddings,//4096
            config_.num_key_value_heads,    //12
            config_.head_dim                //170
        };

        k_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
        v_cache_ = Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    }

    void update_cache(tensor_t cache, tensor_t update, size_t pos) {
        // Copy update into cache at position pos
        size_t bytes_per_elem = 4;  // F32
        if (update->dtype() == LLAISYS_DTYPE_F16) bytes_per_elem = 2;
        if (update->dtype() == LLAISYS_DTYPE_BF16) bytes_per_elem = 2;

        size_t row_size = config_.kv_dim() * bytes_per_elem;
        
        // 第pos行
        uint8_t* dst = reinterpret_cast<uint8_t*>(cache->data()) + row_size * pos;
        // update的数据
        uint8_t* src = reinterpret_cast<uint8_t*>(update->data());

        //内存复制进去
        std::memcpy(dst, src, row_size);
    }
};

} // namespace llaisys
