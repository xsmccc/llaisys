#pragma once
#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/linear_quantized/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include <vector>
#include <cstring>

// 实现基础构建块
namespace llaisys {

// 将void* 句柄转换为tensor_t*类型
static tensor_t cast_handle(void* handle) {
    if (!handle) return nullptr;
    return *reinterpret_cast<tensor_t*>(handle);
}

// input:  [seq_len] 
// weight: [vocab_size, hidden_size]
// output: [seq_len, hidden_size]
class Embedding {
public:
    Embedding() = default;

    void set_weight(void* w_handle) {
        weight_ = cast_handle(w_handle);
    }

    // 使用预分配输出张量的版本
    void forward(tensor_t output, tensor_t input) {
        if (!weight_) return;
        ops::embedding(output, input, weight_);
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;
        std::vector<size_t> out_shape = input->shape();
        out_shape.push_back(weight_->shape()[1]);
        auto out = Tensor::create(out_shape, weight_->dtype(), weight_->deviceType(), weight_->deviceId());
        ops::embedding(out, input, weight_);
        return out;
    }

private:
    tensor_t weight_;
};

// 归一化隐藏层状态 使数值稳定
class RMSNorm {
public:
    RMSNorm(float eps = 1e-6) : eps_(eps) {}

    void set_weight(void* w_handle) {
        weight_ = cast_handle(w_handle);
    }

    // 使用预分配输出张量
    void forward(tensor_t output, tensor_t input) {
        if (!weight_) return;
        ops::rms_norm(output, input, weight_, eps_);
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;
        auto out = Tensor::create(input->shape(), input->dtype(), input->deviceType(), input->deviceId());
        ops::rms_norm(out, input, weight_, eps_);
        return out;
    }

private:
    tensor_t weight_;
    float eps_;
};

// 线性变换层 (支持 F32 和 W8A32 量化)
class Linear {
public:
    Linear() = default;

    // 设置 F32 权重 (原有接口, 向后兼容)
    void set_params(void* w_handle, void* b_handle = nullptr) {
        weight_ = cast_handle(w_handle);
        if (b_handle) {
            bias_ = cast_handle(b_handle);
        }
        quantized_ = false;
    }

    // 设置 INT8 量化权重 + per-channel scales
    void set_params_quantized(void* w_handle, void* scales_handle,
                              void* b_handle = nullptr) {
        weight_ = cast_handle(w_handle);
        scales_ = cast_handle(scales_handle);
        if (b_handle) {
            bias_ = cast_handle(b_handle);
        }
        quantized_ = (weight_ && scales_);
    }

    // 使用预分配输出张量
    void forward(tensor_t output, tensor_t input) {
        if (!weight_) return;
        if (quantized_ && scales_) {
            ops::linear_quantized(output, input, weight_, scales_, bias_);
        } else {
            ops::linear(output, input, weight_, bias_);
        }
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;
        std::vector<size_t> out_shape = input->shape();
        out_shape.back() = weight_->shape()[0];
        auto out = Tensor::create(out_shape, input->dtype(), input->deviceType(), input->deviceId());
        forward(out, input);
        return out;
    }

    size_t out_features() const {
        return weight_ ? weight_->shape()[0] : 0;
    }

    bool is_quantized() const { return quantized_; }

private:
    tensor_t weight_;
    tensor_t bias_;
    tensor_t scales_;     // per-channel F32 scales for W8A32
    bool quantized_ = false;
};

} // namespace llaisys
