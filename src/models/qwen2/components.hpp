#pragma once
#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
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

    
    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;

        // 计算输出维度和形状
        std::vector<size_t> out_shape = input->shape();
        out_shape.push_back(weight_->shape()[1]);

        auto out = Tensor::create(
            out_shape,
            weight_->dtype(),
            weight_->deviceType(),
            weight_->deviceId()
        );

        ops::embedding(out, input, weight_);
        return out;
    }

private:
    tensor_t weight_;
};

// 归一化隐藏层状态 使数值稳定
// 输入：[seq_len, hidden_size]
// 权重：[hidden_size]
// 输出：[seq_len, hidden_size]
class RMSNorm {
public:
    RMSNorm(float eps = 1e-6) : eps_(eps) {}

    // 获得归一化权重
    void set_weight(void* w_handle) {
        weight_ = cast_handle(w_handle);
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;

        auto out = Tensor::create(
            input->shape(),
            input->dtype(),
            input->deviceType(),
            input->deviceId()
        );

        ops::rms_norm(out, input, weight_, eps_);
        return out;
    }

private:
    tensor_t weight_;
    float eps_; // 防止除以0
};

// 线性变换层
// 输入：[seq_len, in_features]
// 权重：[out_features, in_features]
// 偏置：[out_features]
// 输出：[seq_len, out_features]
class Linear {
public:
    Linear() = default;

    void set_params(void* w_handle, void* b_handle = nullptr) {
        weight_ = cast_handle(w_handle);
        if (b_handle) { // 偏置可能存在
            bias_ = cast_handle(b_handle);
        }
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;

        std::vector<size_t> out_shape = input->shape();
        out_shape.back() = weight_->shape()[0];

        auto out = Tensor::create(
            out_shape,
            input->dtype(),
            input->deviceType(),
            input->deviceId()
        );

        ops::linear(out, input, weight_, bias_);
        return out;
    }

private:
    tensor_t weight_;  // [out_features, in_features]
    tensor_t bias_;    // [out_features]
};

} // namespace llaisys
