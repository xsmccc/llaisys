#pragma once
#include "../../tensor/tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include <vector>
#include <cstring>

namespace llaisys {

static tensor_t cast_handle(void* handle) {
    if (!handle) return nullptr;
    return *reinterpret_cast<tensor_t*>(handle);
}

class Embedding {
public:
    Embedding() = default;

    void set_weight(void* w_handle) {
        weight_ = cast_handle(w_handle);
    }

    tensor_t forward(tensor_t input) {
        if (!weight_) return nullptr;

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

class RMSNorm {
public:
    RMSNorm(float eps = 1e-6) : eps_(eps) {}

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
    float eps_;
};

class Linear {
public:
    Linear() = default;

    void set_params(void* w_handle, void* b_handle = nullptr) {
        weight_ = cast_handle(w_handle);
        if (b_handle) {
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
    tensor_t weight_;
    tensor_t bias_;
};

} // namespace llaisys
