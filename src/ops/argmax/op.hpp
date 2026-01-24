#pragma once

#include "../../tensor/tensor.hpp" //定义张量需要

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
