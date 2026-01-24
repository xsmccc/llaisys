#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}
/*
RoPE 通过旋转向量的角度注入位置信息
输入张量in的每一个向量——对应于pos_ids中的位置id
计算如下内容：
    xi = [ai,bi]为输入
    yi = [ai',bi']为i处的输出向量
    θ为固定基数（j = 0，1，...，d/2-1）
*/