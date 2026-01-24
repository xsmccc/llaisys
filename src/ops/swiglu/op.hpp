#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
/*
SwiGLU是llama等在FFN层做的改进
SwiGLU是Swish+GLU，其中GLU是门控线性单元
UP-实际的信息流数据
Gate-控制水龙头开合程度的信号
GLU原理-out=up*Sigmoid（gate）
Swish是一种激活函数，公式x*σ（x），比ReLU更平滑
*/
