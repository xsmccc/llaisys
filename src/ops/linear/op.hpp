#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}

/*
Linear层实现——全连接层
Y = X * W_T + b
X——输入数据 [M,K],M行样本，K行长度
W——权重矩阵 Linear权重存储为[out_Features,In_Feartures]，因此需要转置一下
b——偏置向量 
Y——输出结果 [M,N]

输入: [M, K]
M: 有多少个 Token (Batch * Seq_Len)
K: 输入特征维度 (In_Features)。
权重: [N, K]
权重是转置存储的。
N: 输出特征维度 
K: 输入特征维度。
偏置: [N] 
每一个输出神经元对应一个偏置。
输出: [M, N]
结果的行数等于输入的行数，列数等于输出特征维度。
*/
