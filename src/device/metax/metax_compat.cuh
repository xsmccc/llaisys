/**
 * @file metax_compat.cuh
 * @brief MetaX MACA 兼容层头文件
 *
 * 沐曦 MACA SDK 提供 CUDA 兼容 API，mxcc 编译器可编译标准 .cu 文件。
 * 本头文件提供统一的错误检查宏和兼容性定义。
 *
 * 【MACA SDK 与 CUDA 的映射关系】
 *   - mxcc 编译器 ↔ nvcc 编译器
 *   - cuda_runtime.h → MACA 兼容头文件（mxcc 自动映射）
 *   - cudaMalloc/cudaFree → MACA 兼容 API（mxcc 编译期映射）
 *   - cublas_v2.h → MACA BLAS 兼容头文件
 *   - libmacart.so ↔ libcudart.so
 *   - libmacablas.so ↔ libcublas.so
 *
 * 【已知差异与注意事项】
 *   1. Warp Size: C500 使用 128 线程/warp（不同于 NVIDIA 的 32）
 *      → 所有 __shfl_down_sync 的 delta 范围和 warp 归约轮数需适配
 *      → 共享内存中 warp buffer 大小需调整
 *   2. SM 数量: 通过 cudaDeviceGetAttribute 查询（兼容 API）
 *   3. FP16/BF16: MACA 提供 cuda_fp16.h/cuda_bf16.h 兼容头文件
 *      → __hadd2、__half2float 等 intrinsic 可直接使用
 *   4. cuBLAS API: MACA 的 macablas 提供 cuBLAS 兼容 API
 *      → cublasCreate/cublasSgemm/cublasGemmEx 等可直接调用
 *
 * 【编译环境要求】
 *   - MACA SDK >= 2.0（建议最新版本）
 *   - 设置环境变量 MACA_PATH 指向 SDK 安装目录
 *   - 使用 mxcc 编译器（MACA_PATH/bin/mxcc）
 */

#pragma once

// MACA SDK 提供 CUDA 兼容头文件，mxcc 编译器自动解析
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::metax {

/**
 * MACA 错误检查工具函数
 * mxcc 编译器将 cudaError_t 等类型映射到 MACA 对应类型
 */
inline void checkMaca(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[MACA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

/**
 * 【MACA C500 Warp 大小适配】
 *
 * 沐曦 C500 的 warp size 可能为 128（而非 NVIDIA 的 32）。
 * 以下常量和函数在部署到实际硬件时需要根据 warpSize 调整。
 *
 * 当前实现使用 NVIDIA 兼容的 warp size = 32 作为默认值，
 * 首次部署到 C500 时需通过 cudaDeviceGetAttribute 查询实际 warpSize 验证。
 *
 * TODO(C500部署): 验证 warpSize 并调整以下归约函数
 */
constexpr int WARP_SIZE = 32;  // 默认值，部署时需验证

} // namespace llaisys::device::metax
