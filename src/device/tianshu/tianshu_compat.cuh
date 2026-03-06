/**
 * @file tianshu_compat.cuh
 * @brief 天数智芯 TOPSRIDER 兼容层头文件
 *
 * 天数智芯 TOPSRIDER SDK 提供 CUDA 兼容 API，topscc 编译器可编译标准 .cu 文件。
 * 本头文件提供统一的错误检查宏和兼容性定义。
 *
 * 【TOPSRIDER SDK 与 CUDA 的映射关系】
 *   - topscc 编译器 ↔ nvcc 编译器
 *   - cuda_runtime.h → TOPSRIDER 兼容头文件（topscc 自动映射）
 *   - cudaMalloc/cudaFree → TOPSRIDER 兼容 API（topscc 编译期映射）
 *   - cublas_v2.h → TOPSRIDER BLAS 兼容头文件
 *   - libtopsrt.so ↔ libcudart.so
 *   - libtopsblas.so ↔ libcublas.so
 *
 * 【已知差异与注意事项】
 *   1. Warp Size: BI-150 GCU 架构可能使用不同于 NVIDIA 32 的 warp size
 *      → 所有 __shfl_down_sync 的 delta 范围和 warp 归约轮数需适配
 *      → 共享内存中 warp buffer 大小需调整
 *   2. SM 数量: 通过 cudaDeviceGetAttribute 查询（兼容 API）
 *   3. FP16/BF16: TOPSRIDER 提供 cuda_fp16.h/cuda_bf16.h 兼容头文件
 *      → __hadd2、__half2float 等 intrinsic 可直接使用
 *   4. cuBLAS API: TOPSRIDER 的 topsblas 提供 cuBLAS 兼容 API
 *      → cublasCreate/cublasSgemm/cublasGemmEx 等可直接调用
 *
 * 【编译环境要求】
 *   - TOPSRIDER SDK >= 3.0（建议最新版本）
 *   - 设置环境变量 TOPS_HOME 指向 SDK 安装目录
 *   - 使用 topscc 编译器（TOPS_HOME/bin/topscc）
 */

#pragma once

// TOPSRIDER SDK 提供 CUDA 兼容头文件，topscc 编译器自动解析
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::tianshu {

/**
 * TOPSRIDER 错误检查工具函数
 * topscc 编译器将 cudaError_t 等类型映射到 TOPSRIDER 对应类型
 */
inline void checkTops(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[TOPS ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

/**
 * 【BI-150 GCU Warp 大小适配】
 *
 * 天数智芯 BI-150 的 warp size 需在实机上确认。
 * 以下常量在部署到实际硬件时需要根据 warpSize 调整。
 *
 * 当前假设 warp size = 32（与 NVIDIA 一致），
 * 如果 BI-150 实际 warp size 不同，需要修改以下内容：
 *   1. WARP_SIZE 常量
 *   2. __shfl_down_sync 的归约轮数
 *   3. 共享内存中的 warp buffer 大小
 */
constexpr int WARP_SIZE = 32;

} // namespace llaisys::device::tianshu
