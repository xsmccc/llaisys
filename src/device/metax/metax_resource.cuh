#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>  // MACA 兼容头文件（mxcc 映射到 macablas）

namespace llaisys::device::metax {

/**
 * MetaX MACA 设备资源管理
 * 管理 macaBLAS handle 的生命周期
 * API 与 cuBLAS 完全兼容（MACA SDK 提供兼容层）
 */
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _blas_handle;  // MACA 兼容 cuBLAS handle

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t getBlasHandle() const { return _blas_handle; }
};

} // namespace llaisys::device::metax
