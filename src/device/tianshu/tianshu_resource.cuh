#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>  // TOPSRIDER 兼容头文件（topscc 映射到 topsblas）

namespace llaisys::device::tianshu {

/**
 * 天数智芯 TOPSRIDER 设备资源管理
 * 管理 topsBLAS handle 的生命周期
 * API 与 cuBLAS 完全兼容（TOPSRIDER SDK 提供兼容层）
 */
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _blas_handle;  // TOPSRIDER 兼容 cuBLAS handle

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t getBlasHandle() const { return _blas_handle; }
};

} // namespace llaisys::device::tianshu
