/**
 * @file tianshu_resource.cu
 * @brief 天数智芯 TOPSRIDER BLAS 资源管理实现
 */
#include "tianshu_resource.cuh"
#include <iostream>

namespace llaisys::device::tianshu {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_TIANSHU, device_id) {
    cublasStatus_t status = cublasCreate(&_blas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[TOPS BLAS ERROR] Failed to create BLAS handle" << std::endl;
    }
}

Resource::~Resource() {
    if (_blas_handle) {
        cublasDestroy(_blas_handle);
    }
}

} // namespace llaisys::device::tianshu
