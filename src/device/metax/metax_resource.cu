/**
 * @file metax_resource.cu
 * @brief MetaX MACA BLAS 资源管理实现
 */
#include "metax_resource.cuh"
#include <iostream>

namespace llaisys::device::metax {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_METAX, device_id) {
    cublasStatus_t status = cublasCreate(&_blas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[MACA BLAS ERROR] Failed to create BLAS handle" << std::endl;
    }
}

Resource::~Resource() {
    if (_blas_handle) {
        cublasDestroy(_blas_handle);
    }
}

} // namespace llaisys::device::metax
