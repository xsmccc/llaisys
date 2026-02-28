#include "nvidia_resource.cuh"
#include <iostream>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cublasStatus_t status = cublasCreate(&_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[CUBLAS ERROR] Failed to create cuBLAS handle" << std::endl;
    }
}

Resource::~Resource() {
    if (_cublas_handle) {
        cublasDestroy(_cublas_handle);
    }
}

} // namespace llaisys::device::nvidia
