#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas_handle;
public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t getCublasHandle() const { return _cublas_handle; }
};
} // namespace llaisys::device::nvidia
