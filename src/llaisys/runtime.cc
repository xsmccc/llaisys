#include "llaisys/runtime.h"
#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"
/*
C API包装层
*/


// Llaisys API for setting context runtime.
// 设置当前执行设备（CPU/NVIDIA）
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::core::context().setDevice(device_type, device_id);
}

// Llaisys API for getting the runtime APIs
// 获取指定设备的运行时 API 对象（包含内存分配、流管理、内存拷贝等）
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::device::getRuntimeAPI(device_type);
}