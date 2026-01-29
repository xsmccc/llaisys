#pragma once

#include "llaisys.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <unordered_map>
#include <vector>

namespace llaisys::core {
class Context {
private:
    // 所有设备的 Runtime 对象池
    std::unordered_map<llaisysDeviceType_t, std::vector<Runtime *>> _runtime_map;
    // ├─ Key: 设备类型（CPU、NVIDIA）
    // └─ Value: 该设备类型的所有 Runtime 对象（支持多 GPU）

    // 当前激活的 Runtime
    Runtime *_current_runtime;
    
    Context();

public:
    ~Context();

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void setDevice(llaisysDeviceType_t device_type, int device_id);
    Runtime &runtime();

    friend Context &context();
};
} // namespace llaisys::core
