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

    // 禁止拷贝
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // 禁止移动
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    // 切换设备函数
    void setDevice(llaisysDeviceType_t device_type, int device_id);
    
    //获取当前激活的Runtime
    Runtime &runtime();

    friend Context &context();
};
} // namespace llaisys::core
/*
创建和管理所有设备的Runtime对象
切换设备
维护当前激活设备的Runtime

*/
