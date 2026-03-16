#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {   // 嵌套命名空间
class Storage {
private:
    std::byte *_memory;           // 原始内存地址 原始字节
    size_t _size;                 // 字节数
    Runtime &_runtime;            // 内存归属的 Runtime（CPU/GPU）
    bool _is_host;                // true=主机内存，false=设备内存
    // 私有构造 → 只能由 Runtime 创建 工厂模式
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;   //Runtime为友元，即可访问构造Storage
    ~Storage();

    std::byte *memory() const;  // 获取内存首地址
    size_t size() const;        // 获取大小
    llaisysDeviceType_t deviceType() const; // 所属设备类型（CPU/NVIDIA）
    int deviceId() const;       // 设备 ID
    bool isHost() const;        // 是否主机内存
};

}; // namespace llaisys::core
