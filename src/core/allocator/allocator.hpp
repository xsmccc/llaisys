#pragma once

#include "llaisys/runtime.h"

#include "../storage/storage.hpp"

namespace llaisys::core {
    // 虚基类
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;  // 指向对应设备的 API
    MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;   //内存分配
    virtual void release(std::byte *memory) = 0;    //内存释放
};

} // namespace llaisys::core
