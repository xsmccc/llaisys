#pragma once

#include "llaisys/runtime.h"

#include "../storage/storage.hpp"

namespace llaisys::core {
    // 虚基类
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;  // 指向对应设备的 API CPU/NVIDIA
    MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    // 定义标准接口，所有具体allocator都要继承并实现这两个虚函数
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;   //纯虚函数：内存分配
    virtual void release(std::byte *memory) = 0;    //纯虚函数：内存释放
};

} // namespace llaisys::core
/*
分离职责：Runtime 负责流和激活，Allocator 负责内存管理
    后续可实现内存池或者显存池等更复杂的分配器
*/
