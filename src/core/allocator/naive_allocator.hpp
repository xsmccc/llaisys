#pragma once

#include "allocator.hpp"

// 具体实现声明
namespace llaisys::core::allocators {
class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const LlaisysRuntimeAPI *runtime_api);
    ~NaiveAllocator() = default;
    std::byte *allocate(size_t size) override;  // 实现父类接口
    void release(std::byte *memory) override;   // 实现父类接口
};
} // namespace llaisys::core::allocators
/*
 内存管理——封装底层内存分配API
 实现接口MemoryAllocator

*/