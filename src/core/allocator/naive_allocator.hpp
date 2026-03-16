#pragma once

#include "allocator.hpp"

// 具体实现声明
// override关键字表示该函数重写了基类中的虚函数，编译器会检查函数签名是否匹配，如果不匹配会报错。这有助于避免由于函数签名错误而导致的意外行为。（C++11）
// override vs 不加 → 不加时函数签名写错会变成独立函数，不报错但不被调用
namespace llaisys::core::allocators { // 公有继承
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