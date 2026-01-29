#include "naive_allocator.hpp"

#include "../runtime/runtime.hpp"

// 内存分配器：朴素实现，直接调用运行时 API 分配和释放内存
namespace llaisys::core::allocators {
NaiveAllocator::NaiveAllocator(const LlaisysRuntimeAPI *runtime_api) : MemoryAllocator(runtime_api) {
}

// 分配 → 调用设备的 malloc_device
std::byte *NaiveAllocator::allocate(size_t size) {
    return static_cast<std::byte *>(_api->malloc_device(size));
}

// 释放 → 调用设备的 free_device
void NaiveAllocator::release(std::byte *memory) {
    _api->free_device(memory);
}
} // namespace llaisys::core::allocators