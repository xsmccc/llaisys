#include "naive_allocator.hpp"

#include "../runtime/runtime.hpp"

// 内存分配器：朴素实现，直接调用运行时 API 分配和释放内存
// 子类构造函数必须在初始化列表中调用基类构造函数
// 因为基类的构造函数是 protected，所以只有子类能调
namespace llaisys::core::allocators {
NaiveAllocator::NaiveAllocator(const LlaisysRuntimeAPI *runtime_api) : MemoryAllocator(runtime_api) { // 调用基类构造函数
}

// 分配 → 调用设备的 malloc_device
std::byte *NaiveAllocator::allocate(size_t size) {
    // static_cast 是 C++ 中的一个类型转换运算符，用于在编译时进行类型转换。它比 C 风格的强制类型转换更安全，因为它会检查类型之间的兼容性。
    return static_cast<std::byte *>(_api->malloc_device(size));
    //                              ^^^^^^ 这个 _api 继承自父类
    //                                    指向 CPU 或 NVIDIA 的 malloc_device
}

// 释放 → 调用设备的 free_device
void NaiveAllocator::release(std::byte *memory) {
    _api->free_device(memory);
}
} // namespace llaisys::core::allocators