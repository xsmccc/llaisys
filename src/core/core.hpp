#pragma once
#include <memory>

namespace llaisys {
namespace core {
class Storage;
using storage_t = std::shared_ptr<Storage>; //引用计数指针 多张量可共享内存

class MemoryAllocator;

class Runtime;
class Context;

// Global function to get thread local context
Context &context();
} // namespace core

} // namespace llaisys