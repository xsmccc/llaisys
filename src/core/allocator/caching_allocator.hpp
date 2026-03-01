#pragma once

#include "allocator.hpp"

#include <map>
#include <vector>
#include <mutex>

namespace llaisys::core::allocators {

// 缓存内存分配器：避免反复调用 cudaMalloc/cudaFree
// 原理：释放时不真正 free，而是放入 free list；分配时优先从 free list 取
class CachingAllocator : public MemoryAllocator {
public:
    CachingAllocator(const LlaisysRuntimeAPI *runtime_api);
    ~CachingAllocator();

    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;

private:
    // free list: size -> vector of available blocks
    // 使用 multimap 使得可以找到 >= 请求大小的最小空闲块
    std::multimap<size_t, std::byte*> free_blocks_;

    // 记录每个已分配指针对应的实际分配大小
    std::map<std::byte*, size_t> allocated_sizes_;

    std::mutex mutex_;

    // 统计信息
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
};

} // namespace llaisys::core::allocators
