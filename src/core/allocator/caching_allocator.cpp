#include "caching_allocator.hpp"

#include <iostream>

namespace llaisys::core::allocators {

CachingAllocator::CachingAllocator(const LlaisysRuntimeAPI *runtime_api)
    : MemoryAllocator(runtime_api) {}

CachingAllocator::~CachingAllocator() {
    // 打印缓存统计信息
    std::cerr << "[CachingAllocator] Stats: hits=" << cache_hits_ 
              << " misses=" << cache_misses_ 
              << " cached_blocks=" << free_blocks_.size() << std::endl;
    // 析构时真正释放所有缓存的内存块
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &[size, ptr] : free_blocks_) {
        _api->free_device(ptr);
    }
    free_blocks_.clear();
    allocated_sizes_.clear();
}

// best-fit算法
std::byte *CachingAllocator::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 查找 free_blocks_ 中 >= size 的最小空闲块
    auto it = free_blocks_.lower_bound(size);

    if (it != free_blocks_.end()) {
        // 找到了可复用的块
        // 只接受不超过 2 倍大小的块，避免浪费过多内存
        if (it->first <= size * 2) {
            std::byte *ptr = it->second;
            size_t actual_size = it->first;
            free_blocks_.erase(it);
            allocated_sizes_[ptr] = actual_size;
            current_cache_bytes_ -= actual_size;
            cache_hits_++;
            return ptr;
        }
    }

    // 没有合适的空闲块，调用底层 API 分配
    std::byte *ptr = static_cast<std::byte *>(_api->malloc_device(size));
    allocated_sizes_[ptr] = size;
    cache_misses_++;
    return ptr;
}

void CachingAllocator::release(std::byte *memory) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_sizes_.find(memory);
    if (it == allocated_sizes_.end()) {
        // 未知指针，直接释放
        _api->free_device(memory);
        return;
    }

    size_t size = it->second;
    allocated_sizes_.erase(it);

    // 超过缓存上限时直接释放，避免 OOM
    if (current_cache_bytes_ + size > MAX_CACHE_BYTES) {
        _api->free_device(memory);
        return;
    }

    // 放入空闲列表而不是真正释放
    free_blocks_.emplace(size, memory);
    current_cache_bytes_ += size;
}

CachingAllocator::Stats CachingAllocator::stats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
    Stats s;
    s.cache_hits = cache_hits_;
    s.cache_misses = cache_misses_;
    s.cached_blocks = free_blocks_.size();
    s.cache_bytes = current_cache_bytes_;
    s.active_blocks = allocated_sizes_.size();
    s.active_bytes = 0;
    for (auto& [ptr, sz] : allocated_sizes_) s.active_bytes += sz;
    return s;
}

} // namespace llaisys::core::allocators
