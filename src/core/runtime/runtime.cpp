#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"

namespace llaisys::core {

    // Runtime 构造函数，初始化成员变量并创建流和分配器
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    // 获取对应设备的API
    _api = llaisys::device::getRuntimeAPI(_device_type);
    //    ├─ CPU → 返回 cpu_runtime_api
    //    └─ NVIDIA → 返回 nvidia_runtime_api

    // 从该API中创建异步流
    _stream = _api->create_stream();
    //    ├─ CPU: 返回 nullptr
    //    └─ NVIDIA: 返回真实的 cudaStream_t


    // 创建内存分配器（这里使用朴素分配器）
    _allocator = new allocators::NaiveAllocator(_api);
}

// 析构函数，释放分配器和流
Runtime::~Runtime() {
    // Runtime 被销毁时必须处于激活状态（防止误操作）
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
}

// CPU没有此概念，GPU设备有当前激活设备的概念
// 激活设备
void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}


//失活设备
void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

// 分配设备内存（GPU 显存或 CPU RAM）
storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(
        new Storage(
            _allocator->allocate(size), // 调用 allocator
             size, 
             *this,     // 传入 this Runtime 指针
             false      // is_host = false（设备内存）
        )
    );
}

// 分配主机内存（pinned 或普通 RAM）
storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(
        new Storage(
            (std::byte *)_api->malloc_host(size),   // 直接调 API
             size, 
             *this, 
             true       // is_host = true（主机内存）
        )
    );
}

// 根据设备类型释放内存
void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

//  返回该设备的异步流
llaisysStream_t Runtime::stream() const {
    return _stream;
}

// 等待流上的所有任务完成
void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
