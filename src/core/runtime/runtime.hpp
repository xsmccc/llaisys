#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

namespace llaisys::core {

// 运行时环境管理器
class Runtime {
private:
    llaisysDeviceType_t _device_type;   //设备类型
    int _device_id;                     //设备id
    const LlaisysRuntimeAPI *_api;      //运行时API
    MemoryAllocator *_allocator;        //内存分配器
    bool _is_active;                    //是否为当前激活的Runtime
    void _activate();                   //激活
    void _deactivate();                 //失活
    llaisysStream_t _stream;            //异步流
    Runtime(llaisysDeviceType_t device_type, int device_id);

public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    ;
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
    void synchronize() const;
};
} // namespace llaisys::core
