#ifndef LLAISYS_RUNTIME_H
#define LLAISYS_RUNTIME_H

#include "../llaisys.h"

__C {
    // Runtime API Functions
    // Device
    typedef int (*get_device_count_api)();
    typedef void (*set_device_api)(int);
    typedef void (*device_synchronize_api)();
    // Stream
    typedef llaisysStream_t (*create_stream_api)();
    typedef void (*destroy_stream_api)(llaisysStream_t);
    typedef void (*stream_synchronize_api)(llaisysStream_t);
    // Memory
    typedef void *(*malloc_device_api)(size_t);
    typedef void (*free_device_api)(void *);
    typedef void *(*malloc_host_api)(size_t);
    typedef void (*free_host_api)(void *);
    // Memory copy
    typedef void (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);
    typedef void (*memcpy_async_api)(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t);

    // 抽象不同硬件设备的底层API接口（如CPU/GPU）
    struct LlaisysRuntimeAPI {
    // ============ 设备管理 ============
    get_device_count_api get_device_count;        // 查询有几个设备
    set_device_api set_device;                     // 切换到某个设备
    device_synchronize_api device_synchronize;    // 等待设备上的所有任务完成
    
    // ============ 流管理（异步执行队列） ============
    create_stream_api create_stream;               // 创建异步执行流
    destroy_stream_api destroy_stream;             // 销毁流
    stream_synchronize_api stream_synchronize;    // 等待特定流完成
    
    // ============ 内存管理 ============
    malloc_device_api malloc_device;               // 在设备上分配内存（GPU显存/CPU RAM）
    free_device_api free_device;                   // 释放设备内存
    malloc_host_api malloc_host;                   // 分配主机内存（CPU RAM，可能是 pinned）
    free_host_api free_host;                       // 释放主机内存
    
    // ============ 内存拷贝 ============
    memcpy_sync_api memcpy_sync;                   // 同步拷贝（CPU ↔ GPU 或 GPU ↔ GPU）
    memcpy_async_api memcpy_async;                 // 异步拷贝（使用 stream 在后台进行）
};

    // Llaisys API for getting the runtime APIs
    __export const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t);

    // Llaisys API for switching device context
    __export void llaisysSetContextRuntime(llaisysDeviceType_t, int);
}

#endif // LLAISYS_RUNTIME_H
