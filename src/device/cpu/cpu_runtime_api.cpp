#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::cpu {

namespace runtime_api {
// 1. 设备管理
int getDeviceCount() { return 1; }           // CPU 是单个"设备"
void setDevice(int) { /* do nothing */ }      // CPU 无需切换，没有设备 ID
void deviceSynchronize() { /* do nothing */ } // CPU 同步执行，无需等待

// 2. 流管理
llaisysStream_t createStream() { return (llaisysStream_t)0; }  // 返回 nullptr
void destroyStream(llaisysStream_t) { /* do nothing */ }
void streamSynchronize(llaisysStream_t) { /* do nothing */ }

// 3. 内存管理
void *mallocDevice(size_t size) { return std::malloc(size); }
void freeDevice(void *ptr) { std::free(ptr); }
void *mallocHost(size_t size) { return std::malloc(size); }   // Host 和 Device 同一地址空间
void freeHost(void *ptr) { std::free(ptr); }

// 4. 内存拷贝
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    std::memcpy(dst, src, size);  // CPU 直接 memcpy，无需区分方向
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    memcpySync(dst, src, size, kind);  // CPU 没有真正的异步，退化为同步
}

// 这是干嘛？？
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::cpu
