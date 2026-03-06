/**
 * @file tianshu_runtime_api.cu
 * @brief 天数智芯 TOPSRIDER Runtime API 实现
 *
 * 天数智芯 TOPSRIDER SDK 提供 CUDA 兼容 Runtime API。
 * 通过 topscc 编译器，所有 cuda* API 调用会被映射到 TOPSRIDER 底层实现。
 * 本文件实现 LlaisysRuntimeAPI 结构体所需的全部 12 个函数指针。
 */
#include "../runtime_api.hpp"
#include "tianshu_compat.cuh"

namespace {

// 将 LLAISYS 的 memcpy kind 转换为 CUDA/TOPSRIDER 兼容的 memcpy kind
cudaMemcpyKind toTopsMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: throw std::invalid_argument("Invalid memcpy kind");
    }
}

} // namespace

namespace llaisys::device::tianshu {

namespace runtime_api {

// ============ 设备管理 ============

int getDeviceCount() {
    int count = 0;
    checkTops(cudaGetDeviceCount(&count), "Failed to get device count");
    return count;
}

void setDevice(int device_id) {
    checkTops(cudaSetDevice(device_id), "Failed to set device");
}

void deviceSynchronize() {
    checkTops(cudaDeviceSynchronize(), "Failed to synchronize device");
}

// ============ 流管理 ============

llaisysStream_t createStream() {
    cudaStream_t stream;
    checkTops(cudaStreamCreate(&stream), "Failed to create stream");
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    checkTops(cudaStreamDestroy((cudaStream_t)stream), "Failed to destroy stream");
}

void streamSynchronize(llaisysStream_t stream) {
    checkTops(cudaStreamSynchronize((cudaStream_t)stream), "Failed to synchronize stream");
}

// ============ 内存管理 ============

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkTops(cudaMalloc(&ptr, size), "Failed to allocate device memory");
    return ptr;
}

void freeDevice(void *ptr) {
    checkTops(cudaFree(ptr), "Failed to free device memory");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkTops(cudaMallocHost(&ptr, size), "Failed to allocate host memory");
    return ptr;
}

void freeHost(void *ptr) {
    checkTops(cudaFreeHost(ptr), "Failed to free host memory");
}

// ============ 内存拷贝 ============

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind tops_kind = toTopsMemcpyKind(kind);
    checkTops(cudaMemcpy(dst, src, size, tops_kind), "Failed to copy memory (sync)");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    cudaMemcpyKind tops_kind = toTopsMemcpyKind(kind);
    checkTops(cudaMemcpyAsync(dst, src, size, tops_kind, (cudaStream_t)stream),
              "Failed to copy memory (async)");
}

// Tianshu TOPSRIDER Runtime API 结构体
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

} // namespace llaisys::device::tianshu
