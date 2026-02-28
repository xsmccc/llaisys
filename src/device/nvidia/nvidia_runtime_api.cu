#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// 避免全局污染
namespace {

// 将 LLAISYS 的 memcpy kind 转换为 CUDA 的 memcpy kind
// 适配器模式
cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid memcpy kind");
    }
}

// 检查 CUDA 错误  确保每个 CUDA 调用后都进行错误检查
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(msg);
    }
}

} // namespace

namespace llaisys::device::nvidia {

namespace runtime_api {

// ============ 设备管理 ============

int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "Failed to get device count");
    return count;
}

void setDevice(int device_id) {
    checkCuda(cudaSetDevice(device_id), "Failed to set device");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "Failed to synchronize device");
}

// ============ 流管理 ============

llaisysStream_t createStream() {
    cudaStream_t stream;     //CUDA流类型
    checkCuda(cudaStreamCreate(&stream), "Failed to create stream");
    return (llaisysStream_t)stream;     // 转换为通用流类型，即void*
}

void destroyStream(llaisysStream_t stream) {
    checkCuda(cudaStreamDestroy((cudaStream_t)stream), "Failed to destroy stream");
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize((cudaStream_t)stream), "Failed to synchronize stream");
}

// ============ 内存管理 ============

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "Failed to allocate device memory");
    return ptr;
}

void freeDevice(void *ptr) {
    checkCuda(cudaFree(ptr), "Failed to free device memory");
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    // 分配 pinned memory (page-locked memory)，用于更快的主机-设备传输
    checkCuda(cudaMallocHost(&ptr, size), "Failed to allocate host memory");
    return ptr;
}

void freeHost(void *ptr) {
    checkCuda(cudaFreeHost(ptr), "Failed to free host memory");
}

// ============ 内存拷贝 ============

// 同步拷贝
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind = toCudaMemcpyKind(kind);
    checkCuda(cudaMemcpy(dst, src, size, cuda_kind), "Failed to copy memory (sync)");
}

// 异步拷贝
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind = toCudaMemcpyKind(kind);
    checkCuda(cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream),
              "Failed to copy memory (async)");
}

// NVIDIA Runtime API 结构体
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
    return &runtime_api::RUNTIME_API;   // 返回结构体指针
}

} // namespace llaisys::device::nvidia
