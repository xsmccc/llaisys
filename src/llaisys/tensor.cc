#include "llaisys_tensor.hpp"

#include <vector>
// C API 包装层，提供对 Tensor 类的访问接口

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        
        // 将 C 数组转换为 C++ vector
        std::vector<size_t> shape_vec(shape, shape + ndim);

        // 调用C++张量创建函数
        // 初始化张量对象
        // 包装为C句柄
        return new LlaisysTensor{llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
    }

    // 张量销毁
    // 张量析构时：
    // ├─ Tensor 对象析构
    // │  └─ _storage (shared_ptr) 析构
    // │     └─ Storage 析构
    // │        └─ _runtime.freeStorage(this)
    // │           └─ api->free_device(ptr)
    // │              └─► cudaFree(ptr)  ◄─── 实际释放显存
    // │
    // └─ LlaisysTensor 销毁
    void tensorDestroy(
        llaisysTensor_t tensor) {
        delete tensor;
    }

    // 数据指针查询
    // Python 获取张量数据指针
    void *tensorGetData(
        llaisysTensor_t tensor) {
        return tensor->tensor->data();
    }

    
    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        return tensor->tensor->ndim();
    }

    // Shape 查询
    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        return tensor->tensor->dtype();
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        return uint8_t(tensor->tensor->isContiguous());
    }

    // 数据加载
    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        tensor->tensor->load(data);
    }

    // 张量视图变形
    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{tensor->tensor->view(shape_vec)};
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
        return new LlaisysTensor{tensor->tensor->permute(order_vec)};
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
    }
}
