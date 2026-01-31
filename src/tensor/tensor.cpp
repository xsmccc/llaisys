#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

//storage——存储张量数据的内存块的共享指针
//offset——张量在存储中的起始索引，byte为单位
//meta——描述张量形状、数据类型和步长的元数据
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (_meta.shape.empty()) return true;
    size_t accumulated = 1;
    for (size_t i = _meta.shape.size(); i-- > 0;) {
        // _meta.strides[i] 是 ptrdiff_t，这里强转一下比较
        if ((size_t)_meta.strides[i] != accumulated) {
            return false;
        }
        accumulated *= _meta.shape[i];
    }
    return true;
}

// 创建新张量，改变原始张量维度的顺序，转置也通过此函数实现，无需移动数据
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查参数合法性 维度是否一样
    if (order.size() != _meta.shape.size()){
        throw std::runtime_error("Permute dimensions mismatch.");
    }

    std::vector<ptrdiff_t> new_stride;
    std::vector<size_t> new_shape;

    new_stride.resize(_meta.strides.size());
    new_shape.resize(_meta.shape.size());
    size_t i = 0;

    for (size_t d : order){
        if (d >= _meta.shape.size()){
            throw std::runtime_error("Permute dimensions mismatch.");
        }
        new_shape[i] = (_meta.shape[d]);
        new_stride[i] = (_meta.strides[d]);
        i++;
    }

    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = new_shape;
    new_meta.strides = new_stride;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

// 返回一个以传入的shape构造的新张量 也就是变换形状
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t current_numel = 1;
    for (size_t dim: _meta.shape) current_numel *= dim; //获取当前数据总量

    size_t new_numel = 1;
    for (size_t dim : shape) new_numel *= dim; //获取当前数据总量

    if (current_numel != new_numel)
        throw std::runtime_error("View shape mismatch: numel must be the same.");
    
    if(!isContiguous())
        throw std::runtime_error("View on non-contiguous tensor is not supported yet.");
    
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = static_cast<ptrdiff_t>(stride);
        stride *= shape[i];
    }

    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;
    new_meta.strides = new_strides;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 输入参数有效性判断
    // 维度判断
    if (dim >= _meta.shape.size()){
        throw std::runtime_error("Slice dimesion out of range.\n");
    }
    // 参数判断
    if (start > _meta.shape[dim] || start < 0 || end > _meta.shape[dim] || start >= end){
        throw std::runtime_error("Slice : Argument start or end out of range.\n");
    }
    
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start; // 输入参数左闭右开

    //数据排布不变 变得是offset 那么stride就不变 
    std::vector<ptrdiff_t> new_stride = _meta.strides;

    //修改偏移量
    size_t elem_size = 0;
    switch (_meta.dtype){
        case LLAISYS_DTYPE_I8:
        case LLAISYS_DTYPE_U8:
        case LLAISYS_DTYPE_BYTE:
        case LLAISYS_DTYPE_BOOL:
        case LLAISYS_DTYPE_F8:
            elem_size = 1;
            break;
        case LLAISYS_DTYPE_I16:
        case LLAISYS_DTYPE_U16:
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16:
            elem_size = 2;
            break;
        case LLAISYS_DTYPE_I32:
        case LLAISYS_DTYPE_U32:
        case LLAISYS_DTYPE_F32:
            elem_size = 4;
            break;
        case LLAISYS_DTYPE_I64:
        case LLAISYS_DTYPE_U64:
        case LLAISYS_DTYPE_F64:
            elem_size = 8;
            break;
        default:
            elem_size = 4; 
            break;
    }
    size_t additional_offset = start * _meta.strides[dim] * elem_size; //多余的偏移量来自于前面跳过的部分，和end无关
    size_t new_offset = _offset + additional_offset;

    //构造新 Tensor
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = new_shape;
    new_meta.strides = new_stride;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage,new_offset));
}

//task1-1
void Tensor::load(const void *src_) {
    size_t numel = 1;
    // 首先获取维度大小
    for(size_t dim : _meta.shape)
        numel *= dim;
    //定义数据类型的大小
    size_t elem_size = 0;
    switch (_meta.dtype){
        case LLAISYS_DTYPE_I8:
        case LLAISYS_DTYPE_U8:
        case LLAISYS_DTYPE_BYTE:
        case LLAISYS_DTYPE_BOOL:
        case LLAISYS_DTYPE_F8:
            elem_size = 1;
            break;
        case LLAISYS_DTYPE_I16:
        case LLAISYS_DTYPE_U16:
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16:
            elem_size = 2;
            break;
        case LLAISYS_DTYPE_I32:
        case LLAISYS_DTYPE_U32:
        case LLAISYS_DTYPE_F32:
            elem_size = 4;
            break;
        case LLAISYS_DTYPE_I64:
        case LLAISYS_DTYPE_U64:
        case LLAISYS_DTYPE_F64:
            elem_size = 8;
            break;
        default:
            elem_size = 4; 
            break;
    }
    size_t total_bytes = numel * elem_size;     //获取总字节数
    std::byte *dst = _storage->memory() + _offset; //获取目标地址
    llaisysDeviceType_t dev_type = _storage->deviceType(); //获取设备类型
    const LlaisysRuntimeAPI* api = llaisysGetRuntimeAPI(dev_type);// 调用runtimeapi
    api->memcpy_sync(dst, src_, total_bytes,LLAISYS_MEMCPY_H2D); //从api中获取内存搬运函数
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
