#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include "cstring"

// 数据搬运
template<typename T,typename IndexT>
void embedding_kernel(
        T* out,
        const IndexT* indices,
        const T* weight,
        size_t num_indices,
        size_t vocab_size,
        size_t embedding_dim){
    size_t row_size_in_bytes = sizeof(T) * embedding_dim;

    for (size_t i = 0;i < num_indices;i++){
        IndexT idx = indices[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < vocab_size,"Embedding index out of bound");
        const T* src_row = weight + idx * embedding_dim;
        T* dst_row = out + i * embedding_dim;
        std::memcpy(dst_row,src_row,row_size_in_bytes);
    }
}

template <typename T>
void dispatch_index_type(
    std::byte* out, 
    const std::byte* index, 
    const std::byte* weight,
    llaisysDataType_t index_dtype, 
    size_t num_indices, 
    size_t vocab_size, 
    size_t embedding_dim){
    if (index_dtype == LLAISYS_DTYPE_I32){
        embedding_kernel<T,int32_t>(
            reinterpret_cast<T*>(out),
            reinterpret_cast<const int32_t*>(index),
            reinterpret_cast<const T*>(weight),
            num_indices,vocab_size,embedding_dim
        );
    }
    else if(index_dtype == LLAISYS_DTYPE_I64){
        embedding_kernel<T,int64_t>(
            reinterpret_cast<T*>(out),
            reinterpret_cast<const int64_t*>(index),
            reinterpret_cast<const T*>(weight),
            num_indices,vocab_size,embedding_dim
        );
    }
}

namespace llaisys::ops::cpu{
    void embedding(std::byte* out_ptr,
        llaisysDataType_t out_type,
        const std::byte* index_ptr,
        size_t num_indices,
        llaisysDataType_t index_dtype,
        const std::byte* weight_ptr,
        size_t vocab_size,
        size_t embedding_dim){
            switch (out_type){
                case LLAISYS_DTYPE_F32:
                dispatch_index_type<float>(out_ptr,index_ptr,weight_ptr,index_dtype,num_indices, vocab_size, embedding_dim);
                break;
                case LLAISYS_DTYPE_BF16:
                dispatch_index_type<llaisys::bf16_t>(out_ptr, index_ptr, weight_ptr, index_dtype, num_indices, vocab_size, embedding_dim);
                break;
            case LLAISYS_DTYPE_F16:
                dispatch_index_type<llaisys::fp16_t>(out_ptr, index_ptr, weight_ptr, index_dtype, num_indices, vocab_size, embedding_dim);
                break;
            default:
                EXCEPTION_UNSUPPORTED_DATATYPE(out_type);
            }
        }
}