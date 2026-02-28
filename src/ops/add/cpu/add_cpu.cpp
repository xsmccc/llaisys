#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <omp.h>  // OpenMP 头文件

/*
1. constexpr的作用：
2. is_same_v<T, U>：用于在编译时检查两个类型是否相同。如果T和U是同一类型，则is_same_v<T, U>为true，否则为false。
3. if constexpr：这是C++17引入的特性，允许在编译时根据条件选择代码路径。与普通的if语句不同，if constexpr会在编译时评估条件，并且只编译满足条件的代码块。这对于模板编程非常有用，可以根据类型特性选择不同的实现。
4. OpenMP并行化：通过#pragma omp parallel for指令，循环迭代被分配给多个线程并行执行。schedule(static)表示迭代均匀分配给线程，适用于计算量均匀的任务。
5. 对于半精度类型（如bf16 和 fp16）：由于这些类型在CPU上无法直接进行数学运算，需要先转换为float类型进行计算，再转换回原始类型。
*/
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    // OpenMP 并行化：将循环迭代分配给多个线程
    // #pragma omp parallel for
    // parallel: 创建线程团队
    // for: 将循环迭代分配给不同线程
    // schedule(static): 静态调度，迭代均匀分配给线程
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // 进行类型转换之后再进行计算 因为对于半精度bf16和fp16不能再CPU上直接做数学运算
            c[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}
//内核层
namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
