#include <cuda_runtime.h>
#include <iostream>

// 宏定义 Tiling 参数
#define TILE_SIZE 32

/*
 * 基本 Tiling SGEMM Kernel (无 Padding 基础版)
 * A: [M, K], B: [K, N], C: [M, N]
 */
__global__ void sgemm_tiling(float *A, float *B, float *C, int M, int N, int K) {
    // 1. 申请 Shared Memory
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // 2. 获取当前线程负责计算的 C 矩阵元素的全局坐标 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 3. 线程在 Block 内部的局部坐标 (tx, ty)，用于写入 Shared Memory
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // 4. 沿着 K 维度进行步进 (Loop over K)，每次步进 TILE_SIZE
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE; 
    
    for (int t = 0; t < num_tiles; ++t) {
        // --- 协作搬运数据到 Shared Memory ---
        // 映射 A 矩阵的加载坐标：当前属于第 row 行，列在 t * TILE_SIZE 基础加上局部 tx
        // int a_col = t * TILE_SIZE + tx;
        // if (row < M && a_col < K) {
        //     sA[ty][tx] = A[row * K + a_col];
        // } else {
        //     sA[ty][tx] = 0.0f; // 越界补0
        // }
        int a_col = t * TILE_SIZE + tx;
        if (row < M && col < K){
            sA[ty][tx] = A[row * K + a_col];
        }else{
            sA[ty][tx] = 0.0f;
        }
        // 映射 B 矩阵的加载坐标：行在 t * TILE_SIZE 基础加上局部 ty，列是当前全局 col
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f; // 越界补0
        }

        // 同步！保证一个 Tile 内的所有数据都被所有线程就绪
        __syncthreads();

        // --- 读取 Shared Memory 并计算 ---
        // 每个线程对自己负责的那个点，做 TILE_SIZE 长度的点积
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        // 同步休整，防止跑得快的线程直接进入下一个 t 循环篡改了 sA/sB 的数据
        __syncthreads();
    }

    // 5. 将算完的结果写回 Global Memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    // 占位
    return 0;
}


