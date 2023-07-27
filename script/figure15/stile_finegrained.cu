
#include <assert.h>
// CUDA runtime
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>
// #include <math>
#include <algorithm>
#include <assert.h>
#include "iostream"
#include "sstream"
#include "time.h"
#include "memory"
#include "vector"
using namespace std;

// #include "utils.hpp"
using namespace std;
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define FULL_MASK 0xffffffff

#define CUBLAS_SAFE_CALL(func)                                                                  \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t e = (func);                                                              \
        if (e != CUBLAS_STATUS_SUCCESS)                                                         \
        {                                                                                       \
            std::stringstream safe_call_ss;                                                     \
            safe_call_ss << "\nerror: " #func " failed with error"                              \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e; \
            throw std::runtime_error(safe_call_ss.str());                                       \
        }                                                                                       \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                                         \
    do                                                                                            \
    {                                                                                             \
        cudaError_t result = (x);                                                                 \
        if (result != cudaSuccess)                                                                \
        {                                                                                         \
            const char *msg = cudaGetErrorString(result);                                         \
            std::stringstream safe_call_ss;                                                       \
            safe_call_ss << "\nerror: " #x " failed with error"                                   \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
            throw std::runtime_error(safe_call_ss.str());                                         \
        }                                                                                         \
    } while (0)

__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}

__device__ __forceinline__ const int* add_ptr_u(const int* src, int offset)      \
{                                                                            \
    const int* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ const float* add_ptr_f(const float* src, int offset)      \
{                                                                            \
    const float* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }

void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            // ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            ptr[i] = 1;
        }
    }
}

void calculate_reference(int m, int k, int n, float * A, float *B, float * C) 
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float sum = 0.0;
            for(int tmp=0; tmp<k; tmp++){
                sum += A[i * k + tmp] * B[tmp * n + j];
            }
            C[i*n+j] = sum;
        }
    }
}
template <
    const int N_TILE_SIZE,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void FINEGRAINED_CONDENSE_KERNEL(const int* __restrict__  csr_row, const int* __restrict__  csr_col, const float* __restrict__  csr_val,  float* __restrict__  B, float* __restrict__  C, const int M, const int K, const int N){
    

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    // int ty = threadIdx.y;
    // int tx = threadIdx.x;
    const int padding = 1;
    __shared__ int Is[BLOCK_SIZE_K];
    __shared__ float Vs[BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K*(BLOCK_SIZE_N+padding)];
    assert(N_TILE_SIZE%BLOCK_SIZE_N==0);
    int index_start = csr_row[by];
    int index_end = csr_row[by+1];
    int row_nnz = index_end - index_start;
    int n_thread_per_row = BLOCK_SIZE_N/4;
    int n_stride = blockDim.x/n_thread_per_row;
    int ty = tid/n_thread_per_row;
    int tx = tid%n_thread_per_row;
    #pragma unroll
    for(int n_round=0; n_round<N_TILE_SIZE/BLOCK_SIZE_N; n_round++){
        float sum = 0;
        int n_start = bx * N_TILE_SIZE + n_round * BLOCK_SIZE_N;
        int n_end = n_start + BLOCK_SIZE_N;
        #pragma unroll
        for(int k_round=0; k_round< (index_end-index_start-1+BLOCK_SIZE_K)/BLOCK_SIZE_K; k_round++){
            // load the A to the shared memory
            int k_start = index_start + k_round * BLOCK_SIZE_K;
            // int k_end = min(k_start+ BLOCK_SIZE_K, index_end);
            int k_end = k_start+ BLOCK_SIZE_K;
            for(int _pos=tid+k_start; _pos<k_end; _pos+=blockDim.x){
                if(_pos<index_end){
                    Is[_pos-k_start] = csr_col[_pos];
                    Vs[_pos-k_start] = csr_val[_pos];
                }else{
                    Vs[_pos-k_start] = 0;
                }
            }
            __syncthreads();
            // load B to the shared memory
            #pragma unroll
            for(int _pos=ty; _pos<min(index_end-k_start, BLOCK_SIZE_K); _pos+=n_stride){
                int k_offset = Is[_pos];
                FETCH_FLOAT4(Bs[OFFSET(_pos, tx*4, BLOCK_SIZE_N)]) = 
                    FETCH_FLOAT4(B[OFFSET(k_offset, n_start+tx*4, N)]);
            }
            __syncthreads();
            // computation the spmv
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i++){
                sum += Vs[i]*Bs[OFFSET(i, tid,BLOCK_SIZE_N)];
            }

        }
        // write backto C
        C[OFFSET(by, n_start+tid, N)] = sum;
    }

}

void FINEGRAINED_CONDESE(int *csr_row, int * csr_col, float* csr_val, float * B, float* C, int M, int K, int N)
{
    const int N_TILE_SIZE = 1024;
    const int BLOCK_SIZE_N = 256;
    const int BLOCK_SIZE_K = 4;
    dim3 gridDim(N/N_TILE_SIZE, M);
    dim3 blockDim(BLOCK_SIZE_N);
    FINEGRAINED_CONDENSE_KERNEL<N_TILE_SIZE, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_val, B, C, M, K, N);

}


template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void FINEGRAINED_CONDENSE_KERNEL_V2(const int* __restrict__  csr_row, const int* __restrict__  csr_col, const float* __restrict__  csr_val,  float* __restrict__  B, float* __restrict__  C, const int M, const int K, const int N){
    

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    // int ty = threadIdx.y;
    // int tx = threadIdx.x;
    const int padding = 1;

    __shared__ int Is[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Vs[BLOCK_SIZE_M * BLOCK_SIZE_K];
    int ty = tid/BLOCK_SIZE_N;
    int tx = tid%BLOCK_SIZE_N;
    int row_id = by * BLOCK_SIZE_M + ty;
    int index_start = csr_row[row_id];
    int index_end = csr_row[row_id+1];
    const int n_thread_per_row = BLOCK_SIZE_N;

    #pragma unroll
    // for(int n_round=0; n_round<N_TILE_SIZE/BLOCK_SIZE_N; n_round++){
    float sum = 0;
    int n_start = bx * BLOCK_SIZE_N;
    // int n_end = n_start + BLOCK_SIZE_N;
    #pragma unroll
    for(int k_round=0; k_round< (index_end-index_start-1+BLOCK_SIZE_K)/BLOCK_SIZE_K; k_round++){
        // load the A to the shared memory
        int k_start = index_start + k_round * BLOCK_SIZE_K;
        // int k_end = min(k_start+ BLOCK_SIZE_K, index_end);
        int k_end = k_start + BLOCK_SIZE_K;
        for(int _pos=tx+k_start; _pos<k_end; _pos+=n_thread_per_row){
            if(_pos<index_end){
                Is[ty*BLOCK_SIZE_K + _pos-k_start] = csr_col[_pos];
                Vs[ty*BLOCK_SIZE_K + _pos-k_start] = csr_val[_pos];
            }else{
                Vs[ty*BLOCK_SIZE_K + _pos-k_start] = 0;
            }
        }
        __syncthreads();
        // load B to the shared memory
        // #pragma unroll
        // for(int _pos=ty; _pos<min(index_end-k_start, BLOCK_SIZE_K); _pos+=n_stride){
        //     int k_offset = Is[_pos];
        //     FETCH_FLOAT4(Bs[OFFSET(_pos, tx*4, BLOCK_SIZE_N)]) = 
        //         FETCH_FLOAT4(B[OFFSET(k_offset, n_start+tx*4, N)]);
        // }
        // __syncthreads();
        // computation the spmv
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_K;i++){
            int k_offset = Is[ty*BLOCK_SIZE_K+i];
            sum += Vs[ty*BLOCK_SIZE_K + i]*B[OFFSET(k_offset, n_start + tx, N)];
        }

    }
    // write backto C
    C[OFFSET(row_id, n_start+tx, N)] = sum;
    // }

}
void FINEGRAINED_CONDESE_V2(int *csr_row, int * csr_col, float* csr_val, float * B, float* C, int M, int K, int N)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 32;
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_M*BLOCK_SIZE_N);
    FINEGRAINED_CONDENSE_KERNEL_V2<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_val, B, C, M, K, N);

}

template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void FINEGRAINED_CONDENSE_KERNEL_V3(const int* __restrict__  csr_row, const int* __restrict__  csr_col, const float* __restrict__  csr_val,  float* __restrict__  B, float* __restrict__  C, const int M, const int K, const int N){
    

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    const int padding = 1;

    __shared__ int Is[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Vs[BLOCK_SIZE_M * BLOCK_SIZE_K];

    int tx = tid % BLOCK_SIZE_N;
    const int n_thread_per_row = BLOCK_SIZE_N;
    // for(int n_round=0; n_round<N_TILE_SIZE/BLOCK_SIZE_N; n_round++){
    float sum = 0;
    int n_start = bx * BLOCK_SIZE_N;
    int m_stride = blockDim.x / BLOCK_SIZE_N;
    #pragma unroll
    for(int ty=tid/BLOCK_SIZE_N; ty<BLOCK_SIZE_M; ty+=m_stride){
        sum = 0;
        int row_id = by * BLOCK_SIZE_M + ty;
        int index_start = csr_row[row_id];
        int index_end = csr_row[row_id+1];

        #pragma unroll
        for(int k_round=0; k_round< (index_end-index_start-1+BLOCK_SIZE_K)/BLOCK_SIZE_K; k_round++){
            // load the A to the shared memory
            int k_start = index_start + k_round * BLOCK_SIZE_K;
            // int k_end = min(k_start+ BLOCK_SIZE_K, index_end);
            int k_end = k_start + BLOCK_SIZE_K;
            for(int _pos=tx+k_start; _pos<k_end; _pos+=n_thread_per_row){
                if(_pos<index_end){
                    Is[ty*BLOCK_SIZE_K + _pos-k_start] = csr_col[_pos];
                    Vs[ty*BLOCK_SIZE_K + _pos-k_start] = csr_val[_pos];
                }else{
                    Vs[ty*BLOCK_SIZE_K + _pos-k_start] = 0;
                }
            }
            __syncthreads();

            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i++){
                int k_offset = Is[ty*BLOCK_SIZE_K+i];
                sum += Vs[ty*BLOCK_SIZE_K + i]*B[OFFSET(k_offset, n_start + tx, N)];
            }

        }
        // write backto C
        C[OFFSET(row_id, n_start+tx, N)] = sum;
    }
    // }

}
void FINEGRAINED_CONDESE_V3(int *csr_row, int * csr_col, float* csr_val, float * B, float* C, int M, int K, int N)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 64;
    const int BLOCK_SIZE_K = 32;
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(512);
    FINEGRAINED_CONDENSE_KERNEL_V3<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_val, B, C, M, K, N);

}



template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_K,
    const int BLOCK_SIZE_N
>
__global__ void FINEGRAINED_CONDENSE_KERNEL_V4(const int* __restrict__  csr_row, const int* __restrict__  csr_col, const float* __restrict__  csr_val,  float* __restrict__  B, float* __restrict__  C, const int M, const int K, const int N){
    

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    const int padding = 1;

    __shared__ int Is[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Vs[BLOCK_SIZE_M * BLOCK_SIZE_K];
    const int n_threads_per_row = 32;
    int ty = tid/n_threads_per_row;
    int tx = tid%n_threads_per_row;
    int row_id = by * BLOCK_SIZE_M + ty;
    int index_start = csr_row[row_id];
    int index_end = csr_row[row_id+1];
    const int n_thread_per_row = BLOCK_SIZE_N;

    #pragma unroll
    // for(int n_round=0; n_round<N_TILE_SIZE/BLOCK_SIZE_N; n_round++){
    float sum = 0;
    int n_start = bx * BLOCK_SIZE_N;
    // int n_end = n_start + BLOCK_SIZE_N;
    #pragma unroll
    for(int k_round=0; k_round< (index_end-index_start-1+BLOCK_SIZE_K)/BLOCK_SIZE_K; k_round++){
        // load the A to the shared memory
        int k_start = index_start + k_round * BLOCK_SIZE_K;
        // int k_end = min(k_start+ BLOCK_SIZE_K, index_end);
        int k_end = k_start + BLOCK_SIZE_K;
        for(int _pos=tx+k_start; _pos<k_end; _pos+=n_thread_per_row){
            if(_pos<index_end){
                Is[ty*BLOCK_SIZE_K + _pos-k_start] = csr_col[_pos];
                Vs[ty*BLOCK_SIZE_K + _pos-k_start] = csr_val[_pos];
            }else{
                Vs[ty*BLOCK_SIZE_K + _pos-k_start] = 0;
            }
        }
        __syncthreads();
        // load B to the shared memory
        // #pragma unroll
        // for(int _pos=ty; _pos<min(index_end-k_start, BLOCK_SIZE_K); _pos+=n_stride){
        //     int k_offset = Is[_pos];
        //     FETCH_FLOAT4(Bs[OFFSET(_pos, tx*4, BLOCK_SIZE_N)]) = 
        //         FETCH_FLOAT4(B[OFFSET(k_offset, n_start+tx*4, N)]);
        // }
        // __syncthreads();
        // computation the spmv
        #pragma unroll
        for(int i=0;i<BLOCK_SIZE_K;i++){
            int k_offset = Is[ty*BLOCK_SIZE_K+i];
            sum += Vs[ty*BLOCK_SIZE_K + i]*B[OFFSET(k_offset, n_start + tx, N)];
        }

    }
    // write backto C
    C[OFFSET(row_id, n_start+tx, N)] = sum;
    // }

}
void FINEGRAINED_CONDESE_V4(int *csr_row, int * csr_col, float* csr_val, float * B, float* C, int M, int K, int N)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 32;
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_M*BLOCK_SIZE_N);
    FINEGRAINED_CONDENSE_KERNEL_V2<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N><<<gridDim, blockDim>>>(csr_row, csr_col, csr_val, B, C, M, K, N);

}


int convert_csr(float * ptr, int32_t row, int32_t col, int32_t * row_idx, int32_t * col_idx, float * values)
{
    auto v_row_idx = std::make_shared<vector<int32_t>>();
    auto v_col_idx = std::make_shared<vector<int32_t>>();
    auto v_values = std::make_shared<vector<float>>();

    for (int i = 0; i < row; i++)
    {
        v_row_idx->push_back(v_values->size());
        for (int j = 0; j < col; j++)
        {
            size_t pos = i * col + j;
            if (ptr[pos] < 1e-8)
            {
                // sparsity
                continue;
            }
            else
            {
                v_values->push_back(ptr[pos]);
                v_col_idx->push_back(j);
            }
        }
    }
    v_row_idx->push_back(v_values->size());
    int row_idx_size = sizeof(int32_t)*v_row_idx->size();
    int col_idx_size = sizeof(int32_t)*v_col_idx->size();
    int values_size = sizeof(float)*v_values->size();
    printf("values_size: %d\n", values_size);

    memcpy(row_idx, v_row_idx->data(), row_idx_size);
    memcpy(col_idx, v_col_idx->data(), col_idx_size);
    memcpy(values, v_values->data(), values_size);
    return v_values->size();
}


int main()
{
    int M, K, N;
    M = 4096;
    K = 4096;
    N = 4096;
    const int n_iter = 100;
    float sparsity_ratio = 0.6;

    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    float * A, *B, *C, *val, *refC;
    float * dA, *dB, *dC, *d_val;

    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col, *d_extra_buffer;
    A = (float*) malloc(sizeof(float) * M * K);
    B = (float*) malloc(sizeof(float) * K * N);
    C = (float*) malloc(sizeof(float) * M * N);
    refC = (float*) malloc(sizeof(float) * M * N);

    row = (int*) malloc(sizeof(int) * (M+1));
    col = (int*) malloc(sizeof(int) *  M * K);
    val = (float*) malloc(sizeof(float) * M * K);
    init(A, M*K, sparsity_ratio);
    init(B, N*K, 0);
    // apply mask

    convert_csr(A, M, K, row, col, val);
    int nnz = row[M];
    
    printf("NNZ: %d\n", nnz);
    printf("Sparsity ratio: %f\n", 1-nnz*1.0/M/K);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K));

    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(float)* M * N));
    
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)* M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    

    // KxM = KxN * (MxN)^T
    CUDA_SAFE_CALL(cudaEventRecord(time_start));

    for(int run=0; run<n_iter; run++){
        // FINEGRAINED_CONDESE_V2(d_row, d_col, d_val, dB, dC, M, K, N);
        FINEGRAINED_CONDESE_V3(d_row, d_col, d_val, dB, dC, M, K, N);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time Cost: %.3fms\n", msecTotal/n_iter);
    CUDA_SAFE_CALL(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    calculate_reference(M, K, N, A, B, refC);
    for(int i=0;i<M*N;i++){
        if(fabs(C[i]-refC[i])/fabs(refC[i])>0.001)
            printf("%f %f\n", C[i], refC[i]);
    }


    return 0;

}
