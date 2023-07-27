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
// CUDA runtime
#include <cuda.h>
using namespace std;
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define MAX_TILE_PER_BLOCK 16
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

void init_mask_blockwise(int * ptr, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            int pos = i*block_h*N+j*block_w;
            if (pro < sparsity)
            {
                ptr[pos] = 0;
            }
            else
            {
                ptr[pos] = 1;
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);

}
__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }


__global__ void convert_bcsr_kernel_1_align_4(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col, int * row_pos,
                                float * values, int * extra_buffer)
{

    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    assert(blockDim.x<=MAX_BLOCK_THREAD_COUNT);
    // initial the shared flag
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x;
    int global_offset =  (by * block_h) * w + bx * block_w;
    int block_size =  block_h * block_w;
    assert(block_w % 4 == 0);
    // cannot handle the misalignment for now
    assert((block_size / 4) % blockDim.x==0);
    int flag = 0;
    for(int _pos = tid; _pos< block_size / 4; _pos+=blockDim.x){
        uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;        
        int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset+block_offset)));
        flag += data.x + data.y + data.z + data.w;
    }
    reduce[tid] = flag;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    int pos_id;
    if(tid==0 && reduce[0]>0){
        pos_id= atomicAdd(&extra_buffer[by], 1);
        atomicAdd(&extra_buffer[by+h], 1);
        atomicAdd(&row[h/block_h], 1);
        extra_buffer[2*h + gridDim.x * by + pos_id] = bx;
    }

}
__global__ void convert_bcsr_kernel_2_align_4(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float * values, int * extra_buffer)
{
    __shared__ int pos_id, prefix_count, ori_bx, ori_by;
    __shared__ int prefix_sum[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;

    if (tid==0){
        pos_id = -1;
        prefix_count = 0;
        // contend for the block

        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if (pos_id>=0){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }
            ori_by = by;
            ori_bx = extra_buffer[ 2*h + by * gridDim.x + pos_id];       
            
            row[by] = prefix_count;
            col[prefix_count+pos_id] = ori_bx;
            row_pos[prefix_count+pos_id] = by;
        }
        else if(pos_id==-1){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }            
            row[by] = prefix_count;
        }
    }
    __syncthreads();
    if(pos_id>=0){
        int global_offset =  (ori_by * block_h) * w + ori_bx * block_w;
        int block_size = block_h * block_w;
        int write_global_offset = (prefix_count + pos_id) * block_size;

        for(int _pos=tid; _pos<block_size/4; _pos+=blockDim.x){
            uint block_offset = _pos / (block_w / 4) * w + _pos % (block_w / 4) * 4;
            float4 data = __ldg((const float4*)(add_ptr_f(dense, global_offset + block_offset)));
            *(float4*)&values[write_global_offset+_pos*4] = data;
        }
        
    }

}


__global__ void convert_bcsr_kernel_1(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col, int * row_pos,
                                float * values, int * extra_buffer)
{

    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    // __shared__ int multi_tile[32];
    assert(blockDim.x<=MAX_BLOCK_THREAD_COUNT);
    // initial the shared flag
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x;
    int global_offset =  (by * block_h) * w + bx * block_w;
    int block_size =  block_h * block_w;
    // cannot handle the misalignment for now
    int flag = 0;
    for(int _pos = tid; _pos< block_size ; _pos+=blockDim.x){
        uint block_offset = _pos / (block_w ) * w + _pos % (block_w );        
        // int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset+block_offset)));
        // flag += data.x + data.y + data.z + data.w;
        flag += mask[global_offset+block_offset];
    }
    reduce[tid] = flag;
    __syncthreads();
    // fast tree reduce accross the block
    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid<s)
            reduce[tid] += reduce[tid+s];
        __syncthreads();
    }
    if(tid<32)
        warpReduce(reduce, tid);
    __syncthreads();
    int pos_id;
    if(tid==0 && reduce[0]>0){
        pos_id= atomicAdd(&extra_buffer[by], 1);
        atomicAdd(&extra_buffer[by+h], 1);
        atomicAdd(&row[h/block_h], 1);
        extra_buffer[2*h + gridDim.x * by + pos_id] = bx;
    }

}
__global__ void convert_bcsr_kernel_2(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float * values, int * extra_buffer)
{
    __shared__ int pos_id, prefix_count, ori_bx, ori_by;
    __shared__ int prefix_sum[MAX_BLOCK_THREAD_COUNT];
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;

    if (tid==0){
        pos_id = -1;
        prefix_count = 0;
        // contend for the block

        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if (pos_id>=0){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }
            ori_by = by;
            ori_bx = extra_buffer[ 2*h + by * gridDim.x + pos_id];       
            
            row[by] = prefix_count;
            col[prefix_count+pos_id] = ori_bx;
            row_pos[prefix_count+pos_id] = by;
        }
        else if(pos_id==-1){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }            
            row[by] = prefix_count;
        }
    }
    __syncthreads();
    if(pos_id>=0){
        int global_offset =  (ori_by * block_h) * w + ori_bx * block_w;
        int block_size = block_h * block_w;
        int write_global_offset = (prefix_count + pos_id) * block_size;

        for(int _pos=tid; _pos<block_size; _pos+=blockDim.x){
            uint block_offset = _pos / (block_w) * w + _pos % (block_w);
            values[write_global_offset+_pos] = dense[global_offset + block_offset];
        }
        
    }

}

__global__ void convert_bcsr_kernel_1_tile(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
                                int block_h, int block_w, int * row, int *col, int * row_pos,
                                float * values, int * extra_buffer, int tile_per_block)
{

    __shared__ int reduce[MAX_BLOCK_THREAD_COUNT];
    __shared__ int result[MAX_TILE_PER_BLOCK];
    int tile_flag[MAX_TILE_PER_BLOCK] = {0};
    assert(blockDim.x<=MAX_BLOCK_THREAD_COUNT);
    // initial the shared flag
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tid = threadIdx.x; 
    int global_offset =  (by * block_h) * w + bx * block_w * tile_per_block;
    int real_block_w = block_w * tile_per_block;
    int block_size =  block_h * real_block_w;
    assert((real_block_w) % 4 == 0);
    assert(tile_per_block <= MAX_TILE_PER_BLOCK);
    // cannot handle the misalignment for now
    // int flag = 0;
    for(int _pos = tid; _pos< block_size /4 ; _pos+=blockDim.x){
        uint col_idx = _pos % (real_block_w/4) *4;
        uint block_offset = _pos / (real_block_w/4) * w + col_idx;
        int4 data = __ldg((const int4*)(add_ptr_u(mask, global_offset+block_offset)));
        tile_flag[(col_idx)/block_w] += data.x;
        tile_flag[(col_idx+1)/block_w] += data.y;
        tile_flag[(col_idx+2)/block_w] += data.z;
        tile_flag[(col_idx+3)/block_w] += data.w;
        // #pragma unroll
        // for(int i=0;i<4;i++){
        //     tile_flag[(col_idx+i)/block_w] += *((int*)(&data)+i);
        // }
        // flag += data.x + data.y + data.z + data.w;
        // flag += mask[global_offset + block_offset];
    }
    __syncthreads();
    for(int tileid=0; tileid<tile_per_block; tileid++){
        reduce[tid] = tile_flag[tileid];
        // fast tree reduce accross the block
        for(uint s=blockDim.x/2; s>32; s>>=1){
            if(tid<s)
                reduce[tid] += reduce[tid+s];
            __syncthreads();
        }
        if(tid<32)
            warpReduce(reduce, tid);
        __syncthreads();
        if(tid==0)
            result[tileid] = reduce[0];
    }
    __syncthreads();
    int pos_id;
    if(tid<tile_per_block && result[tid]>0){
        pos_id= atomicAdd(&extra_buffer[by], 1);
        atomicAdd(&extra_buffer[by+h], 1);
        atomicAdd(&row[h/block_h], 1);
        extra_buffer[2*h + gridDim.x * by + pos_id] = bx*tile_per_block + tid;
    }

}
__global__ void convert_bcsr_kernel_2_tile(const int * __restrict__  mask, float * __restrict__  dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float * values, int * extra_buffer)
{
    __shared__ int pos_id, prefix_count, ori_bx, ori_by;
    __shared__ int prefix_sum[MAX_BLOCK_THREAD_COUNT];
    __shared__ int tile_flag[MAX_TILE_PER_BLOCK];
    uint by = blockIdx.y;
    uint bx = blockIdx.x;
    uint tid = threadIdx.x;

    if (tid==0){
        pos_id = -1;
        prefix_count = 0;
        // contend for the block

        pos_id = atomicSub(&extra_buffer[by], 1);
        pos_id-=1;
        if (pos_id>=0){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }
            ori_by = by;
            ori_bx = extra_buffer[ 2*h + by * gridDim.x + pos_id];       
            
            row[by] = prefix_count;
            col[prefix_count+pos_id] = ori_bx;
            row_pos[prefix_count+pos_id] = by;
        }
        else if(pos_id==-1){
            for(int i=0; i<by;i++){
                prefix_count +=  extra_buffer[h+i];
            }            
            row[by] = prefix_count;
        }
    }
    __syncthreads();
    if(pos_id>=0){
        int global_offset =  (ori_by * block_h) * w + ori_bx * block_w;
        int block_size = block_h * block_w;
        int write_global_offset = (prefix_count + pos_id) * block_size;

        for(int _pos=tid; _pos<block_size; _pos+=blockDim.x){
            uint block_offset = _pos / (block_w) * w + _pos % (block_w);
            values[write_global_offset+_pos] = dense[global_offset + block_offset];
        }
        
    }

}


void convert_bcsr(int * mask, float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float*values, int * extra_buffer)
{
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );
    // need reset the extra buffer here
    if(block_w % 4 == 0){
        
        dim3 block_dim(block_h*block_w/4);
        dim3 grid_dim(w/block_w, h/block_h);
        // std::cout<<"grid_dim "<< w/block_w << ", " <<h/block_h << std::endl;
        convert_bcsr_kernel_1_align_4<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
        convert_bcsr_kernel_2_align_4<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
    }else{
        dim3 block_dim(block_h*block_w);
        dim3 grid_dim(w/block_w, h/block_h);
        convert_bcsr_kernel_1<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
        convert_bcsr_kernel_2<<<grid_dim, block_dim>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);
    }

}
void convert_bcsr_tile(int * mask, float * dense, int h, int w,
    int block_h, int block_w, int * row, int *col, int * row_pos,
    float*values, int * extra_buffer, int tile_per_block)
{
    CUDA_SAFE_CALL(cudaMemset((void*)extra_buffer, 0, sizeof(int)*(2*h+(h/block_h)*(w/block_w))) );
    CUDA_SAFE_CALL(cudaMemset((void*)row, 0, sizeof(int)*(1+(h/block_h))) );


    dim3 block_dim_1(block_h*block_w*tile_per_block/4);
    dim3 grid_dim_1(w/block_w/tile_per_block, h/block_h);

    dim3 block_dim_2(block_h*block_w);
    dim3 grid_dim_2(w/block_w, h/block_h);

    convert_bcsr_kernel_1_tile<<<grid_dim_1, block_dim_1>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer, tile_per_block);
    convert_bcsr_kernel_2_tile<<<grid_dim_2, block_dim_2>>>(mask, dense, h, w, block_h, block_w, row, col, row_pos, values, extra_buffer);

}
bool verify_bcsr(int * mask, float * data, int h, int w, int block_h, int block_w, int* row, int * col, float* values)
{
    for(int rid=0; rid<h/block_h; rid++){
        // printf("row-%d: %d row-%d : %d\n", rid, row[rid], rid+1, row[rid+1]);
        int _start = row[rid];
        int _end = row[rid+1];
        for(int _pos=_start; _pos<_end; _pos++){
            int cid = col[_pos];
            for(int i=0;i<block_h;i++){
                for(int j=0;j<block_w;j++){
                    int offset = (rid * block_h+i) * w + cid * block_w + j;
                    int csr_offset = _pos * block_h * block_w + i * block_w + j;
                    if (mask[offset]>0){
                        // printf("%f %f\n", data[offset], values[csr_offset]);
                        if(abs(data[offset]-values[csr_offset])>1e-8)
                        {
                            return false;
                        }
                        mask[offset]= 0;
                    }
                }
            }
        }
    }
    printf("%d blocks remained\n", row[h/block_h]);
    printf("Blockwise sparsity %f \n", 1.0-1.0*row[h/block_h]/(h/block_w)/(w/block_w));
    for(int i=0;i<block_h*block_w;i++)
        if(mask[i])
            return false;
    return true;
}

int main()
{
    int M, K, N;
    M = 1024;
    K = 1024;
    N = 1024;
    const int n_iter = 10000;
    float sparsity_ratio = 0.95;
    const int BLOCK_H = 32;
    const int BLOCK_W = 32;
    // const int BLOCK_W = 1;
    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    float * A, *B, *C, *val;
    float * dA, *dB, *dC, *d_val;
    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col, *d_extra_buffer;
    A = (float*) malloc(sizeof(float) * M * K);
    B = (float*) malloc(sizeof(float) * K * N);
    C = (float*) malloc(sizeof(float) * M * N);
    mask = (int*) malloc(sizeof(int) * M * K);
    row = (int*) malloc(sizeof(int) * (M+1));
    col = (int*) malloc(sizeof(int) *  M * K / BLOCK_H / BLOCK_W);
    val = (float*) malloc(sizeof(float) * M * K);
    init_mask_blockwise(mask, M, K, BLOCK_H, BLOCK_W, sparsity_ratio);
    // apply mask
    for(int i=0; i< M*K; i++){
        A[i] = A[i] * mask[i];
    }
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMalloc(&d_extra_buffer, sizeof(float) * M * K));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int) * M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int i=0;i<n_iter;i++)
        convert_bcsr(d_mask, dA, M, K, BLOCK_H, BLOCK_W, d_row, d_col, d_row_pos, d_val, d_extra_buffer);
    // for(int i=0;i<n_iter;i++)
    //     convert_bcsr_tile(d_mask, dA, M, K, BLOCK_H, BLOCK_W, d_row, d_col, d_row_pos, d_val, d_extra_buffer, 4);
    
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(val, d_val, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMem)
    printf("Convert tim cost = %f msec\n", msecTotal/n_iter);
    if(verify_bcsr(mask, A, M, K, BLOCK_H, BLOCK_W, row, col, val)){
        printf("Bcsr format verification success\n");
    }else{
        printf("Bcsr format check failed!\n");
    }
    return 0;
}