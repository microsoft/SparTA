#include <assert.h>
// CUDA runtime
#include <cuda.h>
#include <stdio.h>
#include "utils.hpp"
using namespace std;
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define SOFTMAX_ROW_TILE 1
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



template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_SDD(int* csr_row, int * csr_col, float* csr_val, float * B, float* C,  int M, int K, int N, int block_h, int block_w, int sparse_val_size){
    // const int BLOCK_SIZE_M = 32;
    // const int BLOCK_SIZE_K = 32;
    // const int BLOCK_SIZE_N = 64;
    // const int THREAD_SIZE_M = 4;
    // const int THREAD_SIZE_K = 4;
    // const int THREAD_SIZE_N = 4;
    int by = blockIdx.y; // M
    int bx = blockIdx.x; // N
    int bz = blockIdx.z;
    int ty = threadIdx.y; 
    int tx = threadIdx.x;
    csr_val = csr_val + sparse_val_size * bz;
    B = B + K * N * bz;
    C = C + M * N * bz;

    const int padding = 1;
    __shared__ float As[BLOCK_SIZE_M * (padding+BLOCK_SIZE_K)];
    __shared__ float Bs[BLOCK_SIZE_N * (padding+BLOCK_SIZE_K)];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int index_start = csr_row[by], index_end = csr_row[by+1];

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;
    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;

    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int col_pos = csr_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(csr_val[tile_block_idx * BLOCK_SIZE_M * BLOCK_SIZE_K + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]);
        }

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
            FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
                FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N + B_BLOCK_COL_START, N)]);
                // FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
                // FETCH_FLOAT4(B[OFFSET(tile_idx+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N+B_BLOCK_COL_START, N)]);
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j += 1){
                    a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                    //a_frag[j][i] = As[OFFSET(k+i, ty + vBLOCK_SIZE_M * j, BLOCK_SIZE_M)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_N; j += 1){
                    b_frag[j][i] = Bs[OFFSET(k+i, tx + vBLOCK_SIZE_N * j, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_N; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j++){
                    #pragma unroll
                    for(int k_in = 0; k_in < THREAD_SIZE_K; k_in++){
                        // accum[i][j] = fma(a_frag[j][k_in], b_frag[i][k_in], accum[i][j]);
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }


    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]);
        }
    }


}


template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_SDD_CONDENSE_K(int* csr_row, int * csr_col, float* csr_val, float * B, float* C,  int M, int K, int N, int block_h, int block_w){

    int by = blockIdx.y; // M
    int bx = blockIdx.x; // N
    
    int ty = threadIdx.y; 
    int tx = threadIdx.x;


    const int padding = 1;
    __shared__ float As[BLOCK_SIZE_M * (padding+BLOCK_SIZE_K)];
    __shared__ float Bs[BLOCK_SIZE_N * (padding+BLOCK_SIZE_K)];
    __shared__ int local_k_offset[BLOCK_SIZE_K];
    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_M / 4; //data block is BLOCK_SIZE_M x 1
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int index_start = csr_row[by], index_end = csr_row[by+1];

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;
    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    int k_nnz = index_end-index_start;
    int k_round = (k_nnz - 1 + BLOCK_SIZE_K)/BLOCK_SIZE_K;
    // if(tid==0 && by==gridDim.y-1 && bx==0){
    //     printf("bx:%d by%d k_nnz:%d k_round:%d\n", bx, by, k_nnz, k_round);
    // }
    for(int rid=0; rid<k_round; rid++){
        if(tid<min(BLOCK_SIZE_K, k_nnz-rid*BLOCK_SIZE_K)){
            local_k_offset[tid] = csr_col[index_start + rid * BLOCK_SIZE_K + tid];
        }
        __syncthreads();
        #pragma unroll
        for(int k=A_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=A_TILE_ROW_STRIDE){
            float4 tmp_float4={0};
            if(k < k_nnz-rid*BLOCK_SIZE_K)
                tmp_float4 = FETCH_FLOAT4(csr_val[index_start*BLOCK_SIZE_M*1+rid*BLOCK_SIZE_M*BLOCK_SIZE_K+k*BLOCK_SIZE_M+A_BLOCK_COL_START]);
            FETCH_FLOAT4(As[OFFSET(k, A_BLOCK_COL_START, BLOCK_SIZE_M)]) = tmp_float4;
        }
        #pragma unroll
        for(int k=B_BLOCK_ROW_START; k<BLOCK_SIZE_K; k+=B_TILE_ROW_STRIDE){
            float4 tmp_float4 = {0};

            if(k<k_nnz-rid*BLOCK_SIZE_K)
                tmp_float4 = FETCH_FLOAT4(B[OFFSET(local_k_offset[k], bx*BLOCK_SIZE_N + B_BLOCK_COL_START, N)]);            
            FETCH_FLOAT4(Bs[OFFSET(k, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = tmp_float4; 
        }
        __syncthreads();
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j += 1){
                    // a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                    a_frag[j][i] = As[OFFSET(k+i, ty + vBLOCK_SIZE_M * j, BLOCK_SIZE_M)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_N; j += 1){
                    b_frag[j][i] = Bs[OFFSET(k+i, tx + vBLOCK_SIZE_N * j, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_N; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j++){
                    #pragma unroll
                    for(int k_in = 0; k_in < THREAD_SIZE_K; k_in++){
                        // accum[i][j] = fma(a_frag[j][k_in], b_frag[i][k_in], accum[i][j]);
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }
    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]);
        }
    }


}


void convert_bcsr(int * mask, float* dense_val, int M, int N, int block_h, int block_w, int *row_ptr, int* col_idx, float * val )
{
    vector<int> vr(M/block_h+1, 0);
    vector<vector<int>> vc(M/block_h+1, vector<int>());
    int block_nnz = 0;
    assert(M%block_h==0);
    assert(N%block_w==0);
    for(int rid=0; rid<M/block_h; rid++){
        for(int cid=0; cid<N/block_w; cid++){
            int flag = 0;
            for(int i=0; i<block_h; i++){
                for(int j=0; j<block_w; j++){
                    int _pos = (rid * block_h + i) * N + cid * block_w + j;
                    if(mask[_pos]>0)
                        flag = 1;
                }
            }
            if(flag){
                vc[rid].push_back(cid);
            }
        }
    }
    row_ptr[0]=0;
    for(int i=0;i<M/block_h;i++){
        row_ptr[i+1] = row_ptr[i] + vc[i].size();
        for(int j =0; j<vc[i].size(); j++){
            int _block_idx = row_ptr[i]+j;
            col_idx[_block_idx] = vc[i][j];
            for(int b_i=0; b_i<block_h; b_i++){
                for(int b_j=0; b_j<block_w; b_j++){
                    int pos_1 = _block_idx * block_h *block_w + b_i * block_w + b_j;
                    int pos_2 = (b_i + i * block_h) * N + (b_j + block_w * vc[i][j]);
                    val[pos_1] = dense_val[pos_2];
                }
            }
        }
    }

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
    printf("Blockwise sparsity %f \n", 1.0-1.0*row[h/block_h]/(h/block_h)/(w/block_w));
    for(int i=0;i<block_h*block_w;i++)
        if(mask[i])
            return false;
    return true;
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
bool verify_matmul_sdd(float * A, float * B, int M, int N)
{
    bool flag = true;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            int index = i * N + j;
            if(fabs(A[index]-B[index])>0.001){
                printf("%d %d : %f %f\n", i, j, A[index], B[index]);
                flag=false;
            }
        }
    }
    return flag;
}
int main(int argc, char*argv[])
{
    int M, K, N;
    M = 4096;
    K = 3072;
    N = 768;
    const int n_iter = 100;
    float sparsity_ratio = atof(argv[1]);
    // float sparsity_ratio = 0.99;
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 16;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 2;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 2;
    const int BLOCK_H = BLOCK_SIZE_M;
    const int BLOCK_W = 1;
    // const int BLOCK_W = 1;
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
    mask = (int*) malloc(sizeof(int) * M * K);
    row = (int*) malloc(sizeof(int) * (K+1));
    col = (int*) malloc(sizeof(int) *  M * K / BLOCK_H / BLOCK_W);
    val = (float*) malloc(sizeof(float) * M * K);
    init_mask_blockwise(mask, A, M, K, BLOCK_H, BLOCK_W, sparsity_ratio);
    init(B, K*N ,0);
    // init(A, M*K ,0);
    // // apply mask
    // for(int i=0; i< M*K; i++){
    //     A[i] = A[i] * mask[i];
    // }
    convert_bcsr(mask, A, M, K, BLOCK_H, BLOCK_W, row, col, val);
    assert (verify_bcsr(mask, A, M,K,BLOCK_H,BLOCK_W, row, col, val));
    int block_nnz = row[M/BLOCK_H];
    
    printf("Block NNZ: %d\n", block_nnz);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(float)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_extra_buffer, sizeof(float) * M * K));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int) * M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)*M * K / BLOCK_H / BLOCK_W, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    

    // KxM = KxN * (MxN)^T
    dim3 blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    printf("Test Condense-k on our block sparse template\n");
    CUDA_SAFE_CALL(cudaEventRecord(time_start));

    for(int run=0; run<n_iter; run++){
        BLOCK_SPARSE_MATMUL_SDD_CONDENSE_K<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gridDim, blockDim>>>(d_row, d_col, d_val, dB, dC, M, K, N, BLOCK_SIZE_M, 1);
     }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time= %.3f ms\n", msecTotal/n_iter);
    CUDA_SAFE_CALL(cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    // printf("csr_row[63]:%d csr_row[64]:%d\n", row[63], row[64]);

    // calculate_reference(M,K,N,A,B,refC);
    // verify_matmul_sdd(C, refC, M,N);
    return 0;
}