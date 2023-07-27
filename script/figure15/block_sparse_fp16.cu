#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <sstream>
#include <string>
#include <assert.h>
using namespace std;
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])


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



void init_mask_blockwise(int * mask, half * value, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    memset(mask, 0, sizeof(int)*M*N);
    memset(value, 0, sizeof(half)*M*N);
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (pro >= sparsity)
            {
                int pos;
                for (int b_i=0; b_i<block_h; b_i++){
                    for(int b_j=0; b_j<block_w; b_j++){
                        pos = (i * block_h + b_i)*N + j* block_w + b_j;
                        mask[pos]=1;
                        float tmp_float = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        // value[pos] = static_cast<half>(1.0);
                        value[pos] = static_cast<half>(tmp_float);
                        // value[pos] = 1;
                    }
                }
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);

}
void init(half * ptr, size_t length, float sparsity)
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
            float tmp_float = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            ptr[i] = static_cast<half>(tmp_float);
            // ptr[i] = static_cast<half>(1.0);
        }
    }
}


void convert_bcsr(int * mask, half* dense_val, int M, int N, int block_h, int block_w, int *row_ptr, int* col_idx, half * val )
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
bool verify_bcsr(int * mask, half * data, int h, int w, int block_h, int block_w, int* row, int * col, half* values)
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

                        if(abs(static_cast<float>(data[offset])-static_cast<float>(values[csr_offset]))>1e-6)
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



void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K)
{

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float psum = 0.0;
            for (int k = 0; k < K; k++)
            {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

__global__ void HGEMM(
    int * csr_row, int * csr_col, half *__restrict__ csr_val,
    half *__restrict__ B, half *__restrict__ C,
    const int M, const int N, const int K, const int BLOCK_H, const int BLOCK_W)
{

    const int BM = 32;
    const int BN = 32;
    const int BK = 64;
    const int w_per_row = BN/16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    int wy = wid / w_per_row;
    int wx = wid % w_per_row;
    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];



    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c;
    wmma::fill_fragment(frag_c, 0.0);
    // load to the shared memory
    const int A_THREAD_PER_ROW = BK / 8; // 1 float4 = 8 half
    const int B_THREAD_PER_ROW = BN / 8;
    const int A_TILE_ROW_STRIDE = blockDim.x / A_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = blockDim.x / B_THREAD_PER_ROW;
    const int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    const int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;
    const int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 8;
    const int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 8;
    
    int index_start = csr_row[by];
    int index_end = csr_row[by+1];
    for(int tile_block_idx = index_start; tile_block_idx< index_end; tile_block_idx++){
        int col_pos = csr_col[tile_block_idx] * BK;
        #pragma unroll
        for(int k = 0; k < BM; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[k+A_BLOCK_ROW_START][A_BLOCK_COL_START]) = FETCH_FLOAT4(csr_val[tile_block_idx * BM * BK + OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BK)]);
        }

        #pragma unroll
        for(int k = 0; k < BK; k += B_TILE_ROW_STRIDE){
            // FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BN+BPAD)]) = FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BN + B_BLOCK_COL_START, N)]);
            FETCH_FLOAT4(Bs[k+B_BLOCK_ROW_START][B_BLOCK_COL_START]) = FETCH_FLOAT4(B[OFFSET(col_pos+k+B_BLOCK_ROW_START, bx*BN + B_BLOCK_COL_START, N)]);
        }

        __syncthreads();
        // #pragma unroll
        // for(int k_step=0; k_step<BK/16; k_step++){
        //     wmma::load_matrix_sync(frag_a[k_step], &As[wy*16][k_step*16], BK + APAD);
        //     wmma::load_matrix_sync(frag_b[k_step], &Bs[k_step*16][wx*16], BN + BPAD);
        // }
        // #pragma unroll
        // for(int k_step=0; k_step<BK/16; k_step++){
        //     wmma::mma_sync(frag_c, frag_a[k_step], frag_b[k_step], frag_c);
        // }
        #pragma unroll
        for(int k_step=0; k_step<BK/16; k_step++){
            wmma::load_matrix_sync(frag_a[0], &As[wy*16][k_step*16], BK + APAD);
            wmma::load_matrix_sync(frag_b[0], &Bs[k_step*16][wx*16], BN + BPAD);
            wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
        
        }
        __syncthreads();

    }
    int write_offset = (by * BM + wy * 16) * N + bx * BN + wx * 16;
    wmma::store_matrix_sync(&C[write_offset], frag_c, N, wmma::mem_row_major);
}



float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm)(half *, half *, half *, int, int, int),
    int M, int N, int K)
{

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm)(half *, half *, half *, int, int, int),
    int M, int N, int K, int repeat)
{

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
    {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}


int main(int argc, char*argv[])
{
    int M, K, N;
    M = 4096;
    K = 4096;
    N = 4096;
    const int n_iter = 1000;
    float sparsity_ratio = atof(argv[1]);
    // float sparsity_ratio = 0.99;
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_H = BLOCK_SIZE_M;
    const int BLOCK_W = BLOCK_SIZE_K;

    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    half * A, *B, *C, *val, *refC;
    half * dA, *dB, *dC, *d_val;
    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col;
    A = (half*) malloc(sizeof(half) * M * K);
    B = (half*) malloc(sizeof(half) * K * N);
    C = (half*) malloc(sizeof(half) * M * N);
    refC = (half*) malloc(sizeof(half) * M * N);
    mask = (int*) malloc(sizeof(int) * M * K);
    row = (int*) malloc(sizeof(int) * (K+1));
    col = (int*) malloc(sizeof(int) *  M * K / BLOCK_H / BLOCK_W);
    val = (half*) malloc(sizeof(half) * M * K);
    init_mask_blockwise(mask, A, M, K, BLOCK_H, BLOCK_W, sparsity_ratio);
    init(B, K*N ,0);

    convert_bcsr(mask, A, M, K, BLOCK_H, BLOCK_W, row, col, val);
    assert (verify_bcsr(mask, A, M,K,BLOCK_H,BLOCK_W, row, col, val));
    int block_nnz = row[M/BLOCK_H];
    
    printf("Block NNZ: %d\n", block_nnz);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(half) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(half) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(half) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(half) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(half)*M*N));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int) * M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(half)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(half)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(M+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)*M * K / BLOCK_H / BLOCK_W, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(half) * M * K, cudaMemcpyHostToDevice));

    

    // KxM = KxN * (MxN)^T
    dim3 blockDim(128);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    printf("Test Condense-k on our block sparse template\n");
    CUDA_SAFE_CALL(cudaEventRecord(time_start));

    for(int run=0; run<n_iter; run++){
        HGEMM<<<gridDim, blockDim>>>(d_row, d_col, d_val, dB, dC, M, K, N, BLOCK_H, BLOCK_W);
     }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time= %.3f ms\n", msecTotal/n_iter);
    CUDA_SAFE_CALL(cudaMemcpy(C, dC, sizeof(half)*M*N, cudaMemcpyDeviceToHost));
    // printf("csr_row[63]:%d csr_row[64]:%d\n", row[63], row[64]);
    cpuF16F16Gemm(A, B, refC, M, N, K);
    float max_error = -1000000.0;
    for(int i=0; i<M*N; i++){
        float tmp_err = abs((float)refC[i] - (float)C[i]);
        max_error = max(tmp_err, max_error);
    }
    printf("max error:%f \n", max_error);
    // calculate_reference(M,K,N,A,B,refC);
    // verify_matmul_sdd(C, refC, M,N);
    return 0;
}