#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>
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


template <
    const int BLOCK_SIZE_M, // 64
    const int BLOCK_SIZE_K, // 8
    const int BLOCK_SIZE_N, // 128
    const int THREAD_SIZE_M, // 8
    const int THREAD_SIZE_K, // 4
    const int THREAD_SIZE_N  // 8
>
__global__ void BLOCK_SPARSE_MATMUL_BIAS(float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N){
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_K / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = W_row[bx], index_end = W_row[bx+1];

    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    float4 tmp_float4;
    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int tile_idx = W_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(A[OFFSET(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START, tile_idx+A_BLOCK_COL_START, K)]);
        }
        /*
        for(int k = 0; k < BLOCK_SIZE_K; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_M)]) = 
                FETCH_FLOAT4(A[OFFSET(tile_idx+k+A_BLOCK_ROW_START, by*BLOCK_SIZE_M+A_BLOCK_COL_START, M)]);
        }
        */

        // #pragma unroll
        // for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
        //     FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
        //         FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        //         // FETCH_FLOAT4(B[OFFSET(tile_idx+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N+B_BLOCK_COL_START, N)]);
        // }

        #pragma unroll
        for(int k=0; k < BLOCK_SIZE_N; k+= B_TILE_ROW_STRIDE){
            // transpose here
            tmp_float4 =  FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_K + B_BLOCK_COL_START]);
            Bs[OFFSET(B_BLOCK_COL_START, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.x;
            Bs[OFFSET(B_BLOCK_COL_START+1, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.y;
            Bs[OFFSET(B_BLOCK_COL_START+2, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.z;
            Bs[OFFSET(B_BLOCK_COL_START+3, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.w;
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

    float bias_local[THREAD_SIZE_N];
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        bias_local[thread_x] = bias[BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N];
    }

    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]) + bias_local[thread_x];
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
__global__ void BLOCK_SPARSE_MATMUL(float* A, float* W_val, int* W_row, int* W_col, float* C, int M, int K, int N){
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_K / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = W_row[bx], index_end = W_row[bx+1];

    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    float4 tmp_float4;
    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int tile_idx = W_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                FETCH_FLOAT4(A[OFFSET(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START, tile_idx+A_BLOCK_COL_START, K)]);
        }
        /*
        for(int k = 0; k < BLOCK_SIZE_K; k += A_TILE_ROW_STRIDE){
            FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_M)]) = 
                FETCH_FLOAT4(A[OFFSET(tile_idx+k+A_BLOCK_ROW_START, by*BLOCK_SIZE_M+A_BLOCK_COL_START, M)]);
        }
        */

        // #pragma unroll
        // for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
        //     FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
        //         FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        //         // FETCH_FLOAT4(B[OFFSET(tile_idx+k+B_BLOCK_ROW_START, bx*BLOCK_SIZE_N+B_BLOCK_COL_START, N)]);
        // }

        #pragma unroll
        for(int k=0; k < BLOCK_SIZE_N; k+= B_TILE_ROW_STRIDE){
            // transpose here
            tmp_float4 =  FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_K + B_BLOCK_COL_START]);
            Bs[OFFSET(B_BLOCK_COL_START, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.x;
            Bs[OFFSET(B_BLOCK_COL_START+1, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.y;
            Bs[OFFSET(B_BLOCK_COL_START+2, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.z;
            Bs[OFFSET(B_BLOCK_COL_START+3, k+B_BLOCK_ROW_START, BLOCK_SIZE_N)] = tmp_float4.w;
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



void dynamic_forward_function(float* activation, int* row_ptr, int* col_idx,
                    float * val, float* bias, int M, int K, int N, int block_h, int block_w, float* output)
{

    // sparse x dense
    // M: seq_length K: seq_length N:hidden dim
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;
    
    assert(BLOCK_SIZE_N == block_h);
    assert(BLOCK_SIZE_K == block_w);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    // float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N
    BLOCK_SPARSE_MATMUL_BIAS<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gridDim, blockDim>>>(activation, val, row_ptr, col_idx, output, bias, M, K, N);
    
}

void dynamic_backward_function(float* grad_in, int * row_ptr, int *col_idx, float* val, int M, int K, int N, int block_h, int block_w, float* grad_out)
{
        // sparse x dense
    // M: seq_length K: seq_length N:hidden dim
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_K = 4;
    const int THREAD_SIZE_N = 4;
    
    assert(BLOCK_SIZE_N == block_h);
    assert(BLOCK_SIZE_K == block_w);
    dim3 gridDim(N/BLOCK_SIZE_N, M/BLOCK_SIZE_M);
    dim3 blockDim(BLOCK_SIZE_N/THREAD_SIZE_N, BLOCK_SIZE_M/THREAD_SIZE_M);
    // float* A, float* W_val, int* W_row, int* W_col, float* C, float *bias, int M, int K, int N
    BLOCK_SPARSE_MATMUL<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_M, THREAD_SIZE_K, THREAD_SIZE_N><<<gridDim, blockDim>>>(grad_in, val, row_ptr, col_idx, grad_out, M, K, N);
    
}

at::Tensor dynamic_sparse_linear_forward(
    torch::Tensor activation,
    torch::Tensor row_ptr,
    torch::Tensor col_index,
    torch::Tensor val,
    torch::Tensor bias,
    int M, int K, int N, int block_h, int block_w
)
{
    cudaSetDevice(activation.get_device());
    // Q, K, V should have the same shape which is {batchsize, seq_length, hidden_dim}
    torch::Tensor output = torch::empty({M, N}, activation.options());
    
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "dynamic_sparse_linear", ([&]
                            { dynamic_forward_function(
                                    activation.data_ptr<float>(),
                                    row_ptr.data_ptr<int>(),
                                    col_index.data_ptr<int>(),
                                    val.data_ptr<float>(),
                                    bias.data_ptr<float>(),
                                    M, K, N, block_h, block_w,
                                    output.data_ptr<float>()
                                ); }));
    return output;
}

vector<at::Tensor> dynamic_sparse_linear_backward(
    torch::Tensor activation,
    torch::Tensor row_ptr,
    torch::Tensor col_index,
    torch::Tensor val,
    torch::Tensor grad_c,
    int M, int K, int N, int block_h, int block_w
)
{
    cudaSetDevice(activation.get_device());
    // torch::Tensor w_grad = torch.empty({M,N}, activation.options());
    torch::Tensor a_grad = torch::zeros_like(activation);
    torch::Tensor w_grad = at::matmul(grad_c.t(), activation);
    printf("M, K, N: %d %d %d\n", M, K, N);
    AT_DISPATCH_FLOATING_TYPES(activation.type(), "dynamic_sparse_linear", ([&]
        { dynamic_backward_function(
                grad_c.data_ptr<float>(),
                row_ptr.data_ptr<int>(),
                col_index.data_ptr<int>(),
                val.data_ptr<float>(),
                M, K, N, block_h, block_w,
                a_grad.data_ptr<float>()
            ); }));
    vector<torch::Tensor> grads({a_grad, w_grad});
    return grads;
}