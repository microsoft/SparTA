#ifndef _NMSPARSE_KERNEL_ELEMENT_WISE_H_
#define _NMSPARSE_KERNEL_ELEMENT_WISE_H_

#include "context.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

// from MV_one_kernel_block_batch.cu
extern "C" __global__ void nmsparse_ew_gemv_simt_fp32_fp32_fp32_32x32x32(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_odata, int w, int h, int BLOCK_WIDTH, int NUM_THREADS, int VEC_WIDTH, const int minibatch, const int vecNum)
{
    int blockxInd;
    int vecInd;
    int blockElt; // holds the current block width

    if ((blockIdx.y + 1) * BLOCK_WIDTH <= w)
    {
        blockElt = BLOCK_WIDTH;
    }
    else
    {
        blockElt = w % BLOCK_WIDTH;
    }
    blockxInd = blockIdx.y * BLOCK_WIDTH;
    vecInd = blockIdx.y * VEC_WIDTH;

    unsigned int threadyInd = blockIdx.x * NUM_THREADS + threadIdx.x;
    extern __shared__ float vec_data[];

#pragma unroll
    for (int batch = 0; batch < minibatch; ++batch)
    {
#pragma unroll
        for (int i = 4 * threadIdx.x; i < VEC_WIDTH; i += 4 * NUM_THREADS)
        {
            *(float4 *)(vec_data + i + batch * VEC_WIDTH) = *(float4 *)(g_vec + vecInd + i + (batch + blockIdx.z * minibatch) * vecNum);
        }
    }

    __syncthreads();

    // due to a gemv kernel only handle m <= 8, so we use a constant shared memory to store.
    float sdata[8] = {0};

    float data_tmp = 0;
    int index_tmp = 0;

#pragma unroll
    for (int index = 0; index < blockElt; ++index)
    {
        data_tmp = g_mat_data[threadyInd + (index + blockxInd) * h];
        index_tmp = g_mat_index[threadyInd + (index + blockxInd) * h] - vecInd;

#pragma unroll
        for (int batch = 0; batch < minibatch; batch += 1)
        {
            sdata[batch] += data_tmp * vec_data[index_tmp + batch * VEC_WIDTH];
        }
    }

#pragma unroll
    for (int batch = 0; batch < minibatch; batch += 1)
    {
        atomicAdd(g_odata + h * (batch + blockIdx.z * minibatch) + threadyInd, sdata[batch]);
    }
}

// from balance_align.cu
extern "C" __global__ void nmsparse_ew_gemm_simt_fp32_fp32_fp32_32x32x32(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, int M, int K, int N, float sparsity)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K = 32;
    const int THREAD_SIZE_M = 16;
    const int THREAD_SIZE_N = 1;
    const int BANK_VAL = 32;

    const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
    const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
    const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

    int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
    int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

    const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;

    const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;

    __shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];

    float B_reg[THREAD_SIZE_N];
    int B_reg_index[THREAD_SIZE_N];
    float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

    int tid = threadIdx.x;

    int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
    int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;

    for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
    {
        float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES)
        {
            *(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) =
                *(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1)
        {
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1)
            {
                B_reg[k] = g_mat_data[(K_SPARSE_BLOCK_START + i) * N + N_BLOCK_START + t_N * THREAD_SIZE_N + k];
                B_reg_index[k] = g_mat_index[(K_SPARSE_BLOCK_START + i) * N + N_BLOCK_START + t_N * THREAD_SIZE_N + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k += 1)
            {
                int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
                int B_index = B_reg_index[k] % BANK_VAL + bank_idx * BANK_VAL;
#pragma unroll
                for (int j = 0; j < THREAD_SIZE_M; j += 1)
                {
                    C_reg[j][k] += B_reg[k] * A_shared[(t_M * THREAD_SIZE_M + j) * BLOCK_SIZE_K + B_index];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; i += 1)
    {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; j += 1)
        {
            g_data[(BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + i) * N + BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + j] = C_reg[i][j];
        }
    }
}

namespace nmsparse {

    bool is_one(const int x)
    {
        return 1 == x;
    }

    bool is_divisible(const int x, const int be_devide)
    {
        return 0 == (x % be_devide);
    }

    template <typename dtype>
    cudaError_t nmsparseSpMMEW(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);


    template <typename dtype>
    cudaError_t nmsparseSpMMEW(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream)
    {
        assert(is_one(m) || is_divisible(m, 32));
        const float sparsity = ctx.sparsity;
        const int M = m;
        const int N = n;
        const int K = k;

        if (is_one(m))
        {
            std::cout << "nmsparseSpMMEW: m is one" << std::endl;
            const int w = int((1.0f - sparsity) * K);
            const int h = N;
            const int vecNum = K;
            const int minibatch = M;
            const int BANK_VAL = 32;
            const int NUM_THREADS = 128;
            const int NUM_BANK = K / BANK_VAL;
            const int BLOCK_WIDTH = w / NUM_BANK;
            const int BLOCK_minibatch = M;
            const int VEC_WIDTH = vecNum * BLOCK_WIDTH / w;
            dim3 dimBlock(NUM_THREADS);
            dim3 dimGrid(h / NUM_THREADS, w / BLOCK_WIDTH, M / BLOCK_minibatch);
            nmsparse_ew_gemv_simt_fp32_fp32_fp32_32x32x32<<<dimGrid, dimBlock, BLOCK_minibatch * VEC_WIDTH * sizeof(float)>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, w, h, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch, vecNum);
        } else {
            const int w = int((1.0f - sparsity) * K);
            const int h = N;
            const int vecNum = K;
            const int minibatch = M;
            int M = minibatch, N = h, K = vecNum, K_sparse = w;
            const int BLOCK_SIZE_M = 32;
            const int BLOCK_SIZE_N = 32;
            const int BLOCK_SIZE_K = 32;
            const int THREAD_SIZE_M = 16;
            const int THREAD_SIZE_N = 1;

            dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
            dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
            dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
            dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
            nmsparse_ew_gemm_simt_fp32_fp32_fp32_32x32x32<<<dimGrid, dimBlock>>>(g_vec, g_mat_data, g_mat_index, g_result, M, K, K_sparse, N);
        }
        return cudaGetLastError();
    }
}

#endif