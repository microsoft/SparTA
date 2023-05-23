#ifndef _NMSPARSE_KERNEL_VECTORWISE32_H_
#define _NMSPARSE_KERNEL_VECTORWISE32_H_

#include "context.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

namespace nmsparse
{
	__global__ void nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x128_4x4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 32;
		const int BLOCK_SIZE_K = 128;
		const int THREAD_SIZE_M = 4;
		const int THREAD_SIZE_N = 4;
		extern __shared__ float shared_mem[];
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));

		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;

		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES)
			{
				int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + idx * M + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1)
			{
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_M; k += 1)
				{
					A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
				}
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_N; k += 1)
				{
					B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
				}
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_N; k += 1)
				{
#pragma unroll
					for (int j = 0; j < THREAD_SIZE_M; j += 1)
					{
						C_reg[k][j] += B_reg[k] * A_reg[j];
					}
				}
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_SIZE_N; i += 1)
		{
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_M; j += 1)
			{
				g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] =
					C_reg[i][j];
			}
		}
	}

	__global__ void nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x256_4x4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 32;
		const int BLOCK_SIZE_K = 256;
		const int THREAD_SIZE_M = 4;
		const int THREAD_SIZE_N = 4;
		extern __shared__ float shared_mem[];
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));

		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;

		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES)
			{
				int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + idx * M + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += 1)
			{
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_M; k += 1)
				{
					A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
				}
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_N; k += 1)
				{
					B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
				}
#pragma unroll
				for (int k = 0; k < THREAD_SIZE_N; k += 1)
				{
#pragma unroll
					for (int j = 0; j < THREAD_SIZE_M; j += 1)
					{
						C_reg[k][j] += B_reg[k] * A_reg[j];
					}
				}
			}
		}

#pragma unroll
		for (int i = 0; i < THREAD_SIZE_N; i += 1)
		{
#pragma unroll
			for (int j = 0; j < THREAD_SIZE_M; j += 1)
			{
				g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] =
					C_reg[i][j];
			}
		}
	}

	template <typename dtype>
	cudaError_t nmsparseSpMMVW32(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);

	template <typename dtype>
	cudaError_t nmsparseSpMMVW32(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream)
	{
		assert(is_divisible(m, 32) && is_divisible(n, 128) && is_divisible(k, 64));
		const float sparsity = ctx.sparsity;
		const int M = m;
		const int N = n;
		const int K = k;
		const int w = int((1.0f - sparsity) * K);
		if (sparsity == 0.5f){
			const int BLOCK_SIZE_M = 32;
			const int BLOCK_SIZE_N = 32;
			const int BLOCK_SIZE_K = 128;
			const int THREAD_SIZE_M = 4;
			const int THREAD_SIZE_N = 4;
			const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
			dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
			dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
			int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
			nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x128_4x4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
		}else{
			const int BLOCK_SIZE_M = 32;
			const int BLOCK_SIZE_N = 32;
			const int BLOCK_SIZE_K = 256;
			const int THREAD_SIZE_M = 4;
			const int THREAD_SIZE_N = 4;
			const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
			dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
			dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
			int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
			nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x256_4x4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
		}
		

		return cudaGetLastError();
	}
}

#endif