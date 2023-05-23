#ifndef _NMSPARSE_KERNEL_BLOCKWISE4x4_H_
#define _NMSPARSE_KERNEL_BLOCKWISE4x4_H_

#include "context.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>

namespace nmsparse
{
	__global__ void nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K3(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 128;
		const int BLOCK_SIZE_K = 128;
		const int THREAD_SIZE_M = 8;
		const int THREAD_SIZE_N = 4;

		const int ALIGN_K = 3;
		const int BANK_VAL = 32;

		const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
		const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
		extern __shared__ float shared_mem[];

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K;
		int *B_index_shared = reinterpret_cast<int *>(B_shared + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);

		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		int B_reg_index;
		float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES)
			{
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);

				*(float4 *)(B_index_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_index_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += ALIGN_K)
			{
				int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
				int B_reg_index_alignk = B_index_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N] % BANK_VAL + bank_idx * BANK_VAL;
				for (int i_in = 0; i_in < ALIGN_K; i_in += 1)
				{
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
						B_reg[k] = B_shared[(i + i_in) * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
					}
					B_reg_index = B_reg_index_alignk + i_in;
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_M; k += 1)
					{
						A_reg[k] = A_shared[(t_M * THREAD_SIZE_M + k) * BLOCK_SIZE_K + B_reg_index];
					}
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
#pragma unroll
						for (int j = 0; j < THREAD_SIZE_M; j += 1)
						{
							C_reg[j][k] += B_reg[k] * A_reg[j];
						}
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

	__global__ void nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4_K3(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 128;
		const int BLOCK_SIZE_K = 64;
		const int THREAD_SIZE_M = 8;
		const int THREAD_SIZE_N = 4;

		const int ALIGN_K = 3;
		const int BANK_VAL = 32;

		const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
		const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
		extern __shared__ float shared_mem[];

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K;
		int *B_index_shared = reinterpret_cast<int *>(B_shared + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);

		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		int B_reg_index;
		float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES)
			{
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);

				*(float4 *)(B_index_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_index_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += ALIGN_K)
			{
				int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
				int B_reg_index_alignk = B_index_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N] % BANK_VAL + bank_idx * BANK_VAL;
				for (int i_in = 0; i_in < ALIGN_K; i_in += 1)
				{
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
						B_reg[k] = B_shared[(i + i_in) * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
					}
					B_reg_index = B_reg_index_alignk + i_in;
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_M; k += 1)
					{
						A_reg[k] = A_shared[(t_M * THREAD_SIZE_M + k) * BLOCK_SIZE_K + B_reg_index];
					}
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
#pragma unroll
						for (int j = 0; j < THREAD_SIZE_M; j += 1)
						{
							C_reg[j][k] += B_reg[k] * A_reg[j];
						}
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

	__global__ void nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 128;
		const int BLOCK_SIZE_K = 128;
		const int THREAD_SIZE_M = 8;
		const int THREAD_SIZE_N = 4;

		const int ALIGN_K = 4;
		const int BANK_VAL = 32;

		const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
		const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;

		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
		extern __shared__ float shared_mem[];

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K;
		int *B_index_shared = reinterpret_cast<int *>(B_shared + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);

		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		int B_reg_index;
		float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES)
			{
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);

				*(float4 *)(B_index_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_index_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += ALIGN_K)
			{
				int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
				int B_reg_index_alignk = B_index_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N] % BANK_VAL + bank_idx * BANK_VAL;
				for (int i_in = 0; i_in < ALIGN_K; i_in += 1)
				{
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
						B_reg[k] = B_shared[(i + i_in) * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
					}
					B_reg_index = B_reg_index_alignk + i_in;
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_M; k += 1)
					{
						A_reg[k] = A_shared[(t_M * THREAD_SIZE_M + k) * BLOCK_SIZE_K + B_reg_index];
					}
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
#pragma unroll
						for (int j = 0; j < THREAD_SIZE_M; j += 1)
						{
							C_reg[j][k] += B_reg[k] * A_reg[j];
						}
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

	__global__ void nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4_K4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, const int M, const int N, const int K, const float sparsity)
	{
		const int BLOCK_SIZE_M = 32;
		const int BLOCK_SIZE_N = 128;
		const int BLOCK_SIZE_K = 64;
		const int THREAD_SIZE_M = 8;
		const int THREAD_SIZE_N = 4;

		const int ALIGN_K = 4;
		const int BANK_VAL = 32;

		const int BANK_NUM_PER_BLOCK = BLOCK_SIZE_K / BANK_VAL;
		const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
		const int LEN_OF_BANK_PER_SPARSE_BLOCK = BLOCK_SIZE_K_SPARSE / BANK_NUM_PER_BLOCK;
		int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
		int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

		const int A_THREADS_PER_ROW = BLOCK_SIZE_K / 4;
		const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;

		const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);

		const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
		const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
		extern __shared__ float shared_mem[];

		float *A_shared = shared_mem;
		float *B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K;
		int *B_index_shared = reinterpret_cast<int *>(B_shared + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
		
		float A_reg[THREAD_SIZE_M];
		float B_reg[THREAD_SIZE_N];
		int B_reg_index;
		float C_reg[THREAD_SIZE_M][THREAD_SIZE_N] = {0};

		int tid = threadIdx.x;

		int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
		int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);

		int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
		int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

		int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
		int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;

		for (int K_BLOCK_START = 0, K_SPARSE_BLOCK_START = 0; K_BLOCK_START < K; K_BLOCK_START += BLOCK_SIZE_K, K_SPARSE_BLOCK_START += BLOCK_SIZE_K_SPARSE)
		{
			float *A_global_ptr = g_vec + M_BLOCK_START * K + K_BLOCK_START;
			float *B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
			int *B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_M; i += A_STRIDES)
			{
				*(float4 *)(A_shared + (i + A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START) =
					*(float4 *)(A_global_ptr + (i + A_BLOCK_ROW_START) * K + A_BLOCK_COL_START);
			}

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES)
			{
				*(float4 *)(B_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);

				*(float4 *)(B_index_shared + (i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START) =
					*(float4 *)(B_index_global_ptr + (i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START);
			}

			__syncthreads();

#pragma unroll
			for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += ALIGN_K)
			{
				int bank_idx = i / LEN_OF_BANK_PER_SPARSE_BLOCK;
				int B_reg_index_alignk = B_index_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N] % BANK_VAL + bank_idx * BANK_VAL;
				for (int i_in = 0; i_in < ALIGN_K; i_in += 1)
				{
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
						B_reg[k] = B_shared[(i + i_in) * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
					}
					B_reg_index = B_reg_index_alignk + i_in;
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_M; k += 1)
					{
						A_reg[k] = A_shared[(t_M * THREAD_SIZE_M + k) * BLOCK_SIZE_K + B_reg_index];
					}
#pragma unroll
					for (int k = 0; k < THREAD_SIZE_N; k += 1)
					{
#pragma unroll
						for (int j = 0; j < THREAD_SIZE_M; j += 1)
						{
							C_reg[j][k] += B_reg[k] * A_reg[j];
						}
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

	template <typename dtype>
	cudaError_t nmsparseSpMMBW4x4(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);

	template <typename dtype>
	cudaError_t nmsparseSpMMBW4x4(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream)
	{
		assert(is_divisible(m, 32) && is_divisible(n, 128) && is_divisible(k, 64));
		const float sparsity = ctx.sparsity;
		const int M = m;
		const int N = n;
		const int K = k;
		const int w = int((1.0f - sparsity) * K);
		if (sparsity == 0.5f){
			const int BLOCK_SIZE_M = 32;
			const int BLOCK_SIZE_N = 128;
			const int BLOCK_SIZE_K = 64;
			const int THREAD_SIZE_M = 8;
			const int THREAD_SIZE_N = 4;
			const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
			dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
			dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
			int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
			nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4_K4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
		}else if(sparsity == 0.90625){
			const int BLOCK_SIZE_M = 32;
			const int BLOCK_SIZE_N = 128;
			const int BLOCK_SIZE_K = 128;
			const int THREAD_SIZE_M = 8;
			const int THREAD_SIZE_N = 4;
			const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
			dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
			dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
			int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
			nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K3<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
		}else{
			const int BLOCK_SIZE_M = 32;
			const int BLOCK_SIZE_N = 128;
			const int BLOCK_SIZE_K = 128;
			const int THREAD_SIZE_M = 8;
			const int THREAD_SIZE_N = 4;
			const int BLOCK_SIZE_K_SPARSE = int(BLOCK_SIZE_K * (1 - sparsity));
			dim3 dimBlock(int((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N)));
			dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
			int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE + BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
			nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K4<<<dimGrid, dimBlock, shared_mem_size>>>(mat_a_dense, mat_b_sparse_val, mat_b_sparse_idx, output, M, N, K, sparsity);
		}

		return cudaGetLastError();
	}
}

#endif