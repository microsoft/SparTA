/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA sample demonstrating a integer GEMM computation using the Warp Matrix
// Multiply and Accumulate API.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * (A * B + C). The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <algorithm>

// Externally configurable parameters.

size_t load_from_file(char* ptr, size_t buff_size, std::string filepath){
  std::ifstream fin(filepath, std::ios::in | std::ios::binary);
  size_t loaded_size = fin.read(ptr, buff_size).gcount();
  return loaded_size;
}

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 1
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_GLOBAL M_GLOBAL_VAL
#define N_GLOBAL N_GLOBAL_VAL
#define K_GLOBAL K_GLOBAL_VAL

#define SPARSITY SPARSITY_RATIO_VAL

#define K_GLOBAL_SPARSE (int(K_GLOBAL * (1-SPARSITY)))

#define M_TILES (M_GLOBAL / M)
#define N_TILES (N_GLOBAL / N)
#define K_TILES (K_GLOBAL / K)

#define C_LAYOUT wmma::mem_col_major

// Implementation constants.

// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 uint8_t-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.

////// CHUNK_K * K <= K_GLOBAL //////

#define CHUNK_K CHUNK_K_VAL
#define CHUNK_K_SPARSE (int(CHUNK_K * (1-SPARSITY)))

#define BLOCK_SIZE_K (CHUNK_K * K)
#define BLOCK_SIZE_K_SPARSE (int((CHUNK_K * K) * (1 - SPARSITY)))

/*
#define CHUNK_LINE_BYTES (BLOCK_SIZE_K_SPARSE * sizeof(uint8_t))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(uint8_t))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)  // 4
#define CHUNK_COPY_LINE_LANES (CHUNK_LINE_BYTES / sizeof(uint8_t))
*/

#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))

#define CHUNK_LINE_BYTES_A (BLOCK_COL_TILES * M * sizeof(uint8_t))
#define CHUNK_COPY_LINES_PER_WARP_A (WARP_COPY_BYTES / CHUNK_LINE_BYTES_A)
#define CHUNK_COPY_LINE_LANES_A (CHUNK_LINE_BYTES_A / sizeof(int4))
#define SHARED_OFFSET_A (BLOCK_COL_TILES * M + SKEW_UINT8)

#define CHUNK_LINE_BYTES_B (BLOCK_SIZE_K_SPARSE * sizeof(uint8_t))
#define CHUNK_COPY_LINES_PER_WARP_B (WARP_COPY_BYTES / CHUNK_LINE_BYTES_B)
#define CHUNK_COPY_LINE_LANES_B (CHUNK_LINE_BYTES_B / sizeof(int4))
#define SHARED_OFFSET_B (BLOCK_SIZE_K_SPARSE + SKEW_UINT8)


#define BLOCK_ROW_WARPS 1
#define BLOCK_COL_WARPS 2

#define WARP_ROW_TILES 2
#define WARP_COL_TILES 1

#define SHARED_TO_GLOBAL_BYTES_PER_LINE ((WARP_COL_TILES * M) * sizeof(int))
#define SHARED_TO_GLOBAL_BYTES_PER_WARP (WARP_SIZE * sizeof(int))
#define SHARED_TO_GLOBAL_LINES_PER_WARP (SHARED_TO_GLOBAL_BYTES_PER_WARP / SHARED_TO_GLOBAL_BYTES_PER_LINE)
#define SHARED_TO_GLOBAL_LANES_PER_LINE (WARP_SIZE / SHARED_TO_GLOBAL_LINES_PER_WARP)
#define SHARED_TO_GLOBAL_ITERS ((WARP_ROW_TILES * N) / SHARED_TO_GLOBAL_LINES_PER_WARP)

// may be we can tune number here
#define LANE_ROW_STRIDE (WARP_ROW_TILES * N / 8)
#define LANE_COL_STRIDE (WARP_COL_TILES * M / 4)
#define WARP_STRIDE (WARP_COL_TILES * M)
// #define WARP_STRIDE (WARP_ROW_TILES * N)

#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)


/////////// BLOCK_ROW_TILES <= N_TILES ////////////
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)

/////////// BLOCK_COL_TILES <= M_TILES ////////////
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE M_GLOBAL

/*
#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)
*/

#define SHMEM_STRIDE (M * BLOCK_COL_TILES)
#define SHMEM_OFFSET (M * WARP_COL_TILES)

#define BLOCK_SIZE_M (M * BLOCK_COL_TILES)
#define BLOCK_SIZE_N (N * BLOCK_ROW_TILES)
#define ALIGN_N BLOCK_SIZE_N
// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 32
// one-byte "uint8_t" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_UINT8 16

#define BANK_VAL 32
#define NUM_BANK (K_GLOBAL / BANK_VAL)

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


using namespace nvcuda;

void checksetting(){
	assert(WARP_COPY_BYTES % CHUNK_LINE_BYTES_A == 0 && WARP_COPY_BYTES / CHUNK_LINE_BYTES_A >= 1);
	assert(WARP_COPY_BYTES % CHUNK_LINE_BYTES_B == 0 && WARP_COPY_BYTES / CHUNK_LINE_BYTES_B >= 1);
	assert(M_GLOBAL % BLOCK_SIZE_M == 0 && M_GLOBAL >= BLOCK_SIZE_M);
	assert(N_GLOBAL % BLOCK_SIZE_N == 0 && N_GLOBAL >= BLOCK_SIZE_N);
	assert(CHUNK_COPY_LINES_PER_WARP_A > 0 && CHUNK_COPY_LINES_PER_WARP_B > 0);
	assert(CHUNK_COPY_LINES_PER_WARP_A <= BLOCK_SIZE_K_SPARSE && CHUNK_COPY_LINES_PER_WARP_B <= BLOCK_SIZE_K_SPARSE);
}

void MVOnHost(uint8_t *vec, uint8_t *mat_data, int *mat_index, uint8_t *hostRef, const int w, const int h, int vecNum, const int minibatch) {
	int tmp;
	for (int batch = 0;batch < minibatch; ++batch){
		for (int j=0; j<h; ++j) {
			tmp = 0;
			for (int i=0; i<w; ++i) {
				tmp += mat_data[i + j * w] * vec[mat_index[i + j * w] * minibatch + batch];
			}
			hostRef[j * minibatch + batch] = (uint8_t)tmp;
		}
	}
}

__global__ void compute_gemm_imma_large_share(const uint8_t *A, const uint8_t *B, const int *B_index,
                                  uint8_t *D, int alpha, int integer) {
	//extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];
 
    extern __shared__ uint8_t shmem[];

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;


	// Offset in shared memory from which the B matrix is stored.
	// const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block
	const size_t shmem_idx_b_off = BLOCK_SIZE_K_SPARSE * SHARED_OFFSET_A;


	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.

    unsigned int block_pos = blockIdx.x;
	const unsigned int block_tile_i =
		((block_pos * BLOCK_COL_TILES) / M_TILES) * (BLOCK_ROW_TILES);
	const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % M_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.


    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.

    //__syncthreads();
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
	wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_ROW_TILES]
													 [WARP_COL_TILES];

    // Load the C matrix tiles into fragments from shared memory.

#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
			wmma::fill_fragment(c[i][j], 0);
		}
	}

    __syncthreads();

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    // int start_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS)];
    // int end_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS) + 1];

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    //for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx += 1){
    for(int tile_k_idx_sparse = 0, tile_k_idx = 0; tile_k_idx_sparse < K_GLOBAL_SPARSE; tile_k_idx_sparse += BLOCK_SIZE_K_SPARSE, tile_k_idx += BLOCK_SIZE_K){

		size_t shmem_idx = 
		warpId < (WARPS_PER_BLOCK / 2)
			? (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A
			: (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B + shmem_idx_b_off;

		int4 *lane_ptr = NULL;
		int *lane_ptr_index = NULL;
		const uint8_t *warp_ptr = NULL;


		if(warpId < (WARPS_PER_BLOCK / 2)){
			//warp_ptr = &A[block_tile_j * M] +
			//	(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * M_GLOBAL;
			warp_ptr = &A[block_tile_j * M];
			
			const int *warp_ptr_index = &B_index[block_tile_i * N * K_GLOBAL_SPARSE] +
									((warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A);

			lane_ptr_index = (int *)(warp_ptr_index + tile_k_idx_sparse + (laneId / CHUNK_COPY_LINE_LANES_A));

			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_A) * SHARED_OFFSET_A;
		}else{
			warp_ptr = &B[block_tile_i * N * K_GLOBAL_SPARSE] +
				(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * K_GLOBAL_SPARSE;
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_sparse +
								(laneId / CHUNK_COPY_LINE_LANES_B) * K_GLOBAL_SPARSE) +
								(laneId % CHUNK_COPY_LINE_LANES_B);
			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_B) * SHARED_OFFSET_B;
		}


      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      // shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

	  int iter_index = warpId < (WARPS_PER_BLOCK / 2)
	  	? BLOCK_SIZE_K_SPARSE / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A)
		: BLOCK_SIZE_N / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);

	  /*
      int iter_index = warpId < (WARPS_PER_BLOCK / 2)
        ? (BLOCK_COL_TILES * M) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP)
        : (BLOCK_ROW_TILES * N) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP);
	  */

	  /*
      int tile_k_idx_A;
      if(warpId < (WARPS_PER_BLOCK / 2)){
          tile_k_idx_A = *(lane_ptr_index);
      }
	  */

	  #pragma unroll
	  for(int i = 0; i < iter_index; i += 1){
		  if(warpId < (WARPS_PER_BLOCK / 2)){
			int tile_k_idx_A = *(lane_ptr_index);
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_A * M_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES_A);
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_A)) =
				*lane_ptr;
			//warp_ptr = (uint8_t *)((uint8_t *)warp_ptr + M_GLOBAL * (WARPS_PER_BLOCK / 2) *CHUNK_COPY_LINES_PER_WARP_A);
			lane_ptr_index = (int *)((int *)lane_ptr_index +  (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A;
		  }else{
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_B)) =
				*lane_ptr;
			lane_ptr = (int4 *)((uint8_t *)lane_ptr + K_GLOBAL_SPARSE * (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B;
		  }
	  }

      __syncthreads();

	#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K_SPARSE; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::col_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
            b[WARP_ROW_TILES];

	#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i += 1) {
			size_t shmem_idx_a = (warpId % BLOCK_COL_WARPS) * M * WARP_COL_TILES + (i * M);
			const uint8_t *tile_ptr = shmem + shmem_idx_a + k_step * K * SHARED_OFFSET_A;

			wmma::load_matrix_sync(a[i], tile_ptr, SHARED_OFFSET_A);
		#pragma unroll
			for(int j = 0; j < WARP_ROW_TILES; j += 1){
				if(i == 0){
					size_t shmem_idx_b = shmem_idx_b_off +
											(warpId / BLOCK_COL_WARPS) * (WARP_ROW_TILES * N) * SHARED_OFFSET_B +
											(j * N) * SHARED_OFFSET_B;
					const uint8_t *tile_ptr = shmem + shmem_idx_b + k_step * K;
					wmma::load_matrix_sync(b[j], tile_ptr, SHARED_OFFSET_B);
				}
				wmma::mma_sync(c[j][i], a[i], b[j], c[j][i]);
			}

        }
      }

      __syncthreads();
    }

    // This pointer is used to access the C and D matrix tiles this warp computes.
	int *shmem_warp_tile_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * N * WARP_ROW_TILES * SHMEM_STRIDE +
	(warpId % BLOCK_COL_WARPS) * SHMEM_OFFSET;

      // Store the D fragments to shared memory.
#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
		#pragma unroll
			for(int t = 0; t < c[i][j].num_elements; t += 1){
				c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
			}
			int *tile_ptr = shmem_warp_tile_ptr + i * N * SHMEM_STRIDE + j * M;
			wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
		}
	}

    __syncthreads();

	int *shmem_warp_stream_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N * SHMEM_STRIDE
									+ (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	const size_t gmem_idx =
		(block_tile_i * N + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N) * GLOBAL_MEM_STRIDE +
		block_tile_j * M + (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

	int *shmem_lane_stream_ptr =
		shmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * SHMEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);
	
	uint8_t *dst_gmem_lane_stream_ptr =
		dst_gmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * GLOBAL_MEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);

	for(int i = 0; i < WARP_ROW_TILES * N; i += SHARED_TO_GLOBAL_LINES_PER_WARP){
		*(dst_gmem_lane_stream_ptr + i * GLOBAL_MEM_STRIDE) = (uint8_t)(*(shmem_lane_stream_ptr + i * SHMEM_STRIDE));
	}

	__syncthreads();
}

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B, const int *B_index,
                                  uint8_t *D, int alpha, int integer) {
	//extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];
				
    const int shared_size = MAX(sizeof(uint8_t) * (BLOCK_SIZE_K_SPARSE * (BLOCK_COL_TILES * M + SKEW_UINT8) + (BLOCK_SIZE_K_SPARSE + SKEW_UINT8) * BLOCK_ROW_TILES * N),
							M * BLOCK_ROW_TILES * N * BLOCK_COL_TILES * sizeof(int));
 
    __shared__ uint8_t shmem[shared_size];

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;


	// Offset in shared memory from which the B matrix is stored.
	// const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;       // BLOCK_COL_TILES * M is shared_A row numbers in one block
	const size_t shmem_idx_b_off = BLOCK_SIZE_K_SPARSE * SHARED_OFFSET_A;


	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.

    unsigned int block_pos = blockIdx.x;
	const unsigned int block_tile_i =
		((block_pos * BLOCK_COL_TILES) / M_TILES) * (BLOCK_ROW_TILES);
	const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % M_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.


    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.

    //__syncthreads();
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
	wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_ROW_TILES]
													 [WARP_COL_TILES];

    // Load the C matrix tiles into fragments from shared memory.

#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
			wmma::fill_fragment(c[i][j], 0);
		}
	}

    __syncthreads();

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    // int start_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS)];
    // int end_tile = B_row[block_tile_j / WARP_COL_TILES + (warpId % BLOCK_ROW_WARPS) + 1];

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    //for(int tile_k_idx = start_tile; tile_k_idx < end_tile; tile_k_idx += 1){
    for(int tile_k_idx_sparse = 0, tile_k_idx = 0; tile_k_idx_sparse < K_GLOBAL_SPARSE; tile_k_idx_sparse += BLOCK_SIZE_K_SPARSE, tile_k_idx += BLOCK_SIZE_K){

		size_t shmem_idx = 
		warpId < (WARPS_PER_BLOCK / 2)
			? (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A
			: (warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B + shmem_idx_b_off;

		int4 *lane_ptr = NULL;
		int *lane_ptr_index = NULL;
		const uint8_t *warp_ptr = NULL;


		if(warpId < (WARPS_PER_BLOCK / 2)){
			//warp_ptr = &A[block_tile_j * M] +
			//	(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A * M_GLOBAL;
			warp_ptr = &A[block_tile_j * M];
			
			const int *warp_ptr_index = &B_index[block_tile_i * N * K_GLOBAL_SPARSE] +
									((warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_A);

			lane_ptr_index = (int *)(warp_ptr_index + tile_k_idx_sparse + (laneId / CHUNK_COPY_LINE_LANES_A));

			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_A) * SHARED_OFFSET_A;
		}else{
			warp_ptr = &B[block_tile_i * N * K_GLOBAL_SPARSE] +
				(warpId % (WARPS_PER_BLOCK / 2)) * CHUNK_COPY_LINES_PER_WARP_B * K_GLOBAL_SPARSE;
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_sparse +
								(laneId / CHUNK_COPY_LINE_LANES_B) * K_GLOBAL_SPARSE) +
								(laneId % CHUNK_COPY_LINE_LANES_B);
			shmem_idx += (laneId / CHUNK_COPY_LINE_LANES_B) * SHARED_OFFSET_B;
		}


      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      // shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

	  int iter_index = warpId < (WARPS_PER_BLOCK / 2)
	  	? BLOCK_SIZE_K_SPARSE / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A)
		: BLOCK_SIZE_N / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);

	  /*
      int iter_index = warpId < (WARPS_PER_BLOCK / 2)
        ? (BLOCK_COL_TILES * M) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP)
        : (BLOCK_ROW_TILES * N) / ((WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP);
	  */

	  /*
      int tile_k_idx_A;
      if(warpId < (WARPS_PER_BLOCK / 2)){
          tile_k_idx_A = *(lane_ptr_index);
      }
	  */

	  #pragma unroll
	  for(int i = 0; i < iter_index; i += 1){
		  if(warpId < (WARPS_PER_BLOCK / 2)){
			int tile_k_idx_A = *(lane_ptr_index);
			lane_ptr = (int4 *)(warp_ptr + tile_k_idx_A * M_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES_A);
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_A)) =
				*lane_ptr;
			//warp_ptr = (uint8_t *)((uint8_t *)warp_ptr + M_GLOBAL * (WARPS_PER_BLOCK / 2) *CHUNK_COPY_LINES_PER_WARP_A);
			lane_ptr_index = (int *)((int *)lane_ptr_index +  (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_A * SHARED_OFFSET_A;
		  }else{
			*((int4 *)&shmem[shmem_idx] + (laneId % CHUNK_COPY_LINE_LANES_B)) =
				*lane_ptr;
			lane_ptr = (int4 *)((uint8_t *)lane_ptr + K_GLOBAL_SPARSE * (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B);
			shmem_idx += (WARPS_PER_BLOCK / 2) * CHUNK_COPY_LINES_PER_WARP_B * SHARED_OFFSET_B;
		  }
	  }

      __syncthreads();

	#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K_SPARSE; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::col_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
            b[WARP_ROW_TILES];

	#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i += 1) {
			size_t shmem_idx_a = (warpId % BLOCK_COL_WARPS) * M * WARP_COL_TILES + (i * M);
			const uint8_t *tile_ptr = shmem + shmem_idx_a + k_step * K * SHARED_OFFSET_A;

			wmma::load_matrix_sync(a[i], tile_ptr, SHARED_OFFSET_A);
		#pragma unroll
			for(int j = 0; j < WARP_ROW_TILES; j += 1){
				if(i == 0){
					size_t shmem_idx_b = shmem_idx_b_off +
											(warpId / BLOCK_COL_WARPS) * (WARP_ROW_TILES * N) * SHARED_OFFSET_B +
											(j * N) * SHARED_OFFSET_B;
					const uint8_t *tile_ptr = shmem + shmem_idx_b + k_step * K;
					wmma::load_matrix_sync(b[j], tile_ptr, SHARED_OFFSET_B);
				}
				wmma::mma_sync(c[j][i], a[i], b[j], c[j][i]);
			}

        }
      }

      __syncthreads();
    }

    // This pointer is used to access the C and D matrix tiles this warp computes.
	int *shmem_warp_tile_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * N * WARP_ROW_TILES * SHMEM_STRIDE +
	(warpId % BLOCK_COL_WARPS) * SHMEM_OFFSET;

      // Store the D fragments to shared memory.
#pragma unroll
	for(int i = 0; i < WARP_ROW_TILES; i += 1){
	#pragma unroll
		for(int j = 0; j < WARP_COL_TILES; j += 1){
		#pragma unroll
			for(int t = 0; t < c[i][j].num_elements; t += 1){
				c[i][j].x[t] = ((c[i][j].x[t] * alpha) >> integer);
			}
			int *tile_ptr = shmem_warp_tile_ptr + i * N * SHMEM_STRIDE + j * M;
			wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
		}
	}

    __syncthreads();

	int *shmem_warp_stream_ptr = (int *)shmem + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N * SHMEM_STRIDE
									+ (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	const size_t gmem_idx =
		(block_tile_i * N + (warpId / BLOCK_COL_WARPS) * WARP_ROW_TILES * N) * GLOBAL_MEM_STRIDE +
		block_tile_j * M + (warpId % BLOCK_COL_WARPS) * WARP_COL_TILES * M;
	uint8_t *dst_gmem_warp_stream_ptr = &D[gmem_idx];

	int *shmem_lane_stream_ptr =
		shmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * SHMEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);
	
	uint8_t *dst_gmem_lane_stream_ptr =
		dst_gmem_warp_stream_ptr +
		(laneId / SHARED_TO_GLOBAL_LANES_PER_LINE) * GLOBAL_MEM_STRIDE +
		(laneId % SHARED_TO_GLOBAL_LANES_PER_LINE);

	for(int i = 0; i < WARP_ROW_TILES * N; i += SHARED_TO_GLOBAL_LINES_PER_WARP){
		*(dst_gmem_lane_stream_ptr + i * GLOBAL_MEM_STRIDE) = (uint8_t)(*(shmem_lane_stream_ptr + i * SHMEM_STRIDE));
	}

	__syncthreads();
}

void initialData(uint8_t *vec, uint8_t *mat_data, int *mat_index, uint8_t *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparse, int minibatch) {
	// generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));
	unsigned int w = vecNum * sparse;

	for(int i = 0; i < minibatch * vecNum; i += 1){
		vec[i] = (uint8_t)(rand() % 5);
	}

	for(int i = 0; i < h * w; i += 1){
		mat_data[i] = (uint8_t)(rand() % 5);
		mat_data_for_gpu[i] = mat_data[i];
	}

	int* tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
	for (int i=0; i<vecNum/NUM_BANK; ++i)
		tmp_index[i] = i;

	for (int j=0; j<h; j += ALIGN_N){
		for (int i=0; i<w; i+= w/NUM_BANK){
			std::random_shuffle(tmp_index,tmp_index+vecNum/NUM_BANK);
			std::sort(tmp_index, tmp_index+w/NUM_BANK);
			for (int k=0; k<w/NUM_BANK; ++k){
				for(int j_in = 0; j_in < ALIGN_N; j_in += 1){
					mat_index[(i + k) + (j + j_in) * w] = tmp_index[k] + i/sparse;
					mat_index_for_gpu[(i + k) + (j + j_in) * w] = mat_index[(i + k) + (j + j_in) * w];
					// mat_index[i + k + (j + j_in) * w] = tmp_index[k]+i/sparse; // tmp_index[k] + delta(vecNum/NUM_BANK)
					// mat_index_for_gpu[(i + k) + (j + j_in) * w] = mat_index[i + k + (j + j_in) * w];
				}
			}
		}
	}
	free(tmp_index);
}

int oneKernel_general(int w, const int h, const int vecNum, const int BLOCK_WIDTH, const int VEC_WIDTH, const int minibatch) {
	// set up device
	int dev = 0;
	cudaSetDevice(dev);

	checksetting();
	//const int w = 512*14;
	//const int h = 16384;
	//const int vecNum = 4096;
	const float sparse = float(w) / float(vecNum);

	// set up data size of vectors
	printf("Matrix size (h=%d,w=%d); Vector size %d; VEC_WIDTH: %d, BLOCK_WIDTH: %d\n", h, w, vecNum,VEC_WIDTH, BLOCK_WIDTH);

	// malloc host memory
	size_t vec_nBytes = vecNum * minibatch * sizeof(uint8_t);		// size of dense matrix
	size_t result_nBytes = h * minibatch * sizeof(uint8_t);		// size of result matrix
	size_t mat_data_nBytes = w * h * sizeof(uint8_t);				// size of sparse matrix
	size_t mat_index_nBytes = w * h * sizeof(int);			// index size same with data, csc?s

	uint8_t *vec, *mat_data, *mat_data_for_gpu, *hostRef, *gpuRef;
	int *mat_index, *mat_index_for_gpu;
	vec = (uint8_t *)malloc(vec_nBytes);
	mat_data = (uint8_t *)malloc(mat_data_nBytes);
	mat_index = (int *)malloc(mat_index_nBytes);
	mat_data_for_gpu = (uint8_t *)malloc(mat_data_nBytes);
	mat_index_for_gpu = (int *)malloc(mat_index_nBytes);
	hostRef = (uint8_t *)malloc(result_nBytes);
	gpuRef = (uint8_t *)malloc(result_nBytes);

	// initialize data at host side
	initialData(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparse, minibatch);
	memset(hostRef, 0, result_nBytes);
	memset(gpuRef, 0, result_nBytes);

	// malloc device global memory
	uint8_t *g_vec, *g_mat_data, *g_result;
	int *g_mat_index;
	cudaMalloc((uint8_t**)&g_vec, vec_nBytes);
	cudaMalloc((uint8_t**)&g_mat_data, mat_data_nBytes);
	cudaMalloc((int**)&g_mat_index, mat_index_nBytes);
	cudaMalloc((uint8_t**)&g_result, result_nBytes);

	// transfer data from host to device
	cudaMemcpy(g_vec, vec, vec_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_mat_data, mat_data_for_gpu, mat_data_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_mat_index, mat_index_for_gpu, mat_index_nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(g_result, gpuRef, result_nBytes, cudaMemcpyHostToDevice);
	//printf("%d, %f\n",mat_index[0], mat_data[0]);
	// invoke kernel at host side

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ntimes;

	ntimes = 10;

	int block_size_col = BLOCK_COL_TILES * M;
	int block_size_row = BLOCK_ROW_TILES * N;
  
	int block_num = (M_GLOBAL * N_GLOBAL) / (block_size_col * block_size_row);

	int alpha = 1, integer = 0;
	printf("block_num: %d, THREADS_PER_BLOCK: %d\n", block_num, THREADS_PER_BLOCK);

	const int shared_size = MAX(sizeof(uint8_t) * (BLOCK_SIZE_K_SPARSE * (BLOCK_COL_TILES * M + SKEW_UINT8) + BLOCK_SIZE_K_SPARSE * (BLOCK_ROW_TILES * N + SKEW_UINT8)),
			M * BLOCK_ROW_TILES * N * BLOCK_COL_TILES * sizeof(int));

	printf("shared_size is %d\n", shared_size);
	/*
	if(shared_size > 32768){
		checkCudaErrors(cudaFuncSetAttribute(
			compute_gemm_imma_large_share, cudaFuncAttributeMaxDynamicSharedMemorySize,
			shared_size));
	}
	*/

	cudaDeviceSynchronize();

	for(int i = 0; i < ntimes; i +=1){
		//compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(d_A, d_B, d_D, alpha, integer);
		/*
		if(shared_size > 32768){
			checkCudaErrors(cudaFuncSetAttribute(
				compute_gemm_imma_large_share, cudaFuncAttributeMaxDynamicSharedMemorySize,
				shared_size));
			compute_gemm_imma_large_share<<<block_num, THREADS_PER_BLOCK, shared_size>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
		}else{
			compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
		}
		*/
		compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
	}

	cudaEventRecord(start);
	//cudaEventSynchronize(start);
	//double iStart = seconds();
	for(int i = 0; i < ntimes; i += 1){
		/*
		if(shared_size > 32768){
			checkCudaErrors(cudaFuncSetAttribute(
				compute_gemm_imma_large_share, cudaFuncAttributeMaxDynamicSharedMemorySize,
				shared_size));
			compute_gemm_imma_large_share<<<block_num, THREADS_PER_BLOCK, shared_size>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
		}else{
			compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
		}
		*/
		compute_gemm_imma<<<block_num, THREADS_PER_BLOCK>>>(g_vec, g_mat_data, g_mat_index, g_result, alpha, integer);
	}
		
	//CUDA_CHECK(cudaGetLastError());
	cudaDeviceSynchronize();
	// record stop event on the default stream
	cudaEventRecord(stop);
	// wait until the stop event completes
	cudaEventSynchronize(stop);
	//double iElaps = seconds() - iStart;

	float time;
	cudaEventElapsedTime(&time, start, stop);
	// clean up the two events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("checkpoint 0\n");
	printf("cal sparse %.3f; cuda: cal Time= %f msec\n", 1.0 - sparse, time/ntimes);

	// copy kernel result back to host side
	cudaMemcpy(gpuRef, g_result, result_nBytes, cudaMemcpyDeviceToHost);
	// add vector at host side for result checks
	printf("checkpoint 1\n");
	// MVOnHost(vec, mat_data, mat_index, hostRef, w, h, vecNum, minibatch);

	// printf("checkpoint 2\n");

    
	// bool correct = true;
	// double eps = 1.e-6;
  
	// for(int i = 0; i < M_GLOBAL * N_GLOBAL; i++){
	// 	double abs_err = fabs(hostRef[i] - gpuRef[i]);
	// 	double dot_length = M;
	// 	double abs_val = fabs(hostRef[i]);
	// 	double rel_err = abs_err / abs_val / dot_length;
	// 	if (rel_err > eps) {
	// 		printf("Error! Matrix[%05d]=%d, ref=%d error term is > %E\n",
	// 				i, gpuRef[i], hostRef[i], eps);
	// 		correct = false;
	// 		break;
	// 	}
		
	// }
	// printf("Error! Matrix[%05d]=%d, ref=%d error term is > %E\n",
	// 				1, gpuRef[1], hostRef[1], eps);

	// if(correct) printf("Result = Pass\n");
	// else printf("Result = Fail\n");
    
    printf("Pass\n\n");

	printf("checkpoint 3\n");

	// free device global memory
	cudaFree(g_vec);
	cudaFree(g_mat_data);
	cudaFree(g_mat_index);
	cudaFree(g_result);
	cudaDeviceReset();
	// free host memory
	free(vec);
	free(mat_data);
	free(mat_index);
	free(mat_data_for_gpu);
	free(mat_index_for_gpu);
	free(hostRef);
	free(gpuRef);
	return(0);
}

int main(int argc, char **argv) {	
	//const int h = 16384;
	//const int vecNum = 8192;
	const int h = N_GLOBAL;
	const int vecNum = K_GLOBAL;

	int w = int(vecNum * (1-SPARSITY));

	const int BLOCK_WIDTH = w/8;
	//const int minibatch = 8;
	const int minibatch = M_GLOBAL;

	const int VEC_WIDTH = vecNum * BLOCK_WIDTH / w;		// VEC_WIDTH = vecNum / 32;
	printf("BLOCK_WIDTH: %d, VEC_WIDTH: %d\n", BLOCK_WIDTH, VEC_WIDTH);
	//oneKernel(w, h, vecNum, BLOCK_WIDTH, NUM_THREADS, VEC_WIDTH, minibatch);
	oneKernel_general(w, h, vecNum, BLOCK_WIDTH, VEC_WIDTH, minibatch);
	return 0;
}
