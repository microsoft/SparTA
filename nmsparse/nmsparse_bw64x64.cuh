#ifndef _NMSPARSE_KERNEL_BLOCKWISE4x4_H_
#define _NMSPARSE_KERNEL_BLOCKWISE4x4_H_

#include "context.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include "cutlass/cutlass.h"
#include "device/gemm.h"
// #include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

namespace nmsparse
{
	namespace block_wise_64x64{
		const int BLOCK_SIZE_N = 64;
		const int BLOCK_SIZE_K = 64;
		using ElementComputeEpilogue = float; // <- data type of epilogue operations
		using ElementAccumulator = int32_t;
		using ElementInputA = int8_t;
		using ElementInputB = int8_t;
		using ElementOutput = int32_t;

		// The code section below describes matrix layout of input and output matrices. Column Major for
		// Matrix A, Row Major for Matrix B and Row Major for Matrix C
		using LayoutInputA = cutlass::layout::RowMajor;
		using LayoutInputB = cutlass::layout::ColumnMajor;
		using LayoutOutput = cutlass::layout::RowMajor;
		// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
		using MMAOp = cutlass::arch::OpClassTensorOp;

		// This code section describes CUDA SM architecture number
		using SmArch = cutlass::arch::Sm80;
		This code section describes the tile size a thread block will compute using ShapeMMAThreadBlock =
			cutlass::gemm::GemmShape<128, BLOCK_SIZE_N, BLOCK_SIZE_K>; // <- threadblock tile M = 128, N = 256, K = 64
		// This code section describes tile size a warp will compute
		using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, BLOCK_SIZE_K>; // <- warp tile M = 64, N = 64, K = 64
		// This code section describes the size of MMA op
		using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 16

		// This code section describes how threadblocks are scheduled on GPU
		using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

		// This code section describes the epilogue part of the kernel
		using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
			ElementOutput,									  // <- data type of output matrix
			128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
															  // memory access. For a byte, it's 16
															  // elements. This becomes the vector width of
															  // math instructions in the epilogue too
			ElementAccumulator,								  // <- data type of accumulator
			ElementComputeEpilogue>;						  // <- data type for alpha/beta in linear combination function

		// Number of pipelines you want to use
		constexpr int NumStages = 2;

		using Gemm = cutlass::gemm::device::Gemm_Sparse<ElementInputA,
														LayoutInputA,
														ElementInputB,
														LayoutInputB,
														ElementOutput,
														LayoutOutput,
														ElementAccumulator,
														MMAOp,
														SmArch,
														ShapeMMAThreadBlock,
														ShapeMMAWarp,
														ShapeMMAOp,
														EpilogueOp,
														SwizzleThreadBlock,
														NumStages>;
	}
	
	template <typename dtype>
	cudaError_t nmsparseSpMMBW64x64(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);

	template <typename dtype>
	cudaError_t nmsparseSpMMBW64x64(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream)
	{
		assert(is_divisible(m, 32) && is_divisible(n, 128) && is_divisible(k, 64));
		const float sparsity = ctx.sparsity;
		const int M = m;
		const int N = n;
		const int K = k;
		const int w = int((1.0f - sparsity) * K);

		Gemm gemm_op;

		return cudaGetLastError();
	}
}

#endif