#ifndef __NMSPARSE_KERNEL_H_
#define __NMSPARSE_KERNEL_H_

#include <cuda_runtime.h>
#include <cctype>
#include <string>
#include <iostream>
#include <assert.h>
#include "context.cuh"
#include "utils.cuh"
#include "nmsparse_ew.cuh"
#include "nmsparse_vw4.cuh"
#include "nmsparse_vw32.cuh"
#include "nmsparse_vw64.cuh"
#include "nmsparse_bw4x4.cuh"

namespace nmsparse
{
    cudaError_t nmsparseKernelInit(){

        cudaFuncSetAttribute(nmsparse_vw4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_vw4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        cudaFuncSetAttribute(nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x128_4x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_vw32_gemm_simt_fp32_fp32_fp32_32x32x256_4x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        cudaFuncSetAttribute(nmsparse_vw64_gemm_tensorop_fp32_fp32_fp32_32x32x32_4x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_vw64_gemm_tensorop_fp32_fp32_fp32_32x32x32_4x4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        cudaFuncSetAttribute(nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4_K3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x128_8x4_K4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        cudaFuncSetAttribute(nmsparse_bw4x4_gemm_simt_fp32_fp32_fp32_32x128x64_8x4_K4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

        return cudaGetLastError();
    }

    template <typename dtype>
    cudaError_t nmsparseSpMM(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0)
    {
        
        ::std::cout << "nmsparseSpMM<float>" << ::std::endl;
        switch (ctx.nmsparsePattern)
        {
            case SparsePattern::ElementWise:
                std::cout << "ElementWise" << std::endl;
                nmsparseSpMMEW<dtype>(ctx, m, k, n, mat_a_dense, mat_b_sparse_idx, mat_b_sparse_val, output, stream);
                break;
            case SparsePattern::VectorWise4:
                std::cout << "VectorWise4" << std::endl;
                nmsparseSpMMVW4<dtype>(ctx, m, k, n, mat_a_dense, mat_b_sparse_idx, mat_b_sparse_val, output, stream);
                break;
            case SparsePattern::VectorWise32:
                std::cout << "VectorWise32" << std::endl;
                nmsparseSpMMVW32<dtype>(ctx, m, k, n, mat_a_dense, mat_b_sparse_idx, mat_b_sparse_val, output, stream);
                break;
            case SparsePattern::VectorWise64:
                std::cout << "VectorWise64" << std::endl;
                nmsparseSpMMVW64<dtype>(ctx, m, k, n, mat_a_dense, mat_b_sparse_idx, mat_b_sparse_val, output, stream);
                break;
            case SparsePattern::BlockWise4x4:
                std::cout << "BlockWise4x4" << std::endl;
                nmsparseSpMMBW4x4<dtype>(ctx, m, k, n, mat_a_dense, mat_b_sparse_idx, mat_b_sparse_val, output, stream);
                break;
            default:
                throw std::runtime_error("Unsupported sparse pattern");
                break;
        }
        return cudaSuccess;
    }
}

#endif