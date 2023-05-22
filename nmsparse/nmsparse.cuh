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

namespace nmsparse
{

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
            default:
                ::std::cout << "Unsupported sparse pattern" << ::std::endl;
                break;
        }
        return cudaSuccess;
    }
}

#endif