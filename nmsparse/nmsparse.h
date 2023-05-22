#ifndef __NMSPARSE_KERNEL_H_
#define __NMSPARSE_KERNEL_H_

#include <cuda_runtime.h>
#include <cctype>
#include <string>
#include <iostream>

namespace nmsparse
{
    // enumerate the type
    enum SparsePattern
    {
        ElementWise = 0,
        VectorWise4,
        VectorWise32,
        VectorWise64,
        BlockWise4x4,
        BlockWise64x64
    };

    // the context of the nmsparse
    struct nmsparseContext_t
    {

        // type of sparse pattern
        SparsePattern nmsparsePattern = ElementWise;

        // size of the sparse pattern N : M
        __uint32_t nmsparseN = 16;
        __uint32_t nmsparseM = 32;
        // sparsity of the sparse pattern N/M
        float sparsity = 0.5;
    };

    bool nmsparseCreateContext(nmsparseContext_t *ctx);

    bool nmsparseDestroyContext(nmsparseContext_t *ctx);

    bool nmsparseSetContext(nmsparseContext_t *ctx, SparsePattern nmsparsePattern, unsigned int nmsparseN, unsigned int nmsparseM);

    template <typename dtype>
    bool checkCtxPattern(nmsparseContext_t const *ctx);
   
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val);

    template <typename dtype>
    cudaError_t nmsparseSpMM(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, 
                int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);

    template <typename dtype>
    cudaError_t nmsparseSpMMEW(nmsparseContext_t ctx, int m, int k, int n, dtype *mat_a_dense, int *mat_b_sparse_idx, dtype *mat_b_sparse_val, dtype *output, cudaStream_t stream = 0);
}

#endif