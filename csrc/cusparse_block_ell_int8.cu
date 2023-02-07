/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

const int EXIT_UNSUPPORTED = 2;

int main(int argc, char *argv[]) {
    float sparsity_ratio = 1-atof(argv[1]);
    const int m = atoi(argv[2]);
    const int k = atoi(argv[3]);
    const int n = atoi(argv[4]);
    // Host problem definition
    int   A_num_rows      = m;
    int   A_num_cols      = k;
    int   A_ell_blocksize = 64;
    int   A_ell_cols      = int ( k * sparsity_ratio );
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                           (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = n;
    int   B_num_cols      = A_num_cols;
    int   ldb             = B_num_cols;
    int   ldc             = B_num_rows;
    int   A_size          = m * k;
    int   B_size          = k * n;
    int   C_size          = m * n;
    
    float * hA = (float*)malloc(A_size * sizeof(float));
    float * hA_values = (float*) malloc(A_num_blocks * A_ell_blocksize * A_ell_blocksize* sizeof(float));
    float * hB = (float*)malloc(B_size * sizeof(float));
    float * hC = (float*)malloc(C_size * sizeof(float));
    int * hA_columns = (int*)malloc(sizeof(int)*A_num_blocks);
    for(int i = 0; i<B_size; i++)
        hB[i] = 1.0;
    memset(hA, 0, sizeof(float)*A_size);
    // randomize the block position
    std::vector<int> col_idx;
    for(int cid = 0; cid < A_ell_cols/A_ell_blocksize; cid++)
        col_idx.push_back(cid);
    size_t col_size = A_ell_cols / A_ell_blocksize;
    for(int i=0; i<A_num_rows/A_ell_blocksize;i++){
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(col_idx.begin(), col_idx.end(), g);
        std::vector<int> tmp(col_idx.begin(), col_idx.begin()+A_ell_cols/A_ell_blocksize);
        std::sort(tmp.begin(), tmp.end());
        memcpy((void*)&hA_columns[i * col_size], tmp.data(), sizeof(int)*col_size);
        
    }
    // for(int i=0; i<A_num_blocks;i++){
    //     printf("%d\n", hA_columns[i]);
    // }
    // printf("A_num_blocks:%d\n", A_num_blocks);
    float alpha           = 1.0f;
    float beta            = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    int    * dA_columns;
    float * dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                                    A_ell_cols * A_num_rows * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                           A_num_blocks * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_ell_cols * A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                      &matA,
                                      A_num_rows, A_num_cols, A_ell_blocksize,
                                      A_ell_cols, (void*)dA_columns, (void*)dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, (void*)dB,
                                        CUDA_R_8I, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, n, ldc, (void*)dC,
                                        CUDA_R_32I, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32I,
                                 CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    float ms_total;
    int n_iter = 1000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<n_iter;i++){
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32I,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_total, start, stop);
    printf("Time= %f ms\n",ms_total/n_iter);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
        //--------------------------------------------------------------------------
    // device result check
    // CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         float c_value  = static_cast<float>(hC[i + j * ldc]);
    //         float c_result = static_cast<float>(hC_result[i + j * ldc]);
    //         if (c_value != c_result) {
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     std::printf("spmm_blockedell_example test PASSED\n");
    // else
    //     std::printf("spmm_blockedell_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}