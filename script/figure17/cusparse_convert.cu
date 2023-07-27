#include "cusparse.h"
#include "iostream"
#include "sstream"
#include "cuda.h"
#include "time.h"
#include "memory"
#include "cublas_v2.h"
#include "vector"
#include "utils.hpp"
using namespace std;

using namespace std;
// Macro definition for the cuda and cusparse
// cuSparse SPMM interface

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
#define CUSPARSE_SAFE_CALL(func)                                                                \
    do                                                                                          \
    {                                                                                           \
        cusparseStatus_t e = (func);                                                            \
        if (e != CUSPARSE_STATUS_SUCCESS)                                                       \
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

int cusparse_csr_convert(
    float* dense_value,
    int n_row,
    int n_col,
    int * csr_row,
    int * csr_col,
    float * csr_val)
{
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    static void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CUSPARSE_SAFE_CALL(cusparseCreate(&handle));

    CUSPARSE_SAFE_CALL(cusparseCreateDnMat(&matA, n_row, n_col, n_col, dense_value,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_SAFE_CALL( cusparseCreateCsr(&matB, n_row, n_col, 0,
                                    csr_row, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) );
    if (dBuffer == NULL)
        CUDA_SAFE_CALL( cudaMalloc(&dBuffer, bufferSize) );
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CUSPARSE_SAFE_CALL( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                        &nnz) );
    // torch::Tensor csr_col = torch::empty_like({nnz}, csr_row);
    // torch::Tensor csr_values = torch::empty_like({nnz}, dense_values);
    CUSPARSE_SAFE_CALL( cusparseCsrSetPointers(matB, csr_row, csr_col, csr_val) );
    // execute Sparse to Dense conversion
    CUSPARSE_SAFE_CALL( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) );
    CUSPARSE_SAFE_CALL( cusparseDestroyDnMat(matA) );
    CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matB) );
    CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );
    return 0;
}


int main(int argc, char *argv[]){
    float sparsity_ratio = atof(argv[1]);
    printf("Sparsity Ratio=%f\n", sparsity_ratio);
    // Calculate the matA(Activation: Shape=mxk) * matB(Weight:Shape=k*n)
    // Specify the random seed here
    srand(1);
    int32_t * row_idx, *col_idx, *d_row, *d_col;
    int nnz;
    float * values, *d_val;
    float * matA, *matB, *matC, *matC_ref,*d_matA, *d_matB, *d_matC, *dBuffer;
    const int m = atoi(argv[2]);
    const int k = atoi(argv[3]);
    
    //int m=1024, k=1024, n=1024;
    float alpha=1.0, beta=0.0;
    float sparsity = sparsity_ratio;

    matA = (float*) malloc(sizeof(float)*m*k);
    
    init(matA, m*k, sparsity_ratio);

    CUDA_SAFE_CALL(cudaMalloc(&d_matA, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float)*m*k));
    CUDA_SAFE_CALL(cudaMemcpy(d_matA, matA, sizeof(float)*m*k, cudaMemcpyHostToDevice));
   
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 3000;
    


    CUDA_SAFE_CALL(cudaEventRecord(start));

    for(int i = 0; i < nIter; i += 1){
        cusparse_csr_convert(d_matA, m, k, d_row, d_col, d_val);
    }

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    printf("Time= %f msec\n", msecPerMatrixMul);

    return 0;
}