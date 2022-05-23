#include "common.h"
#include "cusparse.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
// Macro definition for the cuda and cusparse
// cuSparse SPMM interface
int cusparse_spmm(
    int M,
    int K,
    int N,
    int * row_idx,
    int * col_idx,
    float * values,
    float * MB,
    float * MC,
    float * alpha,
    float * beta
);
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

int cusparse_spmm(
    int M,
    int K,
    int N,
    int *row_index,
    int *col_index,
    float *values,
    int nnz,
    float *MA,
    float *MC,
    float alpha,
    float beta)
{
    /*
    MA: In activation tensor, Shape: M*K
    NOTE: weight need to be transposed if the weight is stores as NxK
    row_index, col_index, values: Weight in CSR format, Shape: K*N
    MC: Output tensor, Shape M*N
    */
    cusparseHandle_t cusparse_handle;
    printf("M:%d K:%d, N:%d \n", M,K,N);

    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle));
    static size_t bufferSize = 0;
    static float *dBuffer = NULL;
    cusparseSpMatDescr_t sp_weight;
    cusparseDnMatDescr_t in_activation, output_m;
    printf("M:%d K:%d, N:%d \n", M,K,N);

    // printf("%d\n", row_index[K-1]);
    // printf("%d\n", row_index[K]);
    // int nnz = col_index[row_index[K]];
    printf("nnz:%d\n",nnz);
    CUSPARSE_SAFE_CALL(cusparseCreateCsr(&sp_weight,
                                         K,
                                         N,
                                         nnz,
                                         (void *)row_index,
                                         (void *)col_index,
                                         (void *)values,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));
    CUSPARSE_SAFE_CALL(cusparseCreateDnMat(&in_activation, K, M, K, MA,
                                           CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_SAFE_CALL(cusparseCreateDnMat(&output_m, N, M, N, MC,
                                           CUDA_R_32F, CUSPARSE_ORDER_COL));
    if (dBuffer == NULL)
    {
        // allocate the worksparce buffer if this is the first call
        CUSPARSE_SAFE_CALL(cusparseSpMM_bufferSize(
            cusparse_handle,
            CUSPARSE_OPERATION_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, sp_weight, in_activation, &beta, output_m, CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG2, &bufferSize));
        CUDA_SAFE_CALL(cudaMalloc(&dBuffer, bufferSize));
    }
    // Execute the forward matmul
    CUSPARSE_SAFE_CALL(cusparseSpMM(cusparse_handle,
                                    CUSPARSE_OPERATION_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, sp_weight, in_activation, &beta, output_m, CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2, dBuffer));
    // destroy matrix/vector descriptors
    CUSPARSE_SAFE_CALL(cusparseDestroyDnMat(in_activation));
    CUSPARSE_SAFE_CALL(cusparseDestroyDnMat(output_m));
    CUSPARSE_SAFE_CALL(cusparseDestroySpMat(sp_weight));
    CUSPARSE_SAFE_CALL(cusparseDestroy(cusparse_handle));
    return 0;
}

at::Tensor cusparse_linear_forward(
    torch::Tensor input,
    torch::Tensor row_index,
    torch::Tensor col_index,
    torch::Tensor values,
    std::vector<int> weight_shape)
{   
    cudaSetDevice(input.get_device());
    // the weight shape should be KxN
    int n_dim = input.dim();
    auto input_sizes = input.sizes();
    int in_features = input_sizes[n_dim-1];
    int batch_size = std::accumulate(begin(input_sizes), end(input_sizes), 1, std::multiplies<int>());
    batch_size /= in_features;
    std::vector<int64_t> output_shape;
    for(int i=0; i<n_dim-1; i++) output_shape.push_back(input_sizes[i]);
    assert(weight_shape.size()==2);
    int out_features = weight_shape[1];
    output_shape.push_back(out_features);
    c10::ArrayRef<int64_t> _out_size(output_shape.data(), output_shape.data() + output_shape.size());
    torch::Tensor output = torch::empty(_out_size, input.options());
    printf("row index size: %d\n", row_index.size(0));
    printf("m:%d, k:%d, n:%d\n",batch_size, in_features, out_features);
    int nnz = values.size(0);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "cusparse_linear_forward", ([&]
                                                                           { cusparse_spmm(
                                                                                 batch_size,
                                                                                 in_features,
                                                                                 out_features,
                                                                                 row_index.data_ptr<int>(),
                                                                                 col_index.data_ptr<int>(),
                                                                                 values.data_ptr<float>(),
                                                                                 nnz,
                                                                                 input.data_ptr<float>(),
                                                                                 output.data_ptr<float>(),
                                                                                 1,
                                                                                 0); }));
    return output;
}
