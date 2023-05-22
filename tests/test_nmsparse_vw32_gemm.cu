#include <gtest/gtest.h>
#include <iostream>
#include <nmsparse/nmsparse.cuh>
#include <cuda_runtime.h>

struct NMSparseTest : testing::Test
{
    NMSparseTest(){
    }
    ~NMSparseTest()
    {
    }
};

TEST_F(NMSparseTest, VectorWise32GemmTestFloat)
{   
    const int M = 256;
    const int N = 1024;
    const int K = 1024;
    const int nmsparseM = 32;
    const float sparsity = 0.75f;
    const int minibatch = M;
    const int h = N;
    const int vecNum = K;
    const int w = vecNum * (1 - sparsity);

    // malloc host memory
    size_t vec_nBytes = vecNum * minibatch * sizeof(float); // size of dense matrix
    size_t result_nBytes = h * minibatch * sizeof(float);   // size of result matrix
    size_t mat_data_nBytes = w * h * sizeof(float);         // size of sparse matrix
    size_t mat_index_nBytes = w * h * sizeof(int);          // index size same with data, csc?s

    float *vec, *mat_data, *mat_data_for_gpu, *hostRef, *gpuRef;
    int *mat_index, *mat_index_for_gpu;
    vec = (float *)malloc(vec_nBytes);
    mat_data = (float *)malloc(mat_data_nBytes);
    mat_index = (int *)malloc(mat_index_nBytes);
    mat_data_for_gpu = (float *)malloc(mat_data_nBytes);
    mat_index_for_gpu = (int *)malloc(mat_index_nBytes);
    hostRef = (float *)malloc(result_nBytes);
    gpuRef = (float *)malloc(result_nBytes);

    // initialize data at host side
    nmsparse::nmsparseContext_t ctx;
    nmsparse::nmsparseCreateContext(&ctx);
    nmsparse::nmsparseSetContext(&ctx, nmsparse::VectorWise32, nmsparseM, sparsity);
    nmsparse::nmSparseInitialRandomData(ctx, vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch);
    nmsparse::nmsparseKernelInit();
    memset(hostRef, 0, result_nBytes);
    memset(gpuRef, 0, result_nBytes);

    // malloc device global memory
    float *g_vec, *g_mat_data, *g_result;
    int *g_mat_index;
    cudaMalloc(&g_vec, vec_nBytes);
    cudaMalloc(&g_mat_data, mat_data_nBytes);
    cudaMalloc(&g_mat_index, mat_index_nBytes);
    cudaMalloc(&g_result, result_nBytes);

    // transfer data from host to device
    cudaMemcpy(g_vec, vec, vec_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat_data, mat_data_for_gpu, mat_data_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat_index, mat_index_for_gpu, mat_index_nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_result, gpuRef, result_nBytes, cudaMemcpyHostToDevice);

    printf("M: %d, N: %d, K: %d, sparsity: %f\n", M, N, K, sparsity);
    printf("w = %d, h = %d, vecNum = %d, minibatch = %d\n", w, h, vecNum, minibatch);
    nmsparse::nmsparseSpMM(ctx, M, K, N, g_vec, g_mat_index, g_mat_data, g_result, 0);
    nmsparse::nmsparseDestroyContext(&ctx);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, g_result, result_nBytes, cudaMemcpyDeviceToHost);
    nmsparse::nmsparseCPURef_ALIGN_SHARED<float>(vec, mat_data, mat_index, hostRef, w, h, vecNum, minibatch);
    bool match;
    match = nmsparse::nmsparseCheckResult<float>(hostRef, gpuRef, h, minibatch);
    printf("Test %s\n", match ? "PASSED" : "FAILED");
    EXPECT_EQ(1, match);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
