#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1000
#define M 2000
#define K 1500

void matrix_multiply(float *A, float *B, float *C) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    float alpha = 1.0f;
    float beta = 0.0f;

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaStat = cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_B, M * K * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_C, N * K * sizeof(float));

    // Create cuBLAS handle
    stat = cublasCreate(&handle);

    // Set matrix layout
    cublasSetMatrix(N, M, sizeof(float), A, M, d_A, N);
    cublasSetMatrix(M, K, sizeof(float), B, K, d_B, M);
    cublasSetMatrix(N, K, sizeof(float), C, K, d_C, N);

    // Perform matrix multiplication
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, &alpha, d_B, K, d_A, M, &beta, d_C, K);

    // Get the result matrix from device memory
    cublasGetMatrix(N, K, sizeof(float), d_C, N, C, K);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    float *A, *B, *C;
    A = (float*)malloc(N * M * sizeof(float));
    B = (float*)malloc(M * K * sizeof(float));
    C = (float*)malloc(N * K * sizeof(float));

    // Initialize matrices A and B
    // ...

    matrix_multiply(A, B, C);

    // Output matrix C
    // ...

    free(A);
    free(B);
    free(C);

    return 0;
}
