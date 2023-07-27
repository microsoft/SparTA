#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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

__global__ void mma_test(half *__restrict__ a, half *__restrict__ b, half *__restrict__ c,
                         const int M, const int N, const int K)
{
    // for (int i = 0; i < M * K; i++)
    // {
    //     printf("a[%d][%d] %d \n", i / N, i % N, a[i]);
    //     printf("b[%d][%d] %d \n", i / N, i % N, b[i]);
    // }
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0);
    nvcuda::wmma::load_matrix_sync(frag_a, &a[0], 16);
    nvcuda::wmma::load_matrix_sync(frag_b, &b[0], 16);
    __syncthreads();
    // if(threadIdx.x == 4)
    // for (int i = 0; i < frag_a.num_elements; i++)
    // {
    //     printf("frag_a[%d] %f \n", i, (float)frag_a.x[i]);
    // }
    // printf("tid %d: frag_a[0] %f \n", (int)threadIdx.x, (float)frag_a.x[0]);
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    nvcuda::wmma::store_matrix_sync(c, frag_c, 16, nvcuda::wmma::mem_row_major);
}

int main()
{

    const int M = 16;
    const int N = 16;
    const int K = 16;
    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);
    half *h_a, *h_b, *d_a, *d_b;
    half *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    // h_c = (int32_t *)malloc(size_c);
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(i % 32);
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(i % 32);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    mma_test<<<dim3(1, 1, 1), dim3(32, 1, 1)>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++)
    {
        printf("h[%d][%d] %f \n", i / N, i % N, (float)h_d_c[i]);
    }

    return 0;
}