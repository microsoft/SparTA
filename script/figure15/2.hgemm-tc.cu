#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

typedef enum
{
    HGEMMAlignedV1,
    HGEMMAlignedV2,
    HGEMMAlignedV2_5,
    HGEMMAlignedV3,
    HGEMMAlignedV4,
    HGEMMAlignedV5,
    HGEMMAlignedV6
} F16F16GemmTCAlgo_t;

void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K)
{

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float psum = 0.0;
            for (int k = 0; k < K; k++)
            {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

__global__ void myHGEMMAlignedV1(
    half *__restrict__ a, half *__restrict__ b, half *__restrict__ c,
    const int M, const int N, const int K)
{

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid / 2);
    int load_a_smem_k = (tid % 2) * 8;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++)
    {

        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k + 16]) = FLOAT4(a[load_a_gmem_addr + 16]);

        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

template <F16F16GemmTCAlgo_t algo = HGEMMAlignedV1>
void myF16F16GemmTCWarp(half *a, half *b, half *c, int M, int N, int K)
{

    if (algo == HGEMMAlignedV1)
    {
        const int BM = 128, BN = 256;
        dim3 blockDim(256);
        int BX = (N + BN - 1) / BN;
        int BY = (M + BM - 1) / BM;
        dim3 gridDim(BX, BY);
        myHGEMMAlignedV1<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    }
}

float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm)(half *, half *, half *, int, int, int),
    int M, int N, int K)
{

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = (half)(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(rand() / float(RAND_MAX));

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm)(half *, half *, half *, int, int, int),
    int M, int N, int K, int repeat)
{

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
    {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}

int main()
{

    const int test_num = 1;
    const int M_list[test_num] = {4096};
    const int N_list[test_num] = {4096};
    const int K_list[test_num] = {4096};



    const int outer_repeat = 1, inner_repeat = 1;

    {
        printf("\nalgo = HGEMMAlignedV1\n");

        {
            const int M = 1024, N = 1024, K = 1024;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV1>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV1>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV2\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV2>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV2>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV2_5\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV2_5>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV2_5>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV3\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV3>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV3>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV4\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV4>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV4>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV5\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }

    {
        printf("\nalgo = HGEMMAlignedV6\n");

        {
            const int M = 256, N = 256, K = 256;
            float max_error = testF16F16GemmMaxError(
                myF16F16GemmTCWarp<HGEMMAlignedV6>, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        for (int j = 0; j < test_num; j++)
        {
            int M = M_list[j], N = N_list[j], K = K_list[j];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int k = 0; k < outer_repeat; k++)
            {
                double this_sec = testF16F16GemmPerformance(
                    myF16F16GemmTCWarp<HGEMMAlignedV6>, M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, ", M, N, K);
            printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
            printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
        }
    }
}