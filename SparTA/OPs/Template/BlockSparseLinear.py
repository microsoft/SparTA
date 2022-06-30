# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

block_sparse_linear_header_template = '''
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

const int BLOCK_SIZE_M=BLOCK_SIZE_M_VALUE;
const int BLOCK_SIZE_K=BLOCK_SIZE_K_VALUE;
const int BLOCK_SIZE_N=BLOCK_SIZE_N_VALUE;
const int THREAD_SIZE_M=THREAD_SIZE_M_VALUE;
const int THREAD_SIZE_K=THREAD_SIZE_K_VALUE;
const int THREAD_SIZE_N=THREAD_SIZE_N_VALUE;
const int M=GLOBAL_M_VALUE;
const int N=GLOBAL_N_VALUE;
const int K=GLOBAL_K_VALUE;
'''

block_sparse_linear_function_template = '''
__global__ void BLOCK_SPARSE_MATMUL(float* input0, float* input1, int* input2, int* input3, float* input4, float *output0){

    float * A = reinterpret_cast<float*>(input0);
    float * W_val = reinterpret_cast<float*>(input1);
    int * W_row = reinterpret_cast<int*>(input2);
    int * W_col = reinterpret_cast<int*>(input3);
    float * bias = reinterpret_cast<float*>(input4);
    float * C = reinterpret_cast<float*>(output0);
    /* 
    COMMENT_TAG
    */
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_N * BLOCK_SIZE_K];

    float accum[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    float a_frag[THREAD_SIZE_M][THREAD_SIZE_K];
    float b_frag[THREAD_SIZE_N][THREAD_SIZE_K];

    int A_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int bszy = BLOCK_SIZE_M / THREAD_SIZE_M;
    int bszx = BLOCK_SIZE_N / THREAD_SIZE_N;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = W_row[bx], index_end = W_row[bx+1];

    const int vBLOCK_SIZE_M = BLOCK_SIZE_M / THREAD_SIZE_M;
    const int vBLOCK_SIZE_N = BLOCK_SIZE_N / THREAD_SIZE_N;
    for(int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1){
        int tile_idx = W_col[tile_block_idx] * BLOCK_SIZE_K;
        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_M; k += A_TILE_ROW_STRIDE){
            *((float4 *)(&As[(k+A_BLOCK_ROW_START) * BLOCK_SIZE_K + A_BLOCK_COL_START])) =
                *((float4 *)(&A[(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START) * K + tile_idx+A_BLOCK_COL_START]));
            //FETCH_FLOAT4(As[OFFSET(k+A_BLOCK_ROW_START, A_BLOCK_COL_START, BLOCK_SIZE_K)]) =
                //FETCH_FLOAT4(A[OFFSET(by*BLOCK_SIZE_M+k+A_BLOCK_ROW_START, tile_idx+A_BLOCK_COL_START, K)]);
        }

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += B_TILE_ROW_STRIDE){
            *((float4 *)(&Bs[(k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START])) =
                *((float4 *)(&W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]));
            //FETCH_FLOAT4(Bs[OFFSET(k+B_BLOCK_ROW_START, B_BLOCK_COL_START, BLOCK_SIZE_N)]) = 
                //FETCH_FLOAT4(W_val[tile_block_idx * BLOCK_SIZE_N * BLOCK_SIZE_K + (k+B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]);
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < BLOCK_SIZE_K; k += THREAD_SIZE_K){
            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j += 1){
                    a_frag[j][i] = As[(ty + vBLOCK_SIZE_M * j) * BLOCK_SIZE_K + k + i];
                    //a_frag[j][i] = As[OFFSET(ty + vBLOCK_SIZE_M * j, k+i, BLOCK_SIZE_K)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_K; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_N; j += 1){
                    b_frag[j][i] = Bs[(k+i) * BLOCK_SIZE_N + tx + vBLOCK_SIZE_N * j];
                    //b_frag[j][i] = Bs[OFFSET(k+i, tx + vBLOCK_SIZE_N * j, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for(int i = 0; i < THREAD_SIZE_N; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_SIZE_M; j++){
                    #pragma unroll
                    for(int k_in = 0; k_in < THREAD_SIZE_K; k_in++){
                        // accum[i][j] = fma(a_frag[j][k_in], b_frag[i][k_in], accum[i][j]);
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }

    float bias_local[THREAD_SIZE_N];
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        bias_local[thread_x] = bias[BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N];
    }

    #pragma unroll
    for(int thread_x = 0; thread_x < THREAD_SIZE_N; thread_x++){
        #pragma unroll
        for(int thread_y = 0; thread_y < THREAD_SIZE_M; thread_y+=1){
            C[(BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M) * N + BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N] = (accum[thread_x][thread_y]) + bias_local[thread_x];
            /*
            C[OFFSET(
                BLOCK_SIZE_M * by + ty + thread_y * vBLOCK_SIZE_M,
                BLOCK_SIZE_N * bx + tx + thread_x * vBLOCK_SIZE_N,
                N
            )] = (accum[thread_x][thread_y]) + bias_local[thread_x];
            */
        }
    }
}
'''

block_sparse_linear_test_template = '''
#define checkCudaErrors(func)                                                       \\
{                                                                                   \\
    cudaError_t e = (func);                                                         \\
    if(e != cudaSuccess)                                                            \\
        printf ("%s %d CUDA: %s\\n", __FILE__,  __LINE__, cudaGetErrorString(e));   \\
}

int load_int_array_from_file(int* arr, string filepath) {
    ifstream inputFile(filepath);
    int length = 0;
    int current_number = 0;
    while (inputFile >> current_number) {
        arr[length++] = current_number;
    }
    return length;
}

int load_float_array_from_file(float* arr, string filepath) {
    ifstream inputFile(filepath);
    int length = 0;
    float current_number = 0;
    while (inputFile >> current_number) {
        arr[length++] = current_number;
    }
    return length;
}

void HostComputation_sparse(float* A, int* row, int* col, float* val, float* D, float* bias, int M, int K, int N, int BLOCK_SIZE_K, int BLOCK_SIZE_N){
    size_t mem_size_B = sizeof(float) * K * N;
    float* B = (float*)malloc(mem_size_B);
    memset(B, 0, sizeof(B));
    int ROW_BLOCK_NUM = N / BLOCK_SIZE_N;
    for(int i = 0; i < ROW_BLOCK_NUM; i ++){
        int index_start = row[i], index_end = row[i+1];
        for(int index = index_start; index < index_end; index += 1){
            int col_index = col[index] * BLOCK_SIZE_K;
            int row_index = i * BLOCK_SIZE_N;
            float* val_ptr = val + index * BLOCK_SIZE_K * BLOCK_SIZE_N;
            for(int k = 0; k < BLOCK_SIZE_K; k += 1){
                for(int n = 0; n < BLOCK_SIZE_N; n += 1){
                    B[(k + col_index) * N + row_index+n] = *(val_ptr + k * BLOCK_SIZE_N + n);
                }
            }
        }
    }


    for(int i = 0; i < M; i += 1){
        for(int j = 0; j < N; j += 1){
            float cSub = 0;
            for(int k = 0; k < K; k += 1){
                cSub += A[i * K + k] * B[k * N + j];
            }
            D[i * N + j] = cSub + bias[j];
        }
    }
}

int main()
{
    int size_A = M * K;
    int size_C = M * N;

    int mem_size_A = sizeof(float) * size_A;
    int mem_size_C = sizeof(float) * size_C;
    int mem_size_bias = sizeof(float) * N;

    float* h_A = (float*)malloc(mem_size_A);
    float* h_C = (float*)malloc(mem_size_C);
    float* h_bias = (float*)malloc(mem_size_bias);
    float* h_result = (float*)malloc(mem_size_C);

    int mem_size_row = sizeof(int) * (K / BLOCK_SIZE_K + 1);
    int* h_row = (int*)malloc(mem_size_row);
    load_int_array_from_file(h_row, INPUT_FILE_PATH_W_ROW);

    int mem_size_col = sizeof(int) * (K / BLOCK_SIZE_K) * (N / BLOCK_SIZE_N);
    int* h_col = (int*)malloc(mem_size_col);
    int block_num = load_int_array_from_file(h_col, INPUT_FILE_PATH_W_COL);

    int mem_size_val = sizeof(float) * BLOCK_SIZE_K * BLOCK_SIZE_N * block_num;
    float* h_val = (float*)malloc(mem_size_val);
    load_float_array_from_file(h_val, INPUT_FILE_PATH_W_VAL);


    float* d_A;
    float* d_C;
    float* d_bias;

    // device memory allocation
    int* d_row;
    int* d_col;
    float* d_val;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 10;

    load_float_array_from_file(h_A, INPUT_FILE_PATH_A);
    load_float_array_from_file(h_bias, INPUT_FILE_PATH_BIAS);

    checkCudaErrors(cudaMalloc(&d_A, mem_size_A));
    checkCudaErrors(cudaMalloc(&d_C, mem_size_C));
    checkCudaErrors(cudaMalloc(&d_bias, mem_size_bias));

    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, h_bias, mem_size_bias, cudaMemcpyHostToDevice));

    // device csr memory copy
    checkCudaErrors(cudaMalloc(&d_row, mem_size_row));
    checkCudaErrors(cudaMalloc(&d_col, mem_size_col));
    checkCudaErrors(cudaMalloc(&d_val, mem_size_val));

    checkCudaErrors(cudaMemcpy(d_row, h_row, mem_size_row, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_col, h_col, mem_size_col, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, h_val, mem_size_val, cudaMemcpyHostToDevice));

    dim3 dimBlock(float(BLOCK_SIZE_N / THREAD_SIZE_N), BLOCK_SIZE_M / THREAD_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);

    // warm-up
    for(int run = 0; run < nIter; run++){
        BLOCK_SPARSE_MATMUL<<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_bias, d_C);
    }

    checkCudaErrors(cudaEventRecord(start));
    for(int run = 0; run < nIter; run++) {
        BLOCK_SPARSE_MATMUL<<<dimGrid, dimBlock>>>(d_A, d_val, d_row, d_col, d_bias, d_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_result, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    float msecPerMatrixMul = msecTotal / nIter;

    printf("%f", msecPerMatrixMul);

    HostComputation_sparse(h_A, h_row, h_col, h_val, h_C, h_bias, M, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N);
    // bool correct = true;
    double eps = 1.e-4;

    for(int i = 0; i < M * N; i++){
        double abs_err = abs(h_C[i] - h_result[i]);
        double dot_length = M;
        double abs_val = abs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (abs_err > eps) {
            printf("abs_val: %lf, rel_err: %lf, abs_val: %lf, dot_length: %lf \\n", abs_val, rel_err, abs_val, dot_length);
            printf("Error! Matrix[%05d]=%lf, ref=%lf error term is %lf > %E\\n",
                    i, h_result[i], h_C[i], rel_err, eps);
            // correct = false;
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);

    free(h_A);
    free(h_C);
    free(h_row);
    free(h_col);
    free(h_val);
}
'''
