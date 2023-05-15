#ifndef __NMSPARSE_KERNEL_H_
#define __NMSPARSE_KERNEL_H_
#include <cuda_runtime.h>

namespace nmsparse{
    // nmsparse-ew
    cudaError_t CudaSpmmEW(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, int M, int K, int N);
    // nmsparse-vw4
    cudaError_t CudaSpmmVW4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data,  int M, int K, int N);
    // nmsparse-vw32
    cudaError_t CudaSpmmVW32(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, int M, int K, int N);
    // nmsparse-bw4x4
    cudaError_t CudaSpmmBW4x4(float *g_vec, float *g_mat_data, int *g_mat_index, float *g_data, int M, int K, int N);
}

#endif