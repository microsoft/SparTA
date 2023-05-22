#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void _test_kernel()
{
    printf("tid %d, hello world\n", threadIdx.x);
}


int main(int argc, char *argv[])
{
    _test_kernel<<<dim3(1, 1, 1), dim3(32, 1, 1)>>>();
    cudaDeviceSynchronize();
}
