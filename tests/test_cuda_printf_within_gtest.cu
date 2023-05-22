#include <gtest/gtest.h>
#include <iostream>
#include <nmsparse/nmsparse.cuh>
#include <cuda_runtime.h>


struct NMSparseTest : testing::Test
{
    NMSparseTest()
    {
    }
    ~NMSparseTest()
    {
    }
};

extern "C" __global__ void _test_kernel()
{
    printf("tid %d, hello world\n", threadIdx.x);
}

TEST(NMSparseTest, CudaOutputStream)
{
    _test_kernel<<<dim3(1, 1, 1), dim3(32, 1, 1)>>>();
    cudaDeviceSynchronize();
    EXPECT_EQ(0, 0);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
