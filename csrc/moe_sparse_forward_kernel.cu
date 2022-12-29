
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
using namespace std;
// Macro definition for the cuda and cusparse

#include <assert.h>
// CUDA runtime
#include <cuda.h>
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define FULL_MASK 0xffffffff

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

__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32]; 
    sdata[tid] += sdata[tid + 16]; 
    sdata[tid] += sdata[tid + 8]; 
    sdata[tid] += sdata[tid + 4]; 
    sdata[tid] += sdata[tid + 2]; 
    sdata[tid] += sdata[tid + 1]; 
}

__device__ __forceinline__ const int* add_ptr_u(const int* src, int offset)      \
{                                                                            \
    const int* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

__device__ __forceinline__ const float* add_ptr_f(const float* src, int offset)      \
{                                                                            \
    const float* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}


__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }


__global__ void BATCH_BLOCK_SPARSE_MATMUL(float* tokens, int* sparse_index, int* expert_count, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, const int TMAX)
{
    /*
    description:
    tiling k dimension
    tile size: 32x64x32
    smm_sd_d_nt: sparse matmul, sparse (MxK, along K, K major bcsr) x dense (KxN, along N, need transpose) -> dense (MxN, along N)
    block sparse matrix (block size: 32x64) X dense matrix -> dense matrix

    */
    const int BLOCK_SIZE_M = 32;  // 64
    const int BLOCK_SIZE_K = 64;  //8
    const int BLOCK_SIZE_N = 32;  //128
    const int THREAD_SIZE_K = 64;
    const int M = GLOBAL_M;
    const int K = GLOBAL_K;
    const int N = GLOBAL_N;

    int exp_id = blockIdx.z;
    B += K * N * blockIdx.z;

    assert(blockDim.x % 32 == 0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    // THREAD_SIZE_K: one loop k
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int m_index[BLOCK_SIZE_M];
    char* bShare = (char*)fShare;
    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // N
    uint by = blockIdx.y; // M

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint ori_offsetB00 = tid / (BLOCK_SIZE_N/4) * N + bx * BLOCK_SIZE_N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offsetB16 = ori_offsetB00 + N * 32;
    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; // (tid *4 + tid / (BLOCK_SIZE_N/4) / 4 * 2)*4

    // B is stored in sparse format, thus, should be dealt with differently


    uint tid224 = tid & 224;
    uint storAB = (tx * 32 * 4 + ty + tx * 2) * 4;
    uint loadA = (((tid & 16) >> 3) | (tid & 1)) << 4;
    uint loadB = ((tid >> 1) & 7) << 4;
    loadA += (tid224 * 32) + (tid224 / 2);
    loadB += (tid224 * 32) + (tid224 / 2);

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(storAB) : );
    asm("mov.b32 %0, %0;" : "+r"(loadA)  : );
    asm("mov.b32 %0, %0;" : "+r"(loadB)  : );

    float regC[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            regC[i][j] = 0.0f;

    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    int n_token = expert_count[exp_id];
    int index_start = exp_id * TMAX + by * BLOCK_SIZE_M;
    int index_end = min(index_start + BLOCK_SIZE_M, exp_id*TMAX+n_token);

    if(index_start<index_end){
        // load the m-dimension index to the shared memory
        if(tid<index_end-index_start ){
            m_index[tid] = sparse_index[index_start+tid];
        }
        __syncthreads();
        uint ori_offsetA00 = m_index[ty] * K + k;
        uint ori_offsetA16 = m_index[ty+16] * K + k;
        for(int k_seq=0; k_seq < (int) ( K / 64) ; k_seq++){
            uint offsetB00 = ori_offsetB00 + 64 * k_seq * N;
            uint offsetB16 = ori_offsetB16 + 64 * k_seq * N;
            uint offsetA00 = ori_offsetA00 + 64 * k_seq;
            uint offsetA16 = ori_offsetA16 + 64 * k_seq;
            float4 a00 = {0}, a16 = {0};
            float4 b00 = {0}, b16 = {0};
            if(ty<index_end-index_start)
                a00 = __ldg((const float4*)(add_ptr_f(tokens, offsetA00)));
            if(ty+16<index_end-index_start)
                a16 = __ldg((const float4*)(add_ptr_f(tokens, offsetA16)));
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB16)));


            __syncthreads();

            *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
            *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
            *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
            *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
            *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
            *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
            *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
            *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;

            *(float*)&bShare[storB + (1*65*32)*4] = b00.x;
            *(float*)&bShare[storB + (1*65*32 + 1)*4] = b00.y;
            *(float*)&bShare[storB + (1*65*32 + 2)*4] = b00.z;
            *(float*)&bShare[storB + (1*65*32 + 3)*4] = b00.w;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32)*4] = b16.x;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 1)*4] = b16.y;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 2)*4] = b16.z;
            *(float*)&bShare[storB + (32*32 + 8*2 + 1*65*32 + 3)*4] = b16.w;
            __syncthreads();

            float regA[8], regB[4];
            #pragma unroll
            for (int j = 0; j < 4; j++)
            {
                // fetch outer product data
                *(float4*)&regA[0] = *(float4*)&bShare[loadA + (32*j +  0)*4];
                *(float4*)&regA[4] = *(float4*)&bShare[loadA + (32*j + 16)*4];
                *(float4*)&regB[0] = *(float4*)&bShare[loadB + (32*j + 65*32)*4];

                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 4; j++)
                        regC[i][j] += regA[i] * regB[j];
            }
            #pragma unroll
            for (int j = 4; j < 8; j++)
            {
                *(float2*)&regA[0] = *(float2*)&bShare[loadA + (32*j +  0 + (j/4)*2)*4];
                *(float2*)&regA[2] = *(float2*)&bShare[loadA + (32*j +  2 + (j/4)*2)*4];
                *(float2*)&regA[4] = *(float2*)&bShare[loadA + (32*j + 16 + (j/4)*2)*4];
                *(float2*)&regA[6] = *(float2*)&bShare[loadA + (32*j + 18 + (j/4)*2)*4];
                *(float2*)&regB[0] = *(float2*)&bShare[loadB + (32*j +  0 + (j/4)*2 + 65*32)*4];
                *(float2*)&regB[2] = *(float2*)&bShare[loadB + (32*j +  2 + (j/4)*2 + 65*32)*4];

                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 4; j++)
                        regC[i][j] += regA[i] * regB[j];
            }
        }
        
        asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid)   :);
        asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(bx)   :);
        asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(by) :);

        ty = ((tid & 16) >> 3) + (tid & 1);
        tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

        uint storC = ty*32*8*4 + tx*4;

        tx = tid % 16;
        ty = tid / 16;

        uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

        // C should be row major
        // C += (bx * BLOCK_SIZE_M + ty) * N + (by * BLOCK_SIZE_N + tx * 2);

        __syncthreads();
        *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
        *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
        *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
        *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
        __syncthreads();

        float2 c2[8];
        for (int i = 0; i < 8; i++)
            c2[i] = *(float2*)&fShare[readC + i*32];

        // Tree reduce
        for (int j = 4; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                c2[i] = _add(c2[i], c2[i+j]);
        float * WC;
        WC = C + m_index[ty] * N  + blockIdx.x * BLOCK_SIZE_N + tx *2;
        if(ty < index_end-index_start)
            *(float2*)WC = c2[0];

        __syncthreads();
        *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
        *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
        *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
        *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
        __syncthreads();

        for (int i = 0; i < 8; i++)
            c2[i] = *(float2*)&fShare[readC + i*32];

        // Tree reduce
        for (int j = 4; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                c2[i] = _add(c2[i], c2[i+j]);

        WC = C + m_index[ty+16] * N  + blockIdx.x * BLOCK_SIZE_N + tx *2;
        if(ty + 16 < index_end - index_start){
            *(float2*)WC = c2[0];
        }

    }
    
}

 __global__ void convert_sparse_index(int * router_index, int total_token, int * expert_count, int *sparse_index, const int TMAX)
{
    int bx =  blockIdx.x;
    int tid = threadIdx.x;
    int offset = bx * blockDim.x + tid;
    int exp_id = router_index[offset];
    int pos_id = atomicAdd(&expert_count[exp_id], 1);
    sparse_index[exp_id*TMAX+pos_id] = offset;

}


void convert_index(
    int * router_index,
    int * sparse_index,
    int * expert_count,
    int total_token,
    int n_expert,
    const int TMAX
    )
{
    // convert the sparse index on the fly
    dim3 blockDim(256);
    dim3 gridDim(total_token/256);
    CUDA_SAFE_CALL(cudaMemset((void*)expert_count, 0, sizeof(int)*n_expert));
    convert_sparse_index<<<gridDim, blockDim>>>(router_index, total_token, expert_count, sparse_index, TMAX);
    // compuate the sparse forward with stile

}


void forward_function(
    int * router_index,
    int * sparse_index,
    float* tokens,
    float* weight,
    float* output,
    int * expert_count,
    int total_token,
    int in_hidden,
    int out_hidden,
    int n_expert,
    const int GLOBAL_M,
    const int TMAX
    )
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 64;
    const int BLOCK_SIZE_N = 32;
    const int max_block = (GLOBAL_M -1 + BLOCK_SIZE_M)/BLOCK_SIZE_M;
    dim3 blockDim(256);
    dim3 gridDim(out_hidden/BLOCK_SIZE_N, max_block, n_expert);
    BATCH_BLOCK_SPARSE_MATMUL<<<gridDim, blockDim>>>(tokens, sparse_index, expert_count, weight, output, GLOBAL_M, in_hidden, out_hidden, TMAX);
}


void moe_sparse_convert_index(
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count
)
{
    // tokens: [total_token, in_hidden]
    cudaSetDevice(router_index.get_device());
    int total_token = router_index.size(0);
    int n_expert = expert_count.size(0);
    const int TMAX = sparse_index.size(1); // the max token each expert can take
    assert(router_index.size(0) == total_token);
    AT_DISPATCH_INTEGRAL_TYPES(router_index.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { convert_index(
                                    router_index.data_ptr<int>(),
                                    sparse_index.data_ptr<int>(),
                                    expert_count.data_ptr<int>(),
                                    total_token,
                                    n_expert,
                                    TMAX
                                ); }));



}

at::Tensor moe_sparse_forward(
    torch::Tensor tokens,
    torch::Tensor weight,
    torch::Tensor router_index,
    torch::Tensor sparse_index,
    torch::Tensor expert_count,
    const int GLOBAL_M
)
{
    cudaSetDevice(router_index.get_device());
    // tokens: [total_token, in_hidden]
    int total_token = tokens.size(0);
    int in_hidden = tokens.size(1);
    // weight : [n_expert, out_hidden, in_hidden]
    int n_expert = weight.size(0);
    int out_hidden = weight.size(2);
    const int TMAX = sparse_index.size(1); // the max token each expert can take
    assert(router_index.size(0) == total_token);
    torch::Tensor output = torch::zeros({total_token, out_hidden}, tokens.options());
    AT_DISPATCH_FLOATING_TYPES(tokens.type(), "seqlen_dynamic_sparse_attention", ([&]
                            { forward_function(
                                    router_index.data_ptr<int>(),
                                    sparse_index.data_ptr<int>(),
                                    tokens.data_ptr<float>(),
                                    weight.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    expert_count.data_ptr<int>(),
                                    total_token,
                                    in_hidden,
                                    out_hidden,
                                    n_expert,
                                    GLOBAL_M,
                                    TMAX
                                ); }));

    return output;


}