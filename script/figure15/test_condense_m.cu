#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cublas.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>
// #include <math>
#include <algorithm>
#include <assert.h>
// CUDA runtime
#include <cuda.h>
using namespace std;
#define OFFSET(row, col, ld) ((row) * ld + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&pointer))[0]
#define FETCH_UINT32(pointer) (reinterpret_cast<unsigned int*>(&(pointer))[0])
#define FETCH_UINT4(pointer) (reinterpret_cast<uint4*>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])
#define FETCH_INT32(pointer) (reinterpret_cast<int*>(&(pointer))[0])
#define MAX_BLOCK_THREAD_COUNT 1024
#define MAX_TILE_PER_BLOCK 16
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

void init_mask_blockwise(int * ptr, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            int pos = i*block_h*N+j*block_w;
            if (pro < sparsity)
            {
                ptr[pos] = 0;
            }
            else
            {
                ptr[pos] = 1;
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);

}
__device__ __forceinline__ float2  _add(float2 x, float2 y) { float2 res; res.x = x.x + y.x; res.y = x.y + y.y; return res; }

void convert_bcsr_condense_m(int * mask, float* dense_val, int M, int N, int block_h, int block_w, int *row_ptr, int* col_idx, float * val )
{
    vector<int> vr(N/block_w+1, 0);
    vector<vector<int>> vc(M/block_h, vector<int>());
    int block_nnz = 0;
    assert(M%block_h==0);
    assert(N%block_w==0);
    for(int cid=0; cid<N/block_w; cid++){
        for(int rid=0; rid<M/block_h; rid++){
            int flag = 0;
            for(int i=0; i<block_h; i++){
                for(int j=0; j<block_w; j++){
                    int _pos = (rid * block_h + i) * N + cid * block_w + j;
                    if(mask[_pos]>0)
                        flag = 1;
                }
            }
            if(flag){
                vc[cid].push_back(rid);
            }
        }
    }
    row_ptr[0]=0;
    for(int i=0;i<N/block_w;i++){
        row_ptr[i+1] = row_ptr[i] + vc[i].size();
        for(int j =0; j<vc[i].size(); j++){
            int _block_idx = row_ptr[i]+j;
            col_idx[_block_idx] = vc[i][j];
            for(int b_i=0; b_i<block_h; b_i++){
                for(int b_j=0; b_j<block_w; b_j++){
                    int pos_1 = _block_idx * block_h *block_w + b_i * block_w + b_j;
                    int pos_2 = (b_i + vc[i][j] * block_h) * N + (b_j + block_w * i);
                    val[pos_1] = dense_val[pos_2];
                }
            }
        }
    }

}

__global__ void BATCH_BLOCK_SPARSE_MATMUL_CONDENSE_DIM_M(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE)
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

    A_val += SPARSE_VAL_SIZE * blockIdx.z;
    B += K * N * blockIdx.z;
    C += M * N * blockIdx.z;

    assert(blockDim.x % 32 == 0);
    assert(BLOCK_SIZE_K % BLOCK_W==0);
    uint n_warp = 8; // blockDim.x / 32
    assert(THREAD_SIZE_K % n_warp == 0);
    assert(K % THREAD_SIZE_K == 0);

    assert(BLOCK_SIZE_M == BLOCK_SIZE_N);
    __shared__ float fShare[65 * 32 * 2];
    __shared__ int m_index[BLOCK_SIZE_M];
    char* bShare = (char*)fShare;

    uint tid = threadIdx.x;
    uint bx = blockIdx.x; // M
    uint by = blockIdx.y; // K

    uint tx = tid % 16;
    uint ty = tid / 16;
    assert(THREAD_SIZE_K % 16 == 0);
    uint k = tx * 4;

    uint storB = (tid * 4 + tid / (BLOCK_SIZE_N/4) / 4 *2) * 4; 

    uint ori_offset_A00 = A_row[by] * BLOCK_H * BLOCK_SIZE_K + k;
    uint ori_offset_B00 = (by * BLOCK_SIZE_K + tid / (BLOCK_SIZE_N/4)) * N + (tid % (BLOCK_SIZE_N/4)) * 4;
    uint ori_offset_B32 = ori_offset_B00 + 32 * N;
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


    // bx means in index of this thread block on N dimonsion
    // index_start and index_end is block index on column
    float4 const0 = {0};
    float4 a00 = {0,0,0,0}, a16 = {0,0,0,0};


    int index_start = A_row[by] + bx * BLOCK_SIZE_M, index_end = min(A_row[by+1], index_start + BLOCK_SIZE_M);
    if (index_start < index_end){

        if(tid<index_end-index_start){
            m_index[tid] = A_col[tid+index_start];
        }
        // if(threadIdx.x == 0){
        //     printf("bx:%d by:%d index_start:%d index_end:%d\n", bx, by, index_start, index_end);
        // }

        // Load the A_global to A_shared
        __syncthreads();
        uint offsetA00 = 0;
        uint offsetA16 = 0;
        if(ty<index_end-index_start){
            offsetA00 =(bx * BLOCK_SIZE_M + ty) * BLOCK_H * BLOCK_SIZE_K  + ori_offset_A00;
            a00 = __ldg((const float4*)(add_ptr_f(A_val, offsetA00)));
        }
        if(ty+16<index_end-index_start){
            offsetA16 = (bx * BLOCK_SIZE_M + ty + 16) * BLOCK_H * BLOCK_SIZE_K  + ori_offset_A00;
            a16 = __ldg((const float4*)(add_ptr_f(A_val, offsetA16)));
        }

        for(int block_n_id =0; block_n_id < N/BLOCK_SIZE_N; block_n_id++)
        {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 4; j++)
                    regC[i][j] = 0.0f;

            *(float*)&bShare[storAB + (0*32 +  0 + 0*65*32)*4] = a00.x;
            *(float*)&bShare[storAB + (1*32 +  0 + 0*65*32)*4] = a00.y;
            *(float*)&bShare[storAB + (2*32 +  0 + 0*65*32)*4] = a00.z;
            *(float*)&bShare[storAB + (3*32 +  0 + 0*65*32)*4] = a00.w;
            *(float*)&bShare[storAB + (0*32 + 16 + 0*65*32)*4] = a16.x;
            *(float*)&bShare[storAB + (1*32 + 16 + 0*65*32)*4] = a16.y;
            *(float*)&bShare[storAB + (2*32 + 16 + 0*65*32)*4] = a16.z;
            *(float*)&bShare[storAB + (3*32 + 16 + 0*65*32)*4] = a16.w;
            uint offsetB00 = block_n_id * BLOCK_SIZE_N + ori_offset_B00;
            uint offsetB32 = block_n_id * BLOCK_SIZE_N + ori_offset_B32;;
            float4 b00 = {0,0,0,0}, b16 = {0,0,0,0};
            b00 = __ldg((const float4*)(add_ptr_f(B, offsetB00)));
            b16 = __ldg((const float4*)(add_ptr_f(B, offsetB32)));
            // if(offsetA00>1024*1024 || offsetA16 > 1024*1024 || offsetB00>1024*1024 || offsetB32>1024*1024){
            //         printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d m_index[ty]:%d m_index[ty+16]:%d ori_offset_A00:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, offsetB00, offsetB32, m_index[ty], m_index[ty+16], ori_offset_A00, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);

            // }

            // if(bx==0 && ty==11 && by==0){
            //     printf("CK1:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     printf("CK2:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
                
            // }

            // if(bx==0){
            //     if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK1:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetA00:%d offsetA16:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetA00, offsetA16, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     }
            //     if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         printf("CK2:!!!bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     }
            // }
            // if(bx==31 && by==0 && block_n_id==0){
            //     printf("tid:%d tx:%d ty:%d offsetB00:%d a00:(%f %f %f %f) b00:(%f %f %f %f)\n", tid, tx, ty, offsetB00, a00.x, a00.y, a00.z, a00.w, b00.x, b00.y, b00.z, b00.w);
            // }
            // __syncthreads();
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
                    for (int j = 0; j < 4; j++){
                        regC[i][j] += regA[i] * regB[j];

                        // if(bx==31 && by==0 && block_n_id==0){
                        //     printf("tid:%d tx:%d ty:%d i:%d j:%d regc[i][j]:%f regA[i]%f regB[j]:%f\n", tid, tx, ty, i, j, regC[i][j], regA[i], regB[j]);
                        // }
                    }
            }
        
            ty = ((tid & 16) >> 3) + (tid & 1);
            tx = ((tid >> 1) & 7) + ((tid & 224) >> 2) + (ty << 2);

            uint storC = ty*32*8*4 + tx*4;

            tx = tid % 16;
            ty = tid / 16;

            uint readC = ty*32*8 + tx*2 + ((tid & 192)>>2);

            // C should be row major
            float * wC;

            // printf("DEBUG: bx:%d by:%d tid:%d tx:%d ty:%d m_index[ty]:%d block_n_id:%d\n", bx, by, tid, tx, ty, m_index[ty], block_n_id);
            __syncthreads();
            *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[0];
            *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[1];
            *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[2];
            *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[3];
            __syncthreads();

            float2 c2[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                c2[i] = *(float2*)&fShare[readC + i*32];

            // Tree reduce
            #pragma unroll
            for (int j = 4; j > 0; j >>= 1)
                for (int i = 0; i < j; i++)
                    c2[i] = _add(c2[i], c2[i+j]);

            // if(bx==31 && by==0 && block_n_id==0){
            //     printf("tid:%d tx:%d ty:%d  m_index[ty]:%d c2:(%f, %f) regc: %f %f %f %f \n", tid, tx, ty, m_index[ty], c2[0].x, c2[0].y, regC[0][0], regC[0][1], regC[0][2], regC[0][3]);
            // }
            // *(float2*)C = c2[0];
            // if(bx==0 && block_n_id==31){
            //     if(c2[0].x <63.9999 || c2[0].y<=63.999){
            //         printf("bx:%d by:%d block_n_id:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f\n", bx, by, block_n_id, tid, tx, ty, c2[0].x, c2[0].y);
            //     }
            // }
            // if(bx==0&& (c2[0].x<64||c2[0].y<64)){
            //     // if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK3: bx:%d by:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, c2[0].x, c2[0].y , offsetA00, offsetA16, offsetB00, offsetB32, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     // }
            //     // if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         // printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     // }
            // }
            if(ty < index_end-index_start){
                wC = C + m_index[ty] * BLOCK_H * N + (block_n_id * BLOCK_SIZE_N + tx * 2);
                atomicAdd(wC, c2[0].x);
                atomicAdd(wC+1, c2[0].y);
            }
            __syncthreads();
            *(float4*)&fShare[storC + 0*32*8] = *(float4*)regC[4];
            *(float4*)&fShare[storC + 1*32*8] = *(float4*)regC[5];
            *(float4*)&fShare[storC + 2*32*8] = *(float4*)regC[6];
            *(float4*)&fShare[storC + 3*32*8] = *(float4*)regC[7];
            __syncthreads();

            for (int i = 0; i < 8; i++)
                c2[i] = *(float2*)&fShare[readC + i*32];

            // Tree reduce
            #pragma unroll
            for (int j = 4; j > 0; j >>= 1)
                for (int i = 0; i < j; i++)
                    c2[i] = _add(c2[i], c2[i+j]);
            // if(bx==0 && (c2[0].x<64||c2[0].y<64)){
            //     // if(a00.x+a00.y+a00.z+a00.w<3.999 || a16.x+a16.y+a16.z+a16.w<3.99){
            //         printf("CK3: bx:%d by:%d tid:%d tx:%d ty:%d c2[0].x:%f c2[0].y:%f offsetA00:%d offsetA16:%d offsetB00:%d offsetB32:%d a00:(%f %f %f %f) a16:(%f %f %f %f)\n",bx, by, tid, tx, ty, c2[0].x, c2[0].y , offsetA00, offsetA16, offsetB00, offsetB32, a00.x, a00.y, a00.z, a00.w, a16.x, a16.y, a16.z, a16.w);
            //     // }
            //     // if(b00.x+b00.y+b00.z+b00.w<3.999 || b16.x+b16.y+b16.z+b16.w<3.99){
            //         // printf("bx:%d by:%d tid:%d tx:%d ty:%d offsetB00:%d offsetB32:%d b00:(%f %f %f %f) b16:(%f %f %f %f)\n",bx, by, tid, tx, ty, offsetB00, offsetB32, b00.x, b00.y, b00.z, b00.w, b16.x, b16.y, b16.z, b16.w);
            //     // }
            // }
            if(ty+16<index_end-index_start){
                wC = C + m_index[ty+16] * BLOCK_H * N + (block_n_id * BLOCK_SIZE_N + tx * 2);
                atomicAdd(wC, c2[0].x);
                atomicAdd(wC+1, c2[0].y);
            }
            __syncthreads();
        }

    }
}

void openai_bmm_32_64_32_condense_dim_m_launch(float* A_val, int* A_row, int* A_col, float*B, float*C, int GLOBAL_M, int GLOBAL_K, int GLOBAL_N, int BLOCK_H, int BLOCK_W, int SPARSE_VAL_SIZE, int batchsize)
{
    const dim3 dimBlock(256);
    assert(BLOCK_W==64);
    dim3 dimGrid(GLOBAL_M/32, GLOBAL_K/64, batchsize);
    BATCH_BLOCK_SPARSE_MATMUL_CONDENSE_DIM_M<<<dimGrid, dimBlock>>>(A_val, A_row, A_col, B, C, GLOBAL_M, GLOBAL_K, GLOBAL_N, BLOCK_H, BLOCK_W, SPARSE_VAL_SIZE);
}

int main()
{
    int M, K, N;
    M = 4096;
    K = 4096;
    N = 4096;
    const int n_iter = 1000;
    float sparsity_ratio = 0.99;
    const int BLOCK_H = 1;
    const int BLOCK_W = 64;
    // const int BLOCK_W = 1;
    cudaEvent_t time_start, time_end;
    CUDA_SAFE_CALL(cudaEventCreate(&time_start));
    CUDA_SAFE_CALL(cudaEventCreate(&time_end));
    float msecTotal = 0;
    float * A, *B, *C, *val;
    float * dA, *dB, *dC, *d_val;
    int * mask, *d_mask, *row, *d_row, *row_pos, *d_row_pos, *col, *d_col, *d_extra_buffer;
    A = (float*) malloc(sizeof(float) * M * K);
    B = (float*) malloc(sizeof(float) * K * N);
    C = (float*) malloc(sizeof(float) * M * N);
    mask = (int*) malloc(sizeof(int) * M * K);
    row = (int*) malloc(sizeof(int) * (K+1));
    col = (int*) malloc(sizeof(int) *  M * K / BLOCK_H / BLOCK_W);
    val = (float*) malloc(sizeof(float) * M * K);
    init_mask_blockwise(mask, M, K, BLOCK_H, BLOCK_W, sparsity_ratio);
    // apply mask
    for(int i=0; i< M*K; i++){
        A[i] = A[i] * mask[i];
    }
    convert_bcsr_condense_m(mask, A, M, K, BLOCK_H, BLOCK_W, row, col, val);
    int block_nnz = row[K/BLOCK_W];
    int sparse_val_size = (block_nnz * BLOCK_H * BLOCK_W);
    printf("Block NNZ: %d\n", block_nnz);
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, sizeof(int) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&d_row, sizeof(int) * (M + 1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_row_pos, sizeof(int) * M * K / BLOCK_H / BLOCK_W));
    CUDA_SAFE_CALL(cudaMalloc(&d_val, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_SAFE_CALL(cudaMalloc(&dB, sizeof(float) * N * K));
    CUDA_SAFE_CALL(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_SAFE_CALL(cudaMemset(dC, 0, sizeof(float)*M*N));
    CUDA_SAFE_CALL(cudaMalloc(&d_extra_buffer, sizeof(float) * M * K));
    
    CUDA_SAFE_CALL(cudaMemcpy(d_mask, mask, sizeof(int) * M * K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row, row, sizeof(int)*(K+1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col, col, sizeof(int)*M * K / BLOCK_H / BLOCK_W, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_val, val, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventRecord(time_start));
    for(int run=0; run<n_iter; run++){
        openai_bmm_32_64_32_condense_dim_m_launch(d_val, d_row, d_col, dB, dC, M, K, N, BLOCK_H, BLOCK_W, sparse_val_size, 1);
    }
    CUDA_SAFE_CALL(cudaEventRecord(time_end));
    CUDA_SAFE_CALL(cudaEventSynchronize(time_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&msecTotal, time_start, time_end));
    printf("Time Cost: %.3fms\n", msecTotal/n_iter);
    // CUDA_SAFE_CALL(cudaMemcpy(row, d_row, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(col, d_col, sizeof(int) * M * K / BLOCK_H / BLOCK_W, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMemcpy(val, d_val, sizeof(float) * M * K, cudaMemcpyDeviceToHost));
    // CUDA_SAFE_CALL(cudaMem)

    return 0;
}