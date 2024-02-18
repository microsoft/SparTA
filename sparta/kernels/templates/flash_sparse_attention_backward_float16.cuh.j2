{# Copyright (c) Microsoft Corporation. #}
{# Licensed under the MIT license. #}

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

{% set WARP_SIZE = 32 %}
{% set FRAG_SIZE = 256 %}
{% set TS_WARP_SIZE_N_VALUE = FRAG_SIZE // TS_WARP_SIZE_M_VALUE %}
{% set TD_WARP_SIZE_N_VALUE = FRAG_SIZE // TD_WARP_SIZE_M_VALUE %}
{% set SD_WARP_SIZE_N_VALUE = FRAG_SIZE // SD_WARP_SIZE_M_VALUE %}
{% set BLOCK_SIZE = BLOCK_SIZE_T_VALUE * BLOCK_SIZE_S_VALUE %}
{% set THREAD_SIZE = BLOCK_SIZE // THREADS_PER_BLOCK %}{# 8 <= THREAD_SIZE <= BLOCK_SIZE_S_VALUE #}
{% set WARP_REDUCE_SIZE = BLOCK_SIZE_S_VALUE // THREAD_SIZE %}{# WARP_REDUCE_SIZE <= WARP_SIZE #}

const int BS = {{ BLOCK_SIZE_S_VALUE }};
const int BT = {{ BLOCK_SIZE_T_VALUE }};
const int D = {{ GLOBAL_SIZE_D_VALUE }};
const int TS_WARP_M = {{ TS_WARP_SIZE_M_VALUE }};
const int TS_WARP_N = {{ TS_WARP_SIZE_N_VALUE }};
const int TS_WARP_K = 16;
const int TD_WARP_M = {{ TD_WARP_SIZE_M_VALUE }};
const int TD_WARP_N = {{ TD_WARP_SIZE_N_VALUE }};
const int TD_WARP_K = 16;
const int SD_WARP_M = {{ SD_WARP_SIZE_M_VALUE }};
const int SD_WARP_N = {{ SD_WARP_SIZE_N_VALUE }};
const int SD_WARP_K = 16;

const int T = {{ THREAD_SIZE }};
const int THREADS = {{ THREADS_PER_BLOCK }};{# THREADS_PER_BLOCK >= WARP_SIZE #}
const int WARPS = THREADS / {{ WARP_SIZE }};
const int SD = T * D / BS;

const int SMEM_THREADS_D = D / 8;
const int SMEM_THREADS_N = {{ THREADS_PER_BLOCK }} / SMEM_THREADS_D;
const int TS_WARPS_N = BS / TS_WARP_N;
const int TS_STRIDE_M = TS_WARP_M * (WARPS / TS_WARPS_N);
const int TD_WARPS_N = D / TD_WARP_N;
const int TD_STRIDE_M = TD_WARP_M * (WARPS / TD_WARPS_N);
const int SD_WARPS_N = D / SD_WARP_N;
const int SD_STRIDE_M = SD_WARP_M * (WARPS / SD_WARPS_N);

const int D_PAD = 8;
const int S_PAD = 8;

extern "C" {

__global__ void BLOCK_SPARSE_FLASH_ATTENTION_BACKWARD(
    half* Q,
    half* K,
    half* V,
    half* O,
    half* dQ,
    half* dK,
    half* dV,
    half* dO,
    float* ML,
    {# unsigned char* mask, #}
    uint* block_idx,
    uint Ns,
    uint Nt,
    uint block_nnz
) {
    int H = gridDim.x;
    int HEAD_IDX = (blockIdx.y * H + blockIdx.x);
    {% if TRANSPOSED %}
    Q += HEAD_IDX * Nt * D;
    K += HEAD_IDX * Ns * D;
    V += HEAD_IDX * Ns * D;
    O += HEAD_IDX * Nt * D;
    dQ += HEAD_IDX * Nt * D;
    dK += HEAD_IDX * Ns * D;
    dV += HEAD_IDX * Ns * D;
    dO += HEAD_IDX * Nt * D;
    int stride = D;
    {% else %}
    Q += blockIdx.y * Nt * H * D + blockIdx.x * D;
    K += blockIdx.y * Ns * H * D + blockIdx.x * D;
    V += blockIdx.y * Ns * H * D + blockIdx.x * D;
    O += blockIdx.y * Nt * H * D + blockIdx.x * D;
    dQ += blockIdx.y * Nt * H * D + blockIdx.x * D;
    dK += blockIdx.y * Ns * H * D + blockIdx.x * D;
    dV += blockIdx.y * Ns * H * D + blockIdx.x * D;
    dO += blockIdx.y * Nt * H * D + blockIdx.x * D;
    int stride = H * D;
    {% endif %}
    ML += Nt * 2 * HEAD_IDX;

    uint WARP_OFFSET = ((threadIdx.x / {{ WARP_REDUCE_SIZE }}) * {{ WARP_REDUCE_SIZE }}) % {{ WARP_SIZE }};
    uint WARP_MASK = 0b{% for _ in range(WARP_REDUCE_SIZE) %}1{% endfor %} << WARP_OFFSET;

    extern __shared__ half shared[];
    half* shared_Q = &shared[0];
    half* shared_P = &shared_Q[BT * (D + D_PAD)];
    half* shared_K = &shared_P[BT * (BS + S_PAD)];
    half* shared_V = &shared_K[BS * (D + D_PAD)];
    half* shared_O = &shared_V[BS * (D + D_PAD)];
    half* shared_dK = &shared_O[BT * (D + D_PAD)];
    half* shared_dV = &shared_dK[BS * (D + D_PAD)];
    {# __shared__ half shared_Q[BT * (D + D_PAD)];
    __shared__ half shared_P[BT * (BS + S_PAD)];
    __shared__ half shared_K[BS * (D + D_PAD)];
    __shared__ half shared_V[BS * (D + D_PAD)];
    __shared__ half shared_O[BT * (D + D_PAD)];
    __shared__ half shared_dK[BS * (D + D_PAD)];
    __shared__ half shared_dV[BS * (D + D_PAD)]; #}

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int SMEM_TID_N = tid / SMEM_THREADS_D;
    int SMEM_TID_D = tid % SMEM_THREADS_D * 8;

    int wid = threadIdx.x / {{ WARP_SIZE }};
    int ts_wx = wid % TS_WARPS_N;
    int ts_wy = wid / TS_WARPS_N;
    int td_wx = wid % TD_WARPS_N;
    int td_wy = wid / TD_WARPS_N;
    int sd_wx = wid % SD_WARPS_N;
    int sd_wy = wid / SD_WARPS_N;
    int tx = threadIdx.x % {{ WARP_REDUCE_SIZE }};
    int ty = threadIdx.x / {{ WARP_REDUCE_SIZE }};

    wmma::fragment<wmma::matrix_a, TS_WARP_M, TS_WARP_N, TS_WARP_K, half, wmma::row_major> frag_ts_a;
    wmma::fragment<wmma::matrix_b, TS_WARP_M, TS_WARP_N, TS_WARP_K, half, wmma::col_major> frag_ts_b;
    wmma::fragment<wmma::accumulator, TS_WARP_M, TS_WARP_N, TS_WARP_K, half> frag_ts_c;
    wmma::fragment<wmma::matrix_a, TD_WARP_M, TD_WARP_N, TD_WARP_K, half, wmma::row_major> frag_td_a;
    wmma::fragment<wmma::matrix_b, TD_WARP_M, TD_WARP_N, TD_WARP_K, half, wmma::row_major> frag_td_b;
    wmma::fragment<wmma::accumulator, TD_WARP_M, TD_WARP_N, TD_WARP_K, half> frag_td_c;
    wmma::fragment<wmma::matrix_a, SD_WARP_M, SD_WARP_N, SD_WARP_K, half, wmma::col_major> frag_sd_a;
    wmma::fragment<wmma::matrix_b, SD_WARP_M, SD_WARP_N, SD_WARP_K, half, wmma::row_major> frag_sd_b;
    wmma::fragment<wmma::accumulator, SD_WARP_M, SD_WARP_N, SD_WARP_K, half> frag_sd_c;
    float2 tmp_float2;
    half tmp_half8[8];
    half tmp_half8_2[8];
    float frag_P[T];
    float frag_S[T];

    float temperature = __frsqrt_rn((float)D);
    float row_sum;

    int last_col_idx = -1;
    {# BCSC #}
    for (int block = 0; block < block_nnz; block++) {
        uint idx = block_idx[block];
        int row_idx = idx & 0xffff;
        int col_idx = idx >> 16;
        // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("#%d: (%d, %d)\n", block, row_idx, col_idx);

        {# Load Q #}
        #pragma unroll
        for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
            *((float4*)(&shared_Q[k * (D + D_PAD) + SMEM_TID_D])) =
                *((float4*)(&Q[(row_idx * BT + k) * stride + SMEM_TID_D]));
        }
        if (col_idx != last_col_idx) {
            if (last_col_idx >= 0) {
                {# Save dK, dV #}
                #pragma unroll
                for (int k = SMEM_TID_N; k < BS; k += SMEM_THREADS_N) {
                    *((float4*)(&dK[(last_col_idx * BS + k) * stride + SMEM_TID_D])) =
                        *((float4*)(&shared_dK[k * (D + D_PAD) + SMEM_TID_D]));
                    *((float4*)(&dV[(last_col_idx * BS + k) * stride + SMEM_TID_D])) =
                        *((float4*)(&shared_dV[k * (D + D_PAD) + SMEM_TID_D]));
                }
            }
            {# Load K, V, dK, dV #}
            #pragma unroll
            for (int k = SMEM_TID_N; k < BS; k += SMEM_THREADS_N) {
                *((float4*)(&shared_dK[k * (D + D_PAD) + SMEM_TID_D])) =
                    *((float4*)(&dK[(col_idx * BS + k) * stride + SMEM_TID_D]));
                *((float4*)(&shared_dV[k * (D + D_PAD) + SMEM_TID_D])) =
                    *((float4*)(&dV[(col_idx * BS + k) * stride + SMEM_TID_D]));
                *((float4*)(&shared_K[k * (D + D_PAD) + SMEM_TID_D])) =
                    *((float4*)(&K[(col_idx * BS + k) * stride + SMEM_TID_D]));
                *((float4*)(&shared_V[k * (D + D_PAD) + SMEM_TID_D])) =
                    *((float4*)(&V[(col_idx * BS + k) * stride + SMEM_TID_D]));
            }
            last_col_idx = col_idx;
        }
        __syncthreads();

        {# Calc P = Q K^T #}
        #pragma unroll
        for (int j = 0; j < BT; j += TS_STRIDE_M) {
            wmma::fill_fragment(frag_ts_c, 0.0);
            #pragma unroll
            for (int k = 0; k < D; k += TS_WARP_K) {
                wmma::load_matrix_sync(frag_ts_a, &shared_Q[(j + ts_wy * TS_WARP_M) * (D + D_PAD) + k], D + D_PAD);
                wmma::load_matrix_sync(frag_ts_b, &shared_K[(ts_wx * TS_WARP_N) * (D + D_PAD) + k], D + D_PAD);
                wmma::mma_sync(frag_ts_c, frag_ts_a, frag_ts_b, frag_ts_c);
            }
            wmma::store_matrix_sync(
                &shared_P[(j + ts_wy * TS_WARP_M) * (BS + S_PAD) + ts_wx * TS_WARP_N],
                frag_ts_c,
                BS + S_PAD,
                wmma::mem_row_major
            );
        }
        __syncthreads();

        {# Load M, L, P #}
        #pragma unroll
        for (int i = 0; i < T; i += 8) {
            *((float4*)(&tmp_half8[0])) = *((float4*)(&shared_P[ty * (BS + S_PAD) + tx * T + i]));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                frag_P[i + j] = __half2float(tmp_half8[j]);
            }
        }
        if (tx == 0) {
            tmp_float2 = ((float2*)(&ML[(row_idx * BT + ty) * 2]))[0];
        }
        __syncthreads();
        tmp_float2.x = __shfl_sync(WARP_MASK, tmp_float2.x, WARP_OFFSET);
        tmp_float2.y = __shfl_sync(WARP_MASK, tmp_float2.y, WARP_OFFSET);

        {# Calc S = exp(P - M) / L #}
        #pragma unroll
        for (int i = 0; i < T; i++) {
            frag_S[i] = expf(frag_P[i] * temperature - tmp_float2.x) / tmp_float2.y;
        }

        {# Save S #}
        #pragma unroll
        for (int i = 0; i < T; i += 8) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tmp_half8[j] = __float2half(frag_S[i + j]);
            }
            *((float4*)(&shared_P[ty * (BS + S_PAD) + tx * T + i])) = *((float4*)(&tmp_half8[0]));
        }

        {# Load dO #}
        #pragma unroll
        for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
            *((float4*)(&shared_O[k * (D + D_PAD) + SMEM_TID_D])) =
                *((float4*)(&dO[(row_idx * BT + k) * stride + SMEM_TID_D]));
        }
        __syncthreads();

        {# Calc dV = dV + S^T dO #}
        #pragma unroll
        for (int j = 0; j < BS; j += SD_STRIDE_M) {
            wmma::load_matrix_sync(
                frag_sd_c,
                &shared_dV[(j + sd_wy * SD_WARP_M) * (D + D_PAD) + sd_wx * SD_WARP_N],
                D + D_PAD,
                wmma::mem_row_major
            );
            #pragma unroll
            for (int k = 0; k < BT; k += SD_WARP_K) {
                wmma::load_matrix_sync(frag_sd_a, &shared_P[k * (BS + S_PAD) + j + sd_wy * SD_WARP_M], BS + S_PAD);
                wmma::load_matrix_sync(frag_sd_b, &shared_O[k * (D + D_PAD) + sd_wx * SD_WARP_N], D + D_PAD);
                wmma::mma_sync(frag_sd_c, frag_sd_a, frag_sd_b, frag_sd_c);
            }
            wmma::store_matrix_sync(
                &shared_dV[(j + sd_wy * SD_WARP_M) * (D + D_PAD) + sd_wx * SD_WARP_N],
                frag_sd_c,
                D + D_PAD,
                wmma::mem_row_major
            );
        }
        __syncthreads();

        {# Calc dS = dO V^T #}
        #pragma unroll
        for (int j = 0; j < BT; j += TS_STRIDE_M) {
            wmma::fill_fragment(frag_ts_c, 0.0);
            #pragma unroll
            for (int k = 0; k < D; k += TS_WARP_K) {
                wmma::load_matrix_sync(frag_ts_a, &shared_O[(j + ts_wy * TS_WARP_M) * (D + D_PAD) + k], D + D_PAD);
                wmma::load_matrix_sync(frag_ts_b, &shared_V[(ts_wx * TS_WARP_N) * (D + D_PAD) + k], D + D_PAD);
                wmma::mma_sync(frag_ts_c, frag_ts_a, frag_ts_b, frag_ts_c);
            }
            wmma::store_matrix_sync(
                &shared_P[(j + ts_wy * TS_WARP_M) * (BS + S_PAD) + ts_wx * TS_WARP_N],
                frag_ts_c,
                BS + S_PAD,
                wmma::mem_row_major
            );
        }
        __syncthreads();

        // if (blockIdx.x == 0 && threadIdx.x == 0 && row_idx == 0 && col_idx == 0) {
        //     printf("dS[0][0] = %f\n", (float)(shared_P[0 * (BS + S_PAD) + 0]));
        //     printf("dS[0][1] = %f\n", (float)(shared_P[0 * (BS + S_PAD) + 1]));
        //     printf("dS[1][0] = %f\n", (float)(shared_P[1 * (BS + S_PAD) + 0]));
        //     printf("dS[1][1] = %f\n", (float)(shared_P[1 * (BS + S_PAD) + 1]));
        // }

        {# Load dS #}
        #pragma unroll
        for (int i = 0; i < T; i += 8) {
            *((float4*)(&tmp_half8[0])) = *((float4*)(&shared_P[ty * (BS + S_PAD) + tx * T + i]));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                frag_P[i + j] = __half2float(tmp_half8[j]);
            }
        }

        {# Calc dP = S (dS - sum_j(dO * O)) #}
        row_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < SD; i += 8) {
            *((float4*)(&tmp_half8[0])) = *((float4*)(&O[(row_idx * BT + ty) * stride + tx * SD + i]));
            *((float4*)(&tmp_half8_2[0])) = *((float4*)(&shared_O[ty * (D + D_PAD) + tx * SD + i]));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                row_sum += __half2float(tmp_half8[j] * tmp_half8_2[j]);
            }
        }
        #pragma unroll
        for (int offset = {{ WARP_REDUCE_SIZE // 2 }}; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(WARP_MASK, row_sum, offset);
        }
        #pragma unroll
        for (int i = 0; i < T; i ++) {
            frag_P[i] = frag_S[i] * (frag_P[i] - row_sum) * temperature;
        }
        __syncthreads();

        {# Save dP #}
        #pragma unroll
        for (int i = 0; i < T; i += 8) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tmp_half8[j] = __float2half(frag_P[i + j]);
            }
            *((float4*)(&shared_P[ty * (BS + S_PAD) + tx * T + i])) = *((float4*)(&tmp_half8[0]));
        }

        // if (blockIdx.x == 0 && threadIdx.x == 0 && row_idx == 0 && col_idx == 0) {
        //     printf("dP[0][0] = %f\n", (float)(shared_P[0 * (BS + S_PAD) + 0]));
        //     printf("dP[0][1] = %f\n", (float)(shared_P[0 * (BS + S_PAD) + 1]));
        //     printf("dP[1][0] = %f\n", (float)(shared_P[1 * (BS + S_PAD) + 0]));
        //     printf("dP[1][1] = %f\n", (float)(shared_P[1 * (BS + S_PAD) + 1]));
        // }

        {# Load dQ #}
        #pragma unroll
        for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
            *((float4*)(&shared_O[k * (D + D_PAD) + SMEM_TID_D])) =
                *((float4*)(&dQ[(row_idx * BT + k) * stride + SMEM_TID_D]));
        }
        __syncthreads();

        {# Calc dK = dK + dP^T Q #}
        #pragma unroll
        for (int j = 0; j < BS; j += SD_STRIDE_M) {
            wmma::load_matrix_sync(
                frag_sd_c,
                &shared_dK[(j + sd_wy * SD_WARP_M) * (D + D_PAD) + sd_wx * SD_WARP_N],
                D + D_PAD,
                wmma::mem_row_major
            );
            #pragma unroll
            for (int k = 0; k < BT; k += SD_WARP_K) {
                wmma::load_matrix_sync(frag_sd_a, &shared_P[k * (BS + S_PAD) + j + sd_wy * SD_WARP_M], BS + S_PAD);
                wmma::load_matrix_sync(frag_sd_b, &shared_Q[k * (D + D_PAD) + sd_wx * SD_WARP_N], D + D_PAD);
                wmma::mma_sync(frag_sd_c, frag_sd_a, frag_sd_b, frag_sd_c);
            }
            wmma::store_matrix_sync(
                &shared_dK[(j + sd_wy * SD_WARP_M) * (D + D_PAD) + sd_wx * SD_WARP_N],
                frag_sd_c,
                D + D_PAD,
                wmma::mem_row_major
            );
        }
        __syncthreads();

        {# Calc dQ = dQ + dP K #}
        #pragma unroll
        for (int j = 0; j < BT; j += TD_STRIDE_M) {
            wmma::load_matrix_sync(
                frag_td_c,
                &shared_O[(j + td_wy * TD_WARP_M) * (D + D_PAD) + td_wx * TD_WARP_N],
                D + D_PAD,
                wmma::mem_row_major
            );
            #pragma unroll
            for (int k = 0; k < BS; k += TD_WARP_K) {
                wmma::load_matrix_sync(frag_td_a, &shared_P[(j + td_wy * TD_WARP_M) * (BS + S_PAD) + k], BS + S_PAD);
                wmma::load_matrix_sync(frag_td_b, &shared_K[k * (D + D_PAD) + td_wx * TD_WARP_N], D + D_PAD);
                wmma::mma_sync(frag_td_c, frag_td_a, frag_td_b, frag_td_c);
            }
            wmma::store_matrix_sync(
                &shared_O[(j + td_wy * TD_WARP_M) * (D + D_PAD) + td_wx * TD_WARP_N],
                frag_td_c,
                D + D_PAD,
                wmma::mem_row_major
            );
        }
        __syncthreads();

        {# Save dQ #}
        #pragma unroll
        for (int k = SMEM_TID_N; k < BT; k += SMEM_THREADS_N) {
            *((float4*)(&dQ[(row_idx * BT + k) * stride + SMEM_TID_D])) =
                *((float4*)(&shared_O[k * (D + D_PAD) + SMEM_TID_D]));
        }
    }

    {# Save dK, dV for the last column #}
    #pragma unroll
    for (int k = SMEM_TID_N; k < BS; k += SMEM_THREADS_N) {
        *((float4*)(&dK[(last_col_idx * BS + k) * stride + SMEM_TID_D])) =
            *((float4*)(&shared_dK[k * (D + D_PAD) + SMEM_TID_D]));
        *((float4*)(&dV[(last_col_idx * BS + k) * stride + SMEM_TID_D])) =
            *((float4*)(&shared_dV[k * (D + D_PAD) + SMEM_TID_D]));
    }
}

} // extern "C"
