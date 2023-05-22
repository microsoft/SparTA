#ifndef _NMSPARSE_UTILS_H_
#define _NMSPARSE_UTILS_H_

#include "context.cuh"
#include <algorithm>

namespace nmsparse{
    bool is_one(const int x)
    {
        return 1 == x;
    }

    bool is_divisible(const int x, const int be_devide)
    {
        return 0 == (x % be_devide);
    }

    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val){

        // check if the input is valid
        checkCtxPattern<dtype>(ctx);
        // TODO(lei): implement the conversion from dense to sparse pattern.

        return true;
    }

    template <typename dtype>
    void nmSparseInitialRandomDataElementWise(dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned)time(&t));
        unsigned int w = vecNum * (1.0f - sparsity);
        const int NUM_BANK = vecNum / 32;
        for (int batch = 0; batch < minibatch; ++batch)
            for (int i = 0; i < vecNum; ++i)
            {
                vec[i + vecNum * batch] = (dtype)rand() / RAND_MAX;
            }

        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                mat_data[i + j * w] = (dtype)rand() / RAND_MAX;
                mat_data_for_gpu[i * h + j] = mat_data[i + j * w];
            }

        int *tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
        for (int i = 0; i < vecNum / NUM_BANK; ++i)
            tmp_index[i] = i;

        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; i += w / NUM_BANK)
            {
                std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
                std::sort(tmp_index, tmp_index + w / NUM_BANK);
                for (int k = 0; k < w / NUM_BANK; ++k)
                {
                    mat_index[i + k + j * w] = tmp_index[k] + i / (1.0f - sparsity);
                    mat_index_for_gpu[(i + k) * h + j] = mat_index[i + k + j * w];
                }
            }
        }
        free(tmp_index);
    }

    template <typename dtype>
    void nmSparseInitialRandomDataAlignN(dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch, const int ALIGN_N)
    {
        // generate different seed for random number
        time_t t;
        srand((unsigned)time(&t));
        unsigned int w = vecNum * (1.0f - sparsity);
        const int NUM_BANK = vecNum / 32;
        for (int batch = 0; batch < minibatch; ++batch)
            for (int i = 0; i < vecNum; ++i)
            {
                vec[i + vecNum * batch] = (dtype)rand() / RAND_MAX;
            }

        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
            {
                mat_data[i + j * w] = (dtype)rand() / RAND_MAX;
                mat_data_for_gpu[i * h + j] = mat_data[i + j * w];
            }

        int *tmp_index = (int *)malloc(vecNum / NUM_BANK * sizeof(int));
        for (int i = 0; i < vecNum / NUM_BANK; ++i)
            tmp_index[i] = i;

        for (int j = 0; j < h; j += ALIGN_N)
        {
            for (int i = 0; i < w; i += w / NUM_BANK)
            {
                std::random_shuffle(tmp_index, tmp_index + vecNum / NUM_BANK);
                std::sort(tmp_index, tmp_index + w / NUM_BANK);
                for (int k = 0; k < w / NUM_BANK; ++k)
                {
                    for (int j_in = 0; j_in < ALIGN_N; j_in += 1)
                    {
                        mat_index[i + k + (j + j_in) * w] = tmp_index[k] + i / (1.0f - sparsity);
                        mat_index_for_gpu[(i + k) * h + (j + j_in)] = mat_index[i + k + (j + j_in) * w];
                    }
                }
            }
        }
        free(tmp_index);
    }

    template <typename dtype>
    void nmSparseInitialRandomData(nmsparseContext_t ctx, dtype *vec, dtype *mat_data, int *mat_index, dtype *mat_data_for_gpu, int *mat_index_for_gpu, int vecNum, int h, float sparsity, int minibatch){
        switch (ctx.nmsparsePattern)
        {
        case SparsePattern::ElementWise:
            nmSparseInitialRandomDataElementWise(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch);
            break;
        case SparsePattern::VectorWise4:
            nmSparseInitialRandomDataAlignN(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 4);
            break;
        case SparsePattern::VectorWise32:
            nmSparseInitialRandomDataAlignN(vec, mat_data, mat_index, mat_data_for_gpu, mat_index_for_gpu, vecNum, h, sparsity, minibatch, 32);
            break;
        default:
            throw std::runtime_error("Unsupported sparse pattern");
            break;
        }
    }
    
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val);

    template <typename dtype>
    void nmsparseCPURef_ALIGN_SHARED(dtype *vec, dtype *mat_data, int *mat_index, dtype *hostRef, const int condense_k, const int N, const int K, const int M)
    {
        float tmp;
        for (int batch = 0; batch < M; ++batch)
            for (int j = 0; j < N; ++j)
            {
                tmp = 0;
                for (int i = 0; i < condense_k; ++i)
                {
                    tmp += mat_data[i + j * condense_k] * vec[mat_index[i + j * condense_k] * M + batch];
                }
                hostRef[j * M + batch] = tmp;
            }
    }

    template <typename dtype>
    void nmsparseCPURef(dtype *vec, dtype *mat_data, int *mat_index, dtype *hostRef, const int condense_k, const int N, const int K, const int M)
    {
        float tmp;
        for (int batch = 0; batch < M; ++batch)
            for (int j = 0; j < N; ++j)
            {
                tmp = 0;
                for (int i = 0; i < condense_k; ++i)
                {
                    tmp += mat_data[i + j * condense_k] * vec[mat_index[i + j * condense_k] + batch * K];
                }
                hostRef[j + batch * N] = tmp;
            }
    }

    template <typename dtype>
    bool nmsparseCheckResult(dtype *hostRef, dtype *gpuRef, const int M, const int N)
    {
        double epsilon = 1E-4;
        bool match = 1;
        for (int batch = 0; batch < M; ++batch)
            for (int i = 0; i < N; i++)
            {
                if (abs((hostRef[i + batch * N] - gpuRef[i + batch * N]) / hostRef[i + batch * N]) > epsilon)
                {
                    match = 0;
                    printf("Arrays do [NOT] match!\n");
                    printf("host %5.5f gpu %5.5f at current %d\n", hostRef[i + batch * N], gpuRef[i + batch * N], i + batch * N);
                    break;
                }
            }
        return match;
    }
}

#endif
