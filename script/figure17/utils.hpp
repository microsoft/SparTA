#include "iostream"
#include "sstream"
#include "time.h"
#include "memory"
#include "vector"
using namespace std;

void init(float * ptr, size_t length, float sparsity)
{
    for (int i = 0; i < length; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            // ptr[i] = 1;
        }
    }
}

void init_blockwise(float* value, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    memset(value, 0, sizeof(float)*M*N);
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (pro >= sparsity)
            {
                int pos;
                for (int b_i=0; b_i<block_h; b_i++){
                    for(int b_j=0; b_j<block_w; b_j++){
                        pos = (i * block_h + b_i)*N + j* block_w + b_j;
                        value[pos] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        // value[pos] = 1;
                    }
                }
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);
}

void init_mask_blockwise(int * mask, float * value, size_t M, size_t N, int block_h, int block_w, float sparsity)
{
    memset(mask, 0, sizeof(int)*M*N);
    memset(value, 0, sizeof(float)*M*N);
    int m_block_n = M / block_h;
    int n_block_n = N / block_w;
    int block_nnz = 0;
    for (int i = 0; i < m_block_n; i++)
    {
        for(int j=0; j < n_block_n; j++){
            float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (pro >= sparsity)
            {
                int pos;
                for (int b_i=0; b_i<block_h; b_i++){
                    for(int b_j=0; b_j<block_w; b_j++){
                        pos = (i * block_h + b_i)*N + j* block_w + b_j;
                        mask[pos]=1;
                        value[pos] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        // value[pos] = 1;
                    }
                }
                block_nnz++;
            }
        }
        
    }
    printf("random %d blocks in init_mask_blockwise\n", block_nnz);

}