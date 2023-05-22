#include "nmsparse.h"

namespace nmsparse{
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val){

        // check if the input is valid
        checkCtxPattern<dtype>(ctx);
        // TODO(lei): implement the conversion from dense to sparse pattern.

        return true;
    }


    template <typename dtype>
    void nmsparseCPURef(float *vec, float *mat_data, int *mat_index, float *hostRef, const int condense_k, const int N, int vecNum, const int M)
    {
        float tmp;
        for (int batch = 0; batch < M; ++batch)
            for (int j = 0; j < N; ++j)
            {
                tmp = 0;
                for (int i = 0; i < condense_k; ++i)
                {
                    tmp += mat_data[i + j * condense_k] * vec[mat_index[i + j * condense_k] + batch * vecNum];
                }
                hostRef[j + batch * N] = tmp;
            }
    }
}