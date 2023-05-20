#include "nmsparse.h"

namespace nmsparse{
    template <typename dtype>
    bool nmsparseCreateSparse(nmsparseContext_t ctx, int k, int n,
                              dtype *mat_in_dense, int *output_sparse_idx, dtype *output_sparse_val){

        // check if the input is valid
        // for all pattern except BlockWise64x64 and VectorWise64, the dtype must be float
        if (std::is_same<dtype, float>::value == false && ctx.nmsparsePattern != BlockWise64x64 && ctx.nmsparsePattern != VectorWise64){
            std::string _err_msg = "currently do not support " + std::to_string(ctx.nmsparsePattern) 
                                    + " with dtype " + std::to_string(dtype);
            throw std::runtime_error(_err_msg);
            return false; 
        }
        // for BlockWise64x64 and VectorWise64, the dtype must be int8
        if (std::is_same<dtype, int8_t>::value == false && (ctx.nmsparsePattern == BlockWise64x64 || ctx.nmsparsePattern == VectorWise64)){
            std::string _err_msg = "currently do not support " + std::to_string(ctx.nmsparsePattern) 
                                    + " with dtype " + std::to_string(dtype);
            throw std::runtime_error(_err_msg);
            return false; 
        }

        // TODO(lei): implement the conversion from dense to sparse pattern.
        return true;
    }
}