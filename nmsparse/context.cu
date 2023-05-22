#include "nmsparse.h"

namespace nmsparse {

    bool nmsparseCreateContext(nmsparseContext_t *ctx){
        if (!ctx) return false;
        return true;
    }

    bool nmsparseDestroyContext(nmsparseContext_t *ctx){
        return true;
    }

    bool nmsparseSetContext(nmsparseContext_t *ctx, SparsePattern nmsparsePattern, unsigned int nmsparseN, unsigned int nmsparseM){
        if (!ctx) return false;
        ctx->nmsparsePattern = nmsparsePattern;
        ctx->nmsparseN = nmsparseN;
        ctx->nmsparseM = nmsparseM;
        ctx->sparsity = (float)nmsparseN / (float)nmsparseM;
        return true;
    }

    template <typename dtype>
    bool checkCtxPattern(nmsparseContext_t const *ctx)
    {
        if (std::is_same<dtype, float>::value == false && ctx.nmsparsePattern != BlockWise64x64 && ctx.nmsparsePattern != VectorWise64)
        {
            std::string _err_msg = "currently do not support " + std::to_string(ctx.nmsparsePattern) + " with dtype " + std::to_string(dtype);
            throw std::runtime_error(_err_msg);
            return false;
        }
        // for BlockWise64x64 and VectorWise64, the dtype must be int8
        if (std::is_same<dtype, int8_t>::value == false && (ctx.nmsparsePattern == BlockWise64x64 || ctx.nmsparsePattern == VectorWise64))
        {
            std::string _err_msg = "currently do not support " + std::to_string(ctx.nmsparsePattern) + " with dtype " + std::to_string(dtype);
            throw std::runtime_error(_err_msg);
            return false;
        }
        return true;
    }
}