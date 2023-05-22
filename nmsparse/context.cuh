#ifndef _NMSPARSE_CONTEXT_H_
#define _NMSPARSE_CONTEXT_H_

#include <cuda_runtime.h>
#include <cctype>
#include <string>
#include <iostream>
#include <assert.h>

namespace nmsparse {

    // enumerate the type
    enum SparsePattern
    {
        ElementWise = 0,
        VectorWise4,
        VectorWise32,
        VectorWise64,
        BlockWise4x4,
        BlockWise64x64
    };

    // the context of the nmsparse
    struct nmsparseContext_t
    {

        // type of sparse pattern
        SparsePattern nmsparsePattern = ElementWise;

        // size of the sparse pattern N : M
        __uint32_t nmsparseN = 16;
        __uint32_t nmsparseM = 32;
        // sparsity of the sparse pattern N/M
        float sparsity = 0.5;
    };

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

    bool nmsparseSetContext(nmsparseContext_t *ctx, SparsePattern nmsparsePattern, unsigned int nmsparseM, float sparsity)
    {
        if (!ctx)
            return false;
        ctx->nmsparsePattern = nmsparsePattern;
        ctx->nmsparseM = nmsparseM;
        ctx->sparsity = sparsity;
        ctx->nmsparseN = nmsparseM * sparsity;
        return true;
    }

    template <typename dtype>
    bool checkCtxPattern(const nmsparseContext_t ctx)
    {
        if (std::is_same<dtype, float>::value == false && ctx.nmsparsePattern != BlockWise64x64 && ctx.nmsparsePattern != VectorWise64)
        {
            std::string _err_msg = "currently do not support ";
            _err_msg += std::is_same<dtype, float>::value ? "float" : "int8_t";
            _err_msg += " with ";
            if (ctx.nmsparsePattern == BlockWise64x64)
                _err_msg += "BlockWise64x64";
            else if (ctx.nmsparsePattern == VectorWise64)
                _err_msg += "VectorWise64";
            else
                _err_msg += "unknown pattern";
            throw std::runtime_error(_err_msg);
            return false;
        }
        // for BlockWise64x64 and VectorWise64, the dtype must be int8
        if (std::is_same<dtype, int8_t>::value == false && (ctx.nmsparsePattern == BlockWise64x64 || ctx.nmsparsePattern == VectorWise64))
        {
            std::string _err_msg = "currently only support int8_t with ";
            if (ctx.nmsparsePattern == BlockWise64x64)
                _err_msg += "BlockWise64x64";
            else if (ctx.nmsparsePattern == VectorWise64)
                _err_msg += "VectorWise64";
            else
                _err_msg += "unknown pattern";
            throw std::runtime_error(_err_msg);
            return false;
        }
        return true;
    }
}

#endif