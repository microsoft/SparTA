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
        return true;
    }
}