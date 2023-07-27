import torch
import time
import sparta
import sys
from sparta.common.utils import convert_bcsr, verify_bcsr
import openai_bmm_cpp 

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


if __name__ == '__main__':
    torch.manual_seed(4321)
    batchsize = 1
    sparsity_ratio = float(sys.argv[1])
    M = int(sys.argv[2])
    K = int(sys.argv[3])
    N = int(sys.argv[4])
    block_h = int(sys.argv[5])
    block_w = int(sys.argv[6])
    tile_size_h = 32
    tile_size_w = 64
    RUNTIME = 1000
    print(sparsity_ratio, M, K, N, block_h, block_w)
    A = torch.rand(batchsize, M, K).cuda()
    B = torch.rand(batchsize, K, N).cuda()
    block_wise_weight = torch.rand(M//block_h, K//block_w, dtype=torch.float32).cuda()
    block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
    print('Block-wise sparsity ratio:', torch.sum(block_mask)/block_mask.numel())
    full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
    A *= full_mask
    ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    row_ptr, col_idx, vals = convert_bcsr(full_mask, A, max(block_h, tile_size_h), max(block_w, tile_size_w))
    row_ptr, col_idx, vals = row_ptr.cuda(), col_idx.cuda(), vals.cuda()
    block_nnz = row_ptr[M//tile_size_h]
    out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    # if not torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03):
    #     import ipdb; ipdb.set_trace()
    # assert torch.allclose(out, ref_out, rtol=1e-08, atol=1e-03)
    # measure the latency of the original openai kernel
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(RUNTIME):
        out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
    torch.cuda.synchronize()
    t_end = time.time()
    print('Original openai bmm latency(ms):', (t_end-t_start)*1000/RUNTIME)
