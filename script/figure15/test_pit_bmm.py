import torch
import time
import sparta
import sys
import os
from sparta.common.utils import convert_bcsr, verify_bcsr
from sparta.opset.bcsr_converter import BcsrConverter
from sparta.common.utils import convert_bcsr
import openai_bmm_cpp


def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


if __name__ == '__main__':
    torch.manual_seed(4321)
    batchsize = 1
    M = 4096
    K = 4096
    N = 4096
    block_h = 32
    block_w = 64
    RUNTIME = 1000
    os.makedirs('log', exist_ok=True)
    
    for sparsity_ratio in [0.0, 0.5, 0.9, 0.95, 0.99]:
        ########################################################################
        # measure the dense baseline time
        # A = torch.rand(batchsize, M, K).cuda()
        # B = torch.rand(K, N).cuda()
        # torch.cuda.synchronize()
        # t_start = time.time()
        # for _ in range(RUNTIME):
        #     C = torch.matmul(A, B)
        # torch.cuda.synchronize()
        # t_end = time.time()
        # print('Dense time baseline latency(ms):', (t_end-t_start)*1000/RUNTIME)
        
        #######################################################################
        # original openai sparse kernel
        A = torch.rand(batchsize, M, K).cuda()
        A_copy = A.clone().detach()
        B = torch.rand(batchsize, K, N).cuda()
        mask = torch.ones(M, K).cuda()
        block_wise_weight = torch.rand(M//block_h, K//block_w, dtype=torch.float32).cuda()
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        print('Block-wise sparsity ratio:', torch.sum(block_mask)/block_mask.numel())
        full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
        A *= full_mask
        ref_out = torch.einsum('bmk,bkn->bmn',A, B)
        converter_1 = BcsrConverter()
        row_ptr, col_idx, row_pos, vals = converter_1(full_mask, A, block_h, block_w)
        block_nnz = row_ptr[M//block_h]
        out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)

        # measure the latency of the original openai kernel
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(RUNTIME):
            out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
        torch.cuda.synchronize()
        t_end = time.time()
        print('Original openai bmm latency(ms):', (t_end-t_start)*1000/RUNTIME)
        with open(f'log/pit_{sparsity_ratio}_32_64.log', 'w') as f:
            f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))
        ###########################################################################
        # following is the condense matmul computation
        B = torch.rand(batchsize, K, N).cuda()
        A = torch.rand(batchsize, M, K).cuda()    
        new_block_h = block_h
        new_block_w = 1
        converter_2 = BcsrConverter(True)
        t_block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
        t_block_mask = (t_block_wise_weight > sparsity_ratio).to(torch.int32)
        print("Block-wise sparsity ratio:", torch.sum(t_block_mask)/t_block_mask.numel())
        t_full_mask = convert_to_full_mask(t_block_mask, (new_block_h, new_block_w))
        A *= t_full_mask
        ref_out = torch.einsum('bmk,bkn->bmn',A, B)
        # print(torch.squeeze(A).size())
        t_row_ptr, t_col_idx, t_row_pos, t_vals = converter_2(t_full_mask, torch.squeeze(A), new_block_h, new_block_w)
        t_block_nnz = t_row_ptr[M//new_block_h]
        condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)

        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(RUNTIME):
            condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
        torch.cuda.synchronize()
        t_end = time.time()
        print('Condense openai bmm latency(ms):', (t_end-t_start)*1000/RUNTIME)
        with open(f'log/pit_{sparsity_ratio}_32_1.log', 'w') as f:
            f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))
        #############################################################################
        # following is the condense module on the M dimension
        A = torch.rand(batchsize, M, K).cuda()
        B = torch.rand(batchsize, K, N).cuda()
        # A[:,:32] = 1
        new_block_h = 1
        new_block_w = 64
        block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
        block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
        # block_mask[:] = 1
        # block_mask[:32] = 1
        # block_mask[1:10] = 0
        print("Block-wise sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
        # import ipdb; ipdb.set_trace()
        full_mask = convert_to_full_mask(block_mask, (new_block_h, new_block_w))
        A *= full_mask
        ref_out = torch.einsum('bmk,bkn->bmn',A, B)
        m_csr_row, m_csr_col, m_csr_val = convert_bcsr(full_mask.t(), torch.squeeze(A).t(), new_block_w, new_block_h)
        m_csr_row, m_csr_col, m_csr_val = m_csr_row.cuda(), m_csr_col.cuda(), m_csr_val.cuda()
        m_block_nnz = m_csr_row[K//new_block_w].item()
        # print(m_block_nnz)
        # import ipdb; ipdb.set_trace()
        condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)

        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(RUNTIME):
            condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
        torch.cuda.synchronize()
        t_end = time.time()
        print('Condense openai bmm on dim m latency(ms):', (t_end-t_start)*1000/RUNTIME)
        with open(f'log/pit_{sparsity_ratio}_1_64.log', 'w') as f:
            f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))
        # print('##############################\n\n')    
    # for sparsity_ratio in [0.5, 0.9, 0.95, 0.99]:
    #     ###########################################################################
    #     # following is the condense matmul computation
    #     B = torch.rand(batchsize, K, N).cuda()
    #     A = torch.rand(batchsize, M, K).cuda()    
    #     new_block_h = block_h
    #     new_block_w = 1
    #     converter_2 = BcsrConverter(True)
    #     t_block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
    #     t_block_mask = (t_block_wise_weight > sparsity_ratio).to(torch.int32)
    #     print("Block-wise sparsity ratio:", torch.sum(t_block_mask)/t_block_mask.numel())
    #     t_full_mask = convert_to_full_mask(t_block_mask, (new_block_h, new_block_w))
    #     A *= t_full_mask
    #     ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    #     # print(torch.squeeze(A).size())
    #     t_row_ptr, t_col_idx, t_row_pos, t_vals = converter_2(t_full_mask, torch.squeeze(A), new_block_h, new_block_w)
    #     t_block_nnz = t_row_ptr[M//new_block_h]
    #     condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
    #     # assert torch.allclose(condense_out, ref_out, rtol=1e-08, atol=1e-03)
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(RUNTIME):
    #         condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
    #     torch.cuda.synchronize()
    #     t_end = time.time()
    #     with open(f'log/pit_{sparsity_ratio}_{new_block_h}_{new_block_w}.log', 'w') as f:
    #         f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))
        
    #     #############################################################################
    #     # following is the condense module on the M dimension
    #     A = torch.rand(batchsize, M, K).cuda()
    #     B = torch.rand(batchsize, K, N).cuda()
    #     # A[:,:32] = 1
    #     new_block_h = 1
    #     new_block_w = 64
    #     block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
    #     block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)

    #     print("Block-wise sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
    #     # import ipdb; ipdb.set_trace()
    #     full_mask = convert_to_full_mask(block_mask, (new_block_h, new_block_w))
    #     A *= full_mask
    #     ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    #     m_csr_row, m_csr_col, m_csr_val = convert_bcsr(full_mask.t(), torch.squeeze(A).t(), new_block_w, new_block_h)
    #     m_csr_row, m_csr_col, m_csr_val = m_csr_row.cuda(), m_csr_col.cuda(), m_csr_val.cuda()
    #     m_block_nnz = m_csr_row[K//new_block_w].item()
    #     # print(m_block_nnz)
    #     # import ipdb; ipdb.set_trace()
    #     condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
    #     flag = torch.allclose(condense_out_m, ref_out, rtol=1e-08, atol=1e-03)
    #     # if not flag:
    #     #     import ipdb; ipdb.set_trace()
    #     #     print("Correctness Failed!")
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(RUNTIME):
    #         condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
    #     torch.cuda.synchronize()
    #     t_end = time.time()
    #     # print('Condense openai bmm on dim m latency(ms):', (t_end-t_start)*1000/RUNTIME)
    #     with open(f'log/pit_{sparsity_ratio}_{new_block_h}_{new_block_w}.log', 'w') as f:
    #         f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))

    #     #########################################################################
    #     A = torch.rand(batchsize, M, K).cuda()
    #     B = torch.rand(batchsize, K, N).cuda()
    #     # A[:,:32] = 1
    #     new_block_h = 32
    #     new_block_w = 64
    #     block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
    #     block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
    #     # block_mask[:] = 1
    #     # block_mask[:32] = 1
    #     # block_mask[1:10] = 0
    #     print("Block-wise sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
    #     # import ipdb; ipdb.set_trace()
    #     full_mask = convert_to_full_mask(block_mask, (new_block_h, new_block_w))
    #     A *= full_mask
    #     ref_out = torch.einsum('bmk,bkn->bmn',A, B)
    #     m_csr_row, m_csr_col, m_csr_val = convert_bcsr(full_mask.t(), torch.squeeze(A).t(), new_block_w, new_block_h)
    #     m_csr_row, m_csr_col, m_csr_val = m_csr_row.cuda(), m_csr_col.cuda(), m_csr_val.cuda()
    #     m_block_nnz = m_csr_row[K//new_block_w].item()
    #     # print(m_block_nnz)
    #     # import ipdb; ipdb.set_trace()
    #     condense_out_m = openai_bmm_cpp.forward_condense_m_v2(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
    #     flag = torch.allclose(condense_out_m, ref_out, rtol=1e-08, atol=1e-03)
    #     # if not flag:
    #     #     import ipdb; ipdb.set_trace()
    #     #     print("Correctness Failed!")
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(RUNTIME):
    #         condense_out_m = openai_bmm_cpp.forward_condense_m_v2(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
    #     torch.cuda.synchronize()
    #     t_end = time.time()
    #     with open(f'log/pit_{sparsity_ratio}_{new_block_h}_{new_block_w}.log', 'w') as f:
    #         f.write('Condense openai bmm latency(ms): {}'.format((t_end-t_start)*1000/RUNTIME))
