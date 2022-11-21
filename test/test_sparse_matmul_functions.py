# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMul
from sparta.testing import block_mask, check


BATCH_SIZE, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


@pytest.mark.parametrize("sparse_type", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_function(
    sparse_type: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    torch.manual_seed(2022)
    A = torch.rand(size=(BATCH_SIZE, *A_shape), device='cuda')
    B = torch.rand(size=(BATCH_SIZE, *B_shape), device='cuda')
    bias = torch.rand(size=(BATCH_SIZE, N), device='cuda')

    if sparse_type == 'sdd':
        A_mask = block_mask(A_shape, block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask = {'A': A_mask}
        A *= A_mask
    elif sparse_type == 'dsd':
        B_mask = block_mask(B_shape, block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask = {'B': B_mask}
        B *= B_mask
    else:
        C_mask = block_mask((M, N), block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask = {'C': C_mask}

    A.requires_grad = True
    B.requires_grad = True
    bias.requires_grad = True

    grad_C = torch.rand(size=(BATCH_SIZE, M, N)).cuda()

    kernel_names = ['forward:C', 'backward:A', 'backward:B']
    sparse_ctx = SparseBatchMatMulCtx(sparse_type, trans_A, trans_B, biased, compressed)
    sparse_ctx.set_shape(BATCH_SIZE, M, K, N)
    sparse_ctx.set_masks(mask)
    sparse_ctx.build({
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        }
        for kernel_name in kernel_names
    })

    if sparse_type == 'dds':
        grad_C *= sparse_ctx.get_converter('forward:C', 'C').get_mask()

    if biased:
        target_C = sparse_ctx.dense_forward(A, B, bias)
    else:
        target_C = sparse_ctx.dense_forward(A, B)
    target_C.backward(grad_C)
    target_grad_A = A.grad
    A.grad = None
    target_grad_B = B.grad
    B.grad = None
    if biased:
        target_grad_bias = bias.grad
        bias.grad = None

    if sparse_type == 'sdd':
        target_grad_A *= sparse_ctx.get_converter('backward:A', 'A').get_mask()
    if sparse_type == 'dsd':
        target_grad_B *= sparse_ctx.get_converter('backward:B', 'B').get_mask()

    if compressed:
        if sparse_type == 'sdd':
            A = sparse_ctx.get_converter('forward:C', 'A')(A.detach())
            A.requires_grad = True
            target_grad_A = sparse_ctx.get_converter('backward:A', 'A')(target_grad_A)
        elif sparse_type == 'dsd':
            B = sparse_ctx.get_converter('forward:C', 'B')(B.detach())
            B.requires_grad = True
            target_grad_B = sparse_ctx.get_converter('backward:B', 'B')(target_grad_B)
        else:
            target_C = sparse_ctx.get_converter('forward:C', 'C')(target_C)
            grad_C = sparse_ctx.get_converter('forward:C', 'C')(grad_C)

    if biased:
        def matmul_forward_backward(A, B, bias, grad_C):
            C = SparseBatchMatMul.apply(sparse_ctx, A, B, bias)
            C.backward(grad_C)
            return C, A.grad, B.grad, bias.grad
        inputs = [A, B, bias, grad_C]
        target_outputs = [target_C, target_grad_A, target_grad_B, target_grad_bias]
    else:
        def matmul_forward_backward(A, B, grad_C):
            C = SparseBatchMatMul.apply(sparse_ctx, A, B)
            C.backward(grad_C)
            return C, A.grad, B.grad
        inputs = [A, B, grad_C]
        target_outputs = [target_C, target_grad_A, target_grad_B]

    check(matmul_forward_backward, inputs, target_outputs)
