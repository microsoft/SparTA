# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings

import torch
import pytest

from sparta.nn import SparseBatchMatMul
from sparta.testing import block_mask, check


BATCH_SIZE, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_function(mode: str, trans_A: bool, trans_B: bool, compressed: bool):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    torch.manual_seed(2022)
    A = torch.rand(size=(BATCH_SIZE, *A_shape), device='cuda')
    B = torch.rand(size=(BATCH_SIZE, *B_shape), device='cuda')

    if mode == 'sdd':
        A_mask = block_mask(A_shape, block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask_dict = {'A_mask': A_mask}
        A *= A_mask
    elif mode == 'dsd':
        B_mask = block_mask(B_shape, block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask_dict = {'B_mask': B_mask}
        B *= B_mask
    else:
        C_mask = block_mask((M, N), block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask_dict = {'C_mask': C_mask}

    A.requires_grad = True
    B.requires_grad = True

    grad_C = torch.rand(size=(BATCH_SIZE, M, N)).cuda()

    sparse_matmul = SparseBatchMatMul(**mask_dict, transpose_A=trans_A, transpose_B=trans_B, compressed=compressed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        target_C = sparse_matmul.forward(A, B)

    kernel_names = ['forward:C', 'backward:A', 'backward:B']
    sparse_matmul.build(
        params={
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
        },
        sample_inputs=[A, B],
    )

    if mode == 'dds':
        target_C *= sparse_matmul._sparse_ctx.get_converter('forward:C', 'C').get_mask()
        grad_C *= sparse_matmul._sparse_ctx.get_converter('forward:C', 'C').get_mask()

    target_C.backward(grad_C)
    target_grad_A = A.grad
    A.grad = None
    target_grad_B = B.grad
    B.grad = None

    if mode == 'sdd':
        target_grad_A *= sparse_matmul._sparse_ctx.get_converter('backward:A', 'A').get_mask()
    if mode == 'dsd':
        target_grad_B *= sparse_matmul._sparse_ctx.get_converter('backward:B', 'B').get_mask()

    if compressed:
        if mode == 'sdd':
            A = sparse_matmul._sparse_ctx.get_converter('forward:C', 'A')(A.detach())
            A.requires_grad = True
            target_grad_A = sparse_matmul._sparse_ctx.get_converter('backward:A', 'A')(target_grad_A)
        elif mode == 'dsd':
            B = sparse_matmul._sparse_ctx.get_converter('forward:C', 'B')(B.detach())
            B.requires_grad = True
            target_grad_B = sparse_matmul._sparse_ctx.get_converter('backward:B', 'B')(target_grad_B)
        else:
            target_C = sparse_matmul._sparse_ctx.get_converter('forward:C', 'C')(target_C)
            grad_C = sparse_matmul._sparse_ctx.get_converter('forward:C', 'C')(grad_C)

    def matmul_forward_backward(A: torch.Tensor, B: torch.Tensor, grad_C: torch.Tensor):
        C = sparse_matmul.forward(A, B)
        C.backward(grad_C)
        return C, A.grad, B.grad
    inputs = [A, B, grad_C]
    target_outputs = [target_C, target_grad_A, target_grad_B]

    check(matmul_forward_backward, inputs, target_outputs)
