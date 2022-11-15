# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

import torch
import pytest

from sparta.specializer.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.testing import block_mask, check


BATCH_SIZE, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 64, 16, 32, 8, 4, 2
BLOCK = (8, 8)
SPARSITY = 0.95


@pytest.mark.parametrize("impl", ['sparta', 'openai'])
@pytest.mark.parametrize("sparse_type", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_kernel(
    impl: str, sparse_type: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
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

    kernelClass: Type[SparseMatMulKernel] = {
        'sparta': SparTASparseMatMulKernel,
        'openai': OpenAISparseMatMulKernel,
    }[impl]
    kernel = kernelClass(
        sparse_type=sparse_type,
        biased=biased,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
    )

    config = {
        'sparta': {
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        },
        'openai': {},
    }[impl]
    kernel.set_shape(BATCH_SIZE, M, K, N)
    kernel.compile(config, mask)

    if compressed:
        if sparse_type == 'sdd':
            A = kernel.get_converter('A').convert(A)
        elif sparse_type == 'dsd':
            B = kernel.get_converter('B').convert(B)

    inputs = [A, B, bias] if biased else [A, B]
    check(kernel, inputs, kernel.reference(inputs))
