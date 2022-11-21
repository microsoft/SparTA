# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

import torch
import pytest

from sparta.specializer.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.testing import block_mask


BATCH_SIZE, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 64, 16, 32, 8, 4, 2
BLOCK = (8, 8)
SPARSITY = 0.95


@pytest.mark.parametrize("impl", ['sparta', 'openai'])
@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_kernel(
    impl: str, mode: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)

    torch.manual_seed(2022)
    A = torch.rand(size=(BATCH_SIZE, *A_shape), device='cuda')
    B = torch.rand(size=(BATCH_SIZE, *B_shape), device='cuda')
    bias = torch.rand(size=(BATCH_SIZE, N), device='cuda')

    if mode == 'sdd':
        A_mask = block_mask(A_shape, block=BLOCK, sparsity=SPARSITY, device='cuda')
        mask = {'A': A_mask}
        A *= A_mask
    elif mode == 'dsd':
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
        mode=mode,
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
    kernel.set_masks(mask)
    kernel.compile(config)

    inputs = [A, B, bias] if biased else [A, B]
    target_outputs = [kernel.reference(*inputs)]
    kernel.test(inputs, target_outputs, num_warmups=0, num_iters=1, cuda=False)
