# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.specializer.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.testing import block_mask, test_latency


batch_size, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 64, 16, 32, 8, 4, 2
block = (8, 8)
sparsity = 0.95


def test_sparse_matmul_kernel(
    impl: str, sparse_type: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
):
    b_str = '_b' if biased else ''
    c_str = '_c' if compressed else ''
    t_A_str = 't' if trans_A else 'n'
    t_B_str = 't' if trans_B else 'n'
    kernel_name = f'{impl}_sparse_matmul_{sparse_type}{b_str}_{t_A_str}{t_B_str}{c_str}'

    torch.manual_seed(2022)
    A = torch.rand(size=(batch_size, M, K)).cuda()
    B = torch.rand(size=(batch_size, K, N)).cuda()
    bias = torch.rand(size=(batch_size, N)).cuda()
    if sparse_type == 'sdd':
        sparse_tensor = 'A'
        mask = block_mask((M, K), block=block, sparsity=sparsity).cuda()
        A *= mask
    elif sparse_type == 'dsd':
        sparse_tensor = 'B'
        mask = block_mask((K, N), block=block, sparsity=sparsity).cuda()
        B *= mask
    else:
        sparse_tensor = 'C'
        mask = block_mask((M, N), block=block, sparsity=sparsity).cuda()
    target_C = torch.bmm(A, B)
    if biased:
        target_C += bias.reshape((batch_size, 1, N))

    if trans_A:
        A = A.swapaxes(-1, -2).contiguous()
        if sparse_type == 'sdd':
            mask = mask.T.contiguous()
    if trans_B:
        B = B.swapaxes(-1, -2).contiguous()
        if sparse_type == 'dsd':
            mask = mask.T.contiguous()

    kernelClass: type[SparseMatMulKernel] = {
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
    kernel.set_shape(batch_size, M, K, N)
    kernel.compile(config, {sparse_tensor: mask})

    if sparse_type == 'dds':
        block_size_m, block_size_k, block_size_n = kernel.get_block_shape()
        c_mask = mask.reshape((M // block_size_m, block_size_m, N // block_size_n, block_size_n))
        c_mask = c_mask.swapaxes(1, 2).any(dim=-1).any(dim=-1)
        c_mask = c_mask.reshape((M // block_size_m, 1, N // block_size_n, 1))
        c_mask = c_mask.tile((1, block_size_m, 1, block_size_n)).reshape((M, N))
        target_C *= c_mask  # not C * mask because the dds known issue
    if compressed:
        if sparse_type == 'sdd':
            A = kernel.get_converter('A')(A)
        elif sparse_type == 'dsd':
            B = kernel.get_converter('B')(B)
        else:
            target_C = kernel.get_converter('C')(target_C)

    inputs = [A, B]
    if biased:
        inputs.append(bias)

    latency = test_latency(kernel, inputs, [target_C], num_warmups=10, num_iters=10)
    print(f'{kernel_name}: {latency} ms')


class TestSparseMatMulKernels(unittest.TestCase):

    def test_sparse_matmul_kernels(self):
        print('==================== Testing Sparse MatMul Kernels ====================')
        for impl in ['sparta', 'openai']:
            for sparse_type in ['sdd', 'dsd', 'dds']:
                for biased in [False, True]:
                    for trans_A in [False, True]:
                        for trans_B in [False, True]:
                            for compressed in [False, True]:
                                test_sparse_matmul_kernel(
                                    impl, sparse_type, biased, compressed, trans_A, trans_B
                                )


if __name__ == '__main__':
    unittest.main()
