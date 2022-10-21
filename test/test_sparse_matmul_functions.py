# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMul
from sparta.testing import block_mask


batch_size, M, K, N = 4, 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
block = (8, 8)
sparsity = 0.95


def test_sparse_matmul_function(
    sparse_type: str, biased: bool, compressed: bool, trans_A: bool, trans_B: bool
):
    b_str = '_b' if biased else ''
    c_str = '_c' if compressed else ''
    t_A_str = 't' if trans_A else 'n'
    t_B_str = 't' if trans_B else 'n'
    func_name = f'sparse_matmul_{sparse_type}{b_str}_{t_A_str}{t_B_str}{c_str}'

    torch.manual_seed(2022)
    A_shape = (K, M) if trans_A else (M, K)
    A = torch.rand(size=(batch_size, *A_shape), device='cuda')
    B_shape = (N, K) if trans_B else (K, N)
    B = torch.rand(size=(batch_size, *B_shape), device='cuda')
    bias = torch.rand(size=(batch_size, N), device='cuda')
    if sparse_type == 'sdd':
        sparse_tensor = 'A'
        mask = block_mask(A_shape, block=block, sparsity=sparsity).cuda()
        A *= mask
    elif sparse_type == 'dsd':
        sparse_tensor = 'B'
        mask = block_mask(B_shape, block=block, sparsity=sparsity).cuda()
        B *= mask
    else:
        sparse_tensor = 'C'
        mask = block_mask((M, N), block=block, sparsity=sparsity).cuda()

    A.requires_grad = True
    B.requires_grad = True
    bias.requires_grad = True

    order_A = 'bki' if trans_A else 'bik'
    order_B = 'bjk' if trans_B else 'bkj'
    target_C = torch.einsum(f'{order_A},{order_B}->bij', A, B)
    if biased:
        target_C += bias.reshape((batch_size, 1, N))

    grad_C = torch.rand(size=target_C.shape, device='cuda')

    kernel_names = ['forward:C', 'backward:A', 'backward:B']
    sparse_ctx = SparseBatchMatMulCtx(sparse_type, trans_A, trans_B, biased, compressed)
    sparse_ctx.set_shape(batch_size, M, K, N)
    kernel_config = {
        'BLOCK_SIZE_M_VALUE': BM,
        'BLOCK_SIZE_K_VALUE': BK,
        'BLOCK_SIZE_N_VALUE': BN,
        'THREAD_SIZE_M_VALUE': TM,
        'THREAD_SIZE_K_VALUE': TK,
        'THREAD_SIZE_N_VALUE': TN,
    }
    sparse_ctx.build(
        config=dict(
            _impl=';'.join([f'{kernel_name}=sparta' for kernel_name in kernel_names]),
            **{
                f'{kernel_name};{param_name}': param_value
                for param_name, param_value in kernel_config.items()
                for kernel_name in kernel_names
            }
        ),
        mask={sparse_tensor: mask},
    )

    if sparse_type == 'dds':
        c_mask = sparse_ctx.get_converter('forward:C', 'C').get_mask()
        target_C *= c_mask  # not C * mask because the dds known issue
        grad_C *= c_mask

    target_C.backward(grad_C)
    target_grad_A = A.grad
    A.grad = None
    target_grad_B = B.grad
    B.grad = None
    if biased:
        target_grad_bias = bias.grad
        bias.grad = None

    if sparse_type == 'sdd':
        a_mask = sparse_ctx.get_converter('backward:A', 'A').get_mask()
        target_grad_A *= a_mask
    if sparse_type == 'dsd':
        b_mask = sparse_ctx.get_converter('backward:B', 'B').get_mask()
        target_grad_B *= b_mask

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

    print(func_name, end=': ')
    if biased:
        C = SparseBatchMatMul.apply(sparse_ctx, A, B, bias)
    else:
        C = SparseBatchMatMul.apply(sparse_ctx, A, B)
    torch.testing.assert_close(C, target_C)
    print('forward pass', end='; ')

    C.backward(grad_C)
    torch.testing.assert_close(A.grad, target_grad_A)
    torch.testing.assert_close(B.grad, target_grad_B)
    if biased:
        torch.testing.assert_close(bias.grad, target_grad_bias)
    print('backward pass')



class TestSparseMatMulKernels(unittest.TestCase):

    def test_sparse_matmul_functions(self):
        print('==================== Testing Sparse MatMul Functions ====================')
        for sparse_type in ['sdd', 'dsd', 'dds']:
            for biased in [False, True]:
                for trans_A in [False, True]:
                    for trans_B in [False, True]:
                        for compressed in [False, True]:
                            test_sparse_matmul_function(
                                sparse_type, biased, compressed, trans_A, trans_B
                            )


if __name__ == '__main__':
    unittest.main()
