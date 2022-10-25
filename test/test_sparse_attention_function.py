# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.specializer.funtional import SparseMultiHeadAttentionCtx, SparseMultiHeadAttention
from sparta.testing import block_mask, sparse_multi_head_attention_reference


BATCH_SIZE, Ns, Nt, E = 4, 512, 1024, 256
BM, BK, BN, TM, TK, TN, RT = 32, 32, 32, 4, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


def test_sparse_multi_head_attention_function():
    func_name = f'sparta_sparse_multi_head_attention'

    torch.manual_seed(2022)
    mask = block_mask((Nt, Ns), block=BLOCK, sparsity=SPARSITY).cuda()
    query = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()
    key = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    value = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    grad_out = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    target_out = sparse_multi_head_attention_reference(query, key, value, mask)

    matmul_kernel_names = [
        'forward:qk', 'forward:out', 'backward:v',
        'backward:sm', 'backward:q', 'backward:k'
    ]
    matmul_kernel_config = {
        'BLOCK_SIZE_M_VALUE': BM,
        'BLOCK_SIZE_K_VALUE': BK,
        'BLOCK_SIZE_N_VALUE': BN,
        'THREAD_SIZE_M_VALUE': TM,
        'THREAD_SIZE_K_VALUE': TK,
        'THREAD_SIZE_N_VALUE': TN,
    }
    softmax_kernel_names = ['forward:sm', 'backward:qk']
    softmax_kernel_config = {
        'BLOCK_SIZE_H_VALUE': BM,
        'BLOCK_SIZE_W_VALUE': BN,
        'ROW_TILE_VALUE': RT,
    }
    sparse_ctx = SparseMultiHeadAttentionCtx()
    sparse_ctx.set_shape(BATCH_SIZE, Ns, Nt, E)
    sparse_ctx.build(
        config=dict(
            _impl=';'.join([
                f'{kernel_name}=sparta'
                for kernel_name in matmul_kernel_names + softmax_kernel_names
            ]),
            **{
                f'{kernel_name};{param_name}': param_value
                for param_name, param_value in matmul_kernel_config.items()
                for kernel_name in matmul_kernel_names
            },
            **{
                f'{kernel_name};{param_name}': param_value
                for param_name, param_value in softmax_kernel_config.items()
                for kernel_name in softmax_kernel_names
            },
        ),
        mask={'qk': mask},
    )

    target_out.backward(grad_out)
    target_grad_q = query.grad
    query.grad = None
    target_grad_k = key.grad
    key.grad = None
    target_grad_v = value.grad
    value.grad = None

    print(func_name, end=': ')
    out = SparseMultiHeadAttention.apply(sparse_ctx, query, key, value)
    torch.testing.assert_close(out, target_out)
    print('forward pass;', end=' ')

    out.backward(grad_out)
    torch.testing.assert_close(query.grad, target_grad_q)
    torch.testing.assert_close(key.grad, target_grad_k)
    torch.testing.assert_close(value.grad, target_grad_v)
    print('backward pass.')


class TestSparseAttentionFunctions(unittest.TestCase):

    def test_sparse_multi_head_attention_functions(self):
        print('==================== Testing Sparse Softmax Functions ====================')
        test_sparse_multi_head_attention_function()


if __name__ == '__main__':
    unittest.main()
