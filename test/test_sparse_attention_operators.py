# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.nn import SparseAttention
from sparta.testing import block_mask, sparse_multi_head_attention_reference


BATCH_SIZE, Ns, Nt, E = 4, 512, 1024, 256
BM, BK, BN, TM, TK, TN, RT = 32, 32, 32, 4, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


def test_sparse_attention_operator():
    print(f'sparse_attention:', end=' ')

    torch.manual_seed(2022)
    mask = block_mask((Nt, Ns), block=BLOCK, sparsity=SPARSITY).cuda()
    query = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()
    key = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    value = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    grad_out = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()

    sparse_attention = SparseAttention(mask=mask)

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
    sparse_attention.build(
        params=dict(
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
        sample_inputs=[query, key, value]
    )

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    target_out = sparse_multi_head_attention_reference(query, key, value, mask)
    out = sparse_attention.forward(query, key, value)
    torch.testing.assert_close(out, target_out)
    print('forward pass;', end=' ')

    target_out.backward(grad_out)
    target_grad_q = query.grad
    query.grad = None
    target_grad_k = key.grad
    key.grad = None
    target_grad_v = value.grad
    value.grad = None
    # sample_input.grad *= 0
    out.backward(grad_out)
    torch.testing.assert_close(query.grad, target_grad_q)
    torch.testing.assert_close(key.grad, target_grad_k)
    torch.testing.assert_close(value.grad, target_grad_v)
    print('backward pass.')


class TestSparseAttentionOperators(unittest.TestCase):

    def test_sparse_attention_operators(self):
        print('==================== Testing Sparse Attention Operators ====================')
        test_sparse_attention_operator()


if __name__ == '__main__':
    unittest.main()
