# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
import copy

import torch
import pytest

from sparta.nn import SparseAttention
from sparta.testing import block_mask, check


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

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    sparse_attention = SparseAttention(mask=mask)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        target_out = sparse_attention.forward(query, key, value)
        target_out.backward(grad_out)

    target_grad_query = query.grad
    query.grad = None
    target_grad_key = key.grad
    key.grad = None
    target_grad_value = value.grad
    value.grad = None

    matmul_kernel_names = [
        'forward:C', 'backward:A', 'backward:B'
    ]
    matmul_config = {
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        }
        for kernel_name in matmul_kernel_names
    }
    softmax_kernel_names = ['forward:y', 'backward:x']
    softmax_config = {
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_H_VALUE': BM,
            'BLOCK_SIZE_W_VALUE': BN,
            'ROW_TILE_VALUE': RT,
        }
        for kernel_name in softmax_kernel_names
    }
    sparse_attention.build(
        params=dict(
            **{f'qk/{k}': v for k, v in matmul_config.items()},
            **{f'sm/{k}': v for k, v in softmax_config.items()},
            **{f'out/{k}': v for k, v in matmul_config.items()},
        ),
        sample_inputs=[query, key, value],
    )

    def attention_forward_backward(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad: torch.Tensor
    ):
        out = sparse_attention.forward(q, k, v)
        out.backward(grad)
        return out, q.grad, k.grad, v.grad

    check(
        func=attention_forward_backward,
        inputs=[query, key, value, grad_out],
        target_outputs=[target_out, target_grad_query, target_grad_key, target_grad_value]
    )
