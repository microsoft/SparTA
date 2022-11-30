# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple
import warnings

import torch
import pytest

from sparta.nn import SparseAttention
from sparta.testing import block_mask


def get_params():
    matmul_kernel_names = [
        'forward:C', 'backward:A', 'backward:B'
    ]
    matmul_config = {
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': 32,
            'BLOCK_SIZE_K_VALUE': 32,
            'BLOCK_SIZE_N_VALUE': 32,
            'THREAD_SIZE_M_VALUE': 4,
            'THREAD_SIZE_K_VALUE': 4,
            'THREAD_SIZE_N_VALUE': 4,
        }
        for kernel_name in matmul_kernel_names
    }
    softmax_kernel_names = ['forward:y', 'backward:x']
    softmax_config = {
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_H_VALUE': 32,
            'BLOCK_SIZE_W_VALUE': 32,
            'ROW_TILE_VALUE': 4,
        }
        for kernel_name in softmax_kernel_names
    }
    return dict(
        **{f'qk/{k}': v for k, v in matmul_config.items()},
        **{f'sm/{k}': v for k, v in softmax_config.items()},
        **{f'out/{k}': v for k, v in matmul_config.items()},
    )


def test_sparse_attention_operator(
    batch: int = 4,
    Ns: int = 512,
    Nt: int = 1024,
    E: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.95,
):
    torch.manual_seed(2022)
    mask = block_mask((Nt, Ns), block=granularity, sparsity=sparsity).cuda()
    query = torch.rand(size=(batch, Nt, E)).cuda()
    key = torch.rand(size=(batch, Ns, E)).cuda()
    value = torch.rand(size=(batch, Ns, E)).cuda()
    grad_out = torch.rand(size=(batch, Nt, E)).cuda()

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

    sparse_attention.build(
        params=get_params(),
        sample_inputs=[query, key, value],
    )

    out = sparse_attention.forward(query, key, value)
    out.backward(grad_out)

    torch.testing.assert_close(out, target_out)
    torch.testing.assert_close(query.grad, target_grad_query)
    torch.testing.assert_close(key.grad, target_grad_key)
    torch.testing.assert_close(value.grad, target_grad_value)
