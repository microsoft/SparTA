# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Optional

import torch
import pytest

from sparta.nn import SparseLinear, SparseMatMul, SparseSoftmax, SparseAttention, tune
from sparta.testing import block_mask, profile


TUNER_ARGS = {
    'algo': 'rand',
    'max_trials': 1,
    'backward_weight': 0.5,
}
PROFILE_ARGS = {
    'num_warmups': 1000,
    'num_iters': 1000,
    'cuda': False,
}


def profile_module(
    module: torch.nn.Module,
    sample_inputs: List[torch.Tensor],
    sample_grads: List[torch.Tensor]
):
    for x in sample_inputs:
        x.requires_grad = True
    forward_latency = profile(module.forward, sample_inputs, **PROFILE_ARGS)

    def forward_backward():
        output = module.forward(*sample_inputs)
        output.backward(sample_grads[0])

    total_latency = profile(forward_backward, [], **PROFILE_ARGS)
    backward_latency = total_latency - forward_latency
    print(f'Forward latency: {forward_latency} ms; Backward latency: {backward_latency} ms')


def tune_sparse_linear(
    batch_size: int = 1024, in_dims: int = 1024, out_dims: int = 1024,
    block_size: Tuple[int, int] = (8, 8), sparsity: float = 0.95
):
    torch.manual_seed(2022)

    dense_linear = torch.nn.Linear(in_dims, out_dims).cuda()
    sample_input = torch.rand((batch_size, in_dims), dtype=torch.float32).cuda()
    sample_grad = torch.rand((batch_size, out_dims), dtype=torch.float32).cuda()
    mask = block_mask((out_dims, in_dims), block=block_size, sparsity=sparsity).cuda()

    sparse_linear = SparseLinear(dense_linear, weight_mask=mask)
    tune(sparse_linear, [sample_input], [sample_grad], **TUNER_ARGS)

    profile_module(sparse_linear, [sample_input], [sample_grad])


def tune_sparse_matmul(
    mode: str = 'dds',
    batch_size: int = 4, M: int = 1024, K: int = 1024, N: int = 1024,
    trans_A: bool = False, trans_B: bool = True, compressed: bool = True,
    block_size: Tuple[int, int] = (8, 8), sparsity: float = 0.95
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    C_shape = (M, N)

    torch.manual_seed(2022)
    A = torch.rand(size=(batch_size, *A_shape), device='cuda')
    B = torch.rand(size=(batch_size, *B_shape), device='cuda')
    grad_C = torch.rand(size=(batch_size, *C_shape), device='cuda')

    matmul_args = {
        'transpose_A': trans_A,
        'transpose_B': trans_B,
        'compressed': compressed,
    }

    if mode == 'sdd':
        A_mask = block_mask(A_shape, block=block_size, sparsity=sparsity, device='cuda')
        matmul_args['A_mask'] = A_mask
    elif mode == 'dsd':
        B_mask = block_mask(B_shape, block=block_size, sparsity=sparsity, device='cuda')
        matmul_args['B_mask'] = B_mask
    else:
        C_mask = block_mask(C_shape, block=block_size, sparsity=sparsity, device='cuda')
        matmul_args['C_mask'] = C_mask

    sparse_matmul = SparseMatMul(**matmul_args)
    tune(sparse_matmul, [A, B], [grad_C], **TUNER_ARGS)

    if compressed:
        if mode == 'sdd':
            A = sparse_matmul._sparse_ctx.get_converter('forward:C', 'A').convert(A)
        elif mode == 'dsd':
            B = sparse_matmul._sparse_ctx.get_converter('forward:C', 'B').convert(B)
        else:
            grad_C = sparse_matmul._sparse_ctx.get_converter('forward:C', 'C').convert(grad_C)
    profile_module(sparse_matmul, [A, B], [grad_C])


def tune_sparse_softmax(
    head_num: int = 1024, dims: int = 1024,
    compressed: bool = True, batch_size: Optional[int] = None,
    block_size: Tuple[int, int] = (8, 8), sparsity: float = 0.95
):
    torch.manual_seed(2022)
    mask = block_mask((head_num, dims), block=block_size, sparsity=sparsity).cuda()
    shape = (head_num, dims) if batch_size is None else (batch_size, head_num, dims)
    sample_input = torch.rand(shape, dtype=torch.float32).cuda()
    sample_grad = torch.rand(shape, dtype=torch.float32).cuda()

    sparse_softmax = SparseSoftmax(mask=mask, temperature=dims, compressed=compressed)
    tune(sparse_softmax, [sample_input], [sample_grad], **TUNER_ARGS)

    if compressed:
        converter = sparse_softmax._sparse_ctx.get_converter('forward:y', 'x')
        sample_input = converter.convert(sample_input)
        sample_grad = converter.convert(sample_grad)
    profile_module(sparse_softmax, [sample_input], [sample_grad])


def tune_sparse_attention_operator(
    batch_size: int = 1, Ns: int = 4096, Nt: int = 3072, E: int = 768,
    block_size: Tuple[int, int] = (8, 8), sparsity: float = 0.95
):
    torch.manual_seed(2022)
    mask = block_mask((Nt, Ns), block=block_size, sparsity=sparsity).cuda()
    query = torch.rand(size=(batch_size, Nt, E)).cuda()
    key = torch.rand(size=(batch_size, Ns, E)).cuda()
    value = torch.rand(size=(batch_size, Ns, E)).cuda()
    grad_out = torch.rand(size=(batch_size, Nt, E)).cuda()

    sparse_attention = SparseAttention(mask=mask)
    tune(sparse_attention, [query, key, value], [grad_out], **TUNER_ARGS)

    profile_module(sparse_attention, [query, key, value], [grad_out])


if __name__ == '__main__':
    # tune_sparse_linear()
    # tune_sparse_matmul()
    # tune_sparse_softmax()
    tune_sparse_attention_operator()
