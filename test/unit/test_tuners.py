# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import random
from typing import Any, Dict, List, Optional

import torch
import pytest

from sparta.nn import tune, SparseLinear, SparseBatchMatMul, SparseSoftmax, SparseAttention


def debug_tune(
    module: torch.nn.Module,
    sample_inputs: List[torch.Tensor],
    sample_grads: Optional[List[torch.Tensor]] = None,
):
    backward_weight = 0 if sample_grads is None else 0.5

    def debug_func(idx: int, params: Dict[Any, Any]):
        debug_func.count += 1
        return random.random()

    debug_func.count = 0
    try:
        tune(
            module=module,
            sample_inputs=sample_inputs,
            sample_grads=sample_grads,
            algo='grid',
            max_trials=sys.maxsize,
            backward_weight=backward_weight,
            verbose=False,
            debug_func=debug_func,
        )
    except AssertionError:
        pass

    return debug_func.count


@pytest.mark.parametrize("backward", [False, True])
def test_tune_sparse_linear(
    backward: bool,
    batch_size: int = 128,
    in_dims: int = 128,
    out_dims: int = 128,
):
    dense_linear = torch.nn.Linear(in_dims, out_dims, device='cuda')
    sample_input = torch.zeros((batch_size, in_dims), dtype=torch.float32, device='cuda')
    sample_grad = torch.zeros((batch_size, out_dims), dtype=torch.float32, device='cuda')
    mask = torch.ones((out_dims, in_dims), dtype=torch.bool, device='cuda')

    sparse_linear = SparseLinear(dense_linear, weight_mask=mask)
    if backward:
        count = debug_tune(sparse_linear, [sample_input], [sample_grad])
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 3
    else:
        count = debug_tune(sparse_linear, [sample_input], None)
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 1


@pytest.mark.parametrize("backward", [False, True])
def test_tune_sparse_matmul(
    backward: bool,
    mode: str = 'dds',
    batch_size: int = 4,
    M: int = 128,
    K: int = 128,
    N: int = 128,
    trans_A: bool = False,
    trans_B: bool = True,
    compressed: bool = True,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    C_shape = (M, N)

    torch.manual_seed(2022)
    A = torch.zeros(size=(batch_size, *A_shape), device='cuda')
    B = torch.zeros(size=(batch_size, *B_shape), device='cuda')
    grad_C = torch.zeros(size=(batch_size, *C_shape), device='cuda')

    matmul_args = {
        'transpose_A': trans_A,
        'transpose_B': trans_B,
        'compressed': compressed,
    }

    if mode == 'sdd':
        A_mask = torch.ones(A_shape, dtype=torch.bool, device='cuda')
        matmul_args['A_mask'] = A_mask
    elif mode == 'dsd':
        B_mask = torch.ones(B_shape, dtype=torch.bool, device='cuda')
        matmul_args['B_mask'] = B_mask
    else:
        C_mask = torch.ones(C_shape, dtype=torch.bool, device='cuda')
        matmul_args['C_mask'] = C_mask

    sparse_matmul = SparseBatchMatMul(**matmul_args)
    if backward:
        count = debug_tune(sparse_matmul, [A, B], [grad_C])
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 3
    else:
        count = debug_tune(sparse_matmul, [A, B], None)
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 1


@pytest.mark.parametrize("backward", [False, True])
def test_tune_sparse_softmax(
    backward: bool,
    head_num: int = 128,
    dims: int = 128,
    compressed: bool = True,
    batch_size: Optional[int] = None,
):
    torch.manual_seed(2022)
    mask = torch.ones((head_num, dims), dtype=torch.bool, device='cuda')
    shape = (head_num, dims) if batch_size is None else (batch_size, head_num, dims)
    sample_input = torch.zeros(shape, dtype=torch.float32, device='cuda')
    sample_grad = torch.zeros(shape, dtype=torch.float32, device='cuda')

    sparse_softmax = SparseSoftmax(mask=mask, temperature=dims, compressed=compressed)
    if backward:
        count = debug_tune(sparse_softmax, [sample_input], [sample_grad])
        assert count == (5 * 4 * 5) * 2
    else:
        count = debug_tune(sparse_softmax, [sample_input], None)
        assert count == (5 * 4 * 5) * 1


@pytest.mark.parametrize("backward", [False, True])
def test_tune_sparse_attention_operator(
    backward: bool,
    batch_size: int = 1,
    Ns: int = 128,
    Nt: int = 128,
    E: int = 128,
):
    torch.manual_seed(2022)
    mask = torch.ones((Nt, Ns), dtype=torch.bool, device='cuda')
    query = torch.rand(size=(batch_size, Nt, E)).cuda()
    key = torch.rand(size=(batch_size, Ns, E)).cuda()
    value = torch.rand(size=(batch_size, Ns, E)).cuda()
    grad_out = torch.rand(size=(batch_size, Nt, E)).cuda()

    sparse_attention = SparseAttention(mask=mask)
    if backward:
        count = debug_tune(sparse_attention, [query, key, value], [grad_out])
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 6 + (3 * 3 * 5) * 2
    else:
        count = debug_tune(sparse_attention, [query, key, value], None)
        assert count == (3 * 3 * 3 * 2 * 2 * 2 + 1) * 2 + (3 * 3 * 5) * 1
