# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.nn import SparseBatchMatMul


@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("backward", [False, True])
def test_sparse_matmul_search_space(
    mode: str,
    compressed: bool,
    trans_A: bool,
    trans_B: bool,
    backward: bool,
):
    mask = torch.zeros((64, 64), device='cuda')
    sparse_tensor = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    sparse_op = SparseBatchMatMul(
        **{f'{sparse_tensor}_mask': mask},
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
    )

    search_space = sparse_op.get_search_space(backward=backward)
    kernel_names = ['forward:C']
    if backward:
        kernel_names.append('backward:A')
        kernel_names.append('backward:B')
    impls = ['sparta', 'openai']
    sparta_params = [
        f'BLOCK_SIZE_{dim}_VALUE'
        for dim in ['M', 'K', 'N']
    ]
    openai_params = [
        f'BLOCK_SIZE_{dim}_VALUE'
        for dim in ['M', 'K', 'N']
    ]
    assert set(search_space.keys()) == set(kernel_names)
    for kernel_space in search_space.values():
        assert set(kernel_space.keys()) == set(impls)
        assert set(kernel_space['sparta'].keys()) == set(sparta_params)
        assert set(kernel_space['openai'].keys()) == set(openai_params)

    connections = sparse_op.get_connections(backward=backward)
    if compressed and backward:
        assert len(connections) == 2
        # TODO: check connections
    else:
        assert len(connections) == 0
