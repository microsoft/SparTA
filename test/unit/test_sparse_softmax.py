# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Optional

import torch
import pytest
import numpy as np

from sparta.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.operators import SparsityAttr, SparseSoftmax, SparseBatchSoftmax
from sparta.testing import block_mask, sparse_softmax_forward_reference


def prepare_data(
    batch: Optional[int] = 4,
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    requires_grad: bool = False,
    mask: Optional[torch.Tensor] = None,
    random_seed: int = 2022,
):
    torch.manual_seed(random_seed)
    data: Dict[str, torch.Tensor] = {}
    shape = (H, W) if batch is None else (batch, H, W)
    data['input_x'] = torch.rand(shape, device='cuda')
    temperature = np.sqrt(W)

    if mask is None:
        mask = block_mask((H, W), block=granularity, sparsity=sparsity, device='cuda')

    data['grad_y'] = torch.rand(shape, device='cuda')
    data['input_x'].requires_grad = True

    data['target_y'] = sparse_softmax_forward_reference(data['input_x'], mask, temperature)

    data['target_y'].backward(data['grad_y'])
    data['target_grad_x'] = data['input_x'].grad

    if requires_grad:
        data['input_x'].grad = None
    else:
        data['input_x'] = data['input_x'].detach()
        data['target_y'] = data['target_y'].detach()

    return data, mask


def check_results(data: Dict[str, torch.Tensor]):
    for name, val in data.items():
        if name.startswith('target_'):
            torch.testing.assert_close(val, data[name.replace('target', 'output')], msg=name)


def get_params():
    return {
        '_impl': 'sparta',
        'BLOCK_SIZE_H_VALUE': 32,
        'BLOCK_SIZE_W_VALUE': 32,
        'ROW_TILE_VALUE': 4,
    }


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("batch", [None, 4])
def test_sparse_softmax_kernels(
    compressed: bool,
    batch: Optional[int],
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0,
):
    data, mask = prepare_data(batch, H, W, granularity, sparsity, False)

    batched = batch is not None
    forward_kernel = SparTASparseSoftmaxForwardKernel(compressed, batched)
    backward_kernel = SparTASparseSoftmaxBackwardKernel(compressed, batched)

    attr = SparsityAttr(True, False)
    attr.set_mask(mask)

    shape = (batch, H, W)
    forward_kernel.set_parameter('MAX_W_VALUE', W)
    backward_kernel.set_parameter('MAX_W_VALUE', W)
    forward_kernel.compile(get_params(), shape, attr)
    backward_kernel.compile(get_params(), shape, attr)

    attr.set_block_size(
        forward_kernel.get_parameter('BLOCK_SIZE_H_VALUE'),
        forward_kernel.get_parameter('BLOCK_SIZE_W_VALUE'),
    )
    temperature = np.float32(1 / np.sqrt(W))

    if compressed:
        for name in data:
            data[name] = attr.indexes.convert(data[name].detach())

    data['output_y'] = forward_kernel(data['input_x'], mask, temperature)
    data['output_grad_x'] = backward_kernel(data['grad_y'], data['target_y'], mask, temperature)
    check_results(data)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("batch", [None, 4])
def test_sparse_softmax_operator(
    compressed: bool,
    batch: Optional[int],
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(batch, H, W, granularity, sparsity, True)

    if batch is None:
        sparse_softmax = SparseSoftmax(compressed, np.sqrt(W))
    else:
        sparse_softmax = SparseBatchSoftmax(compressed, np.sqrt(W))

    sparse_attr = sparse_softmax.get_sparse_attr()
    sparse_attr.set_mask(mask)

    kernel_names = ['forward', 'backward']
    sparse_softmax.build(
        config={kernel_name: get_params() for kernel_name in kernel_names},
        sample_inputs=[data['input_x']]
    )

    def run_test():
        nonlocal sparse_softmax, data, mask
        if compressed:
            for name in data:
                data[name] = sparse_attr.indexes.convert(data[name].detach())
            data['input_x'].requires_grad = True
        data['output_y'] = sparse_softmax(data['input_x'])
        data['output_y'].backward(data['grad_y'])
        data['output_grad_x'] = data['input_x'].grad
        check_results(data)

    run_test()

    # Dynamic mask
    data, mask = prepare_data(batch, H, W, granularity, sparsity, True, random_seed=2023)
    sparse_softmax.set_mask(mask)
    run_test()
