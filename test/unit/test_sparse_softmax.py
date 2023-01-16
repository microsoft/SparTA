# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Optional

import torch
import pytest
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmaxFunc
from sparta.nn import SparseSoftmax
from sparta.testing import block_mask, sparse_softmax_forward_reference


def prepare_data(
    batch: Optional[int] = 4,
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    requires_grad: bool = False,
    random_seed: int = 2022,
):
    torch.manual_seed(random_seed)
    data: Dict[str, torch.Tensor] = {}
    shape = (H, W) if batch is None else (batch, H, W)
    data['input_x'] = torch.rand(shape, device='cuda')
    temperature = np.sqrt(W)

    mask = block_mask((H, W), block=granularity, sparsity=sparsity, device='cuda')

    if requires_grad:
        data['grad_y'] = torch.rand(shape, device='cuda')
        data['input_x'].requires_grad = True

    data['target_y'] = sparse_softmax_forward_reference(data['input_x'], mask, temperature)

    if requires_grad:
        data['target_y'].backward(data['grad_y'])
        data['target_grad_x'] = data['input_x'].grad
        data['input_x'].grad = None

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
def test_sparse_softmax_kernels(
    compressed: bool,
    batch: Optional[int] = 4,
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(batch, H, W, granularity, sparsity, requires_grad=True)

    forward_kernel = SparTASparseSoftmaxForwardKernel(compressed=compressed)
    backward_kernel = SparTASparseSoftmaxBackwardKernel(compressed=compressed)

    if compressed:
        sparse_port = backward_kernel.ports['y']
        sparse_port.connect(forward_kernel, 'x')
        sparse_port.connect(forward_kernel, 'y')
        sparse_port.set_mask(mask)
    else:
        forward_kernel.ports['y'].set_mask(mask)
        backward_kernel.ports['y'].set_mask(mask)
    forward_kernel.set_shape(batch, H, W)
    forward_kernel.compile(get_params())
    backward_kernel.set_shape(batch, H, W)
    backward_kernel.compile(get_params())

    temperature = np.float32(1 / np.sqrt(W))

    forward_inputs = [data['input_x'], temperature]
    forward_kernel.test(forward_inputs, num_warmups=0, num_iters=1, cuda=False)
    backward_inputs = [data['grad_y'], data['target_y'], temperature]
    backward_kernel.test(backward_inputs, num_warmups=0, num_iters=1, cuda=False)


@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_softmax_function(
    compressed: bool,
    batch: Optional[int] = 4,
    H: int = 128,
    W: int = 256,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(batch, H, W, granularity, sparsity, requires_grad=True)

    sparse_ctx = SparseBatchSoftmaxCtx(compressed, np.sqrt(W))
    kernel_names = sparse_ctx.get_kernel_placeholders(backward=True).keys()
    sparse_ctx.select_impls({
        kernel_name: 'sparta'
        for kernel_name in kernel_names
    })
    for port_name, ports in sparse_ctx.sparse_ports.items():
        for port in ports:
            port.set_mask(mask)
    sparse_ctx.set_shape(batch, H, W)
    sparse_ctx.build({
        kernel_name: get_params()
        for kernel_name in kernel_names
    })

    if compressed:
        indexes = sparse_ctx.sparse_ports['y'][0].indexes
        for name in data:
            data[name] = indexes.convert(data[name].detach())
        data['input_x'].requires_grad = True

    data['output_y'] = SparseBatchSoftmaxFunc.apply(sparse_ctx, data['input_x'])
    data['output_y'].backward(data['grad_y'])
    data['output_grad_x'] = data['input_x'].grad

    check_results(data)


# @pytest.mark.parametrize("batch", [None, 4])
# @pytest.mark.parametrize("compressed", [False, True])
# def test_sparse_softmax_operator(
#     compressed: bool,
#     batch: Optional[int],
#     H: int = 128,
#     W: int = 256,
#     granularity: Tuple[int, int] = (8, 8),
#     sparsity: float = 0.9,
# ):
#     data, mask = prepare_data(batch, H, W, granularity, sparsity, requires_grad=True)

#     sparse_softmax = SparseSoftmax(mask, np.sqrt(W), compressed)
#     sparse_softmax.build(
#         config={
#             kernel_name: get_params()
#             for kernel_name in sparse_softmax.get_kernel_placeholders(backward=True)
#         },
#         sample_inputs=[data['input_x']],
#     )

#     if compressed:
#         converter = sparse_softmax.get_converter('forward:y', 'x')
#         for name in data:
#             data[name] = converter.convert(data[name].detach())
#         data['input_x'].requires_grad = True

#     data['output_y'] = sparse_softmax.forward(data['input_x'])
#     data['output_y'].backward(data['grad_y'])
#     data['output_grad_x'] = data['input_x'].grad

#     check_results(data)
