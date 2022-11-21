# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
import warnings

import torch
import pytest
import numpy as np

from sparta.nn import SparseSoftmax
from sparta.testing import block_mask, check


HEAD_NUM, DIMS = 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.sqrt(DIMS)


@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_softmax_operator(compressed: bool, batch_size: Optional[int]):
    torch.manual_seed(2022)
    mask = block_mask((HEAD_NUM, DIMS), block=BLOCK, sparsity=SPARSITY).cuda()
    shape = (HEAD_NUM, DIMS) if batch_size is None else (batch_size, HEAD_NUM, DIMS)
    sample_input = torch.rand(shape, dtype=torch.float32).cuda()
    sample_grad = torch.rand(shape, dtype=torch.float32).cuda()

    sample_input.requires_grad = True
    sparse_softmax = SparseSoftmax(mask, T, compressed, batch_size)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        target_output = sparse_softmax.forward(sample_input)
    target_output.backward(sample_grad)
    target_grad_input = sample_input.grad
    sample_input.grad = None

    kernel_names = ['forward:y', 'backward:x']
    sparse_softmax.build(
        params={
            kernel_name: {
                '_impl': 'sparta',
                'BLOCK_SIZE_H_VALUE': BH,
                'BLOCK_SIZE_W_VALUE': BW,
                'ROW_TILE_VALUE': RT,
            }
            for kernel_name in kernel_names
        },
        sample_inputs=[sample_input],
    )

    if compressed:
        converter = sparse_softmax._sparse_ctx.get_converter('forward:y', 'x')
        sample_input = converter.convert(sample_input.detach())
        target_output = converter.convert(target_output)
        sample_grad = converter.convert(sample_grad)
        target_grad_input = converter.convert(target_grad_input)
        sample_input.requires_grad = True

    def matmul_forward_backward(x: torch.Tensor, grad: torch.Tensor):
        y = sparse_softmax.forward(x)
        y.backward(grad)
        return y, x.grad

    check(
        func=matmul_forward_backward,
        inputs=[sample_input, sample_grad],
        target_outputs=[target_output, target_grad_input]
    )
