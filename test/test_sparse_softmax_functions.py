# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest
import numpy as np

from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmaxFunc
from sparta.testing import block_mask, check, sparse_softmax_forward_reference


BATCH_SIZE, H, W = 4, 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.sqrt(W)


@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_softmax_function(compressed: bool):
    torch.manual_seed(2022)
    mask = block_mask((H, W), block=BLOCK, sparsity=SPARSITY, device='cuda')
    x = torch.rand(size=(BATCH_SIZE, H, W), device='cuda')
    grad_y = torch.rand(size=(BATCH_SIZE, H, W), device='cuda')

    x.requires_grad = True

    kernel_names = ['forward:y', 'backward:x']
    sparse_ctx = SparseBatchSoftmaxCtx(compressed, T)
    sparse_ctx.set_shape(BATCH_SIZE, H, W)
    sparse_ctx.set_masks({'x': mask})
    sparse_ctx.build({
        kernel_name: {
            '_impl': 'sparta',
            'BLOCK_SIZE_H_VALUE': BH,
            'BLOCK_SIZE_W_VALUE': BW,
            'ROW_TILE_VALUE': RT,
        }
        for kernel_name in kernel_names
    })

    target_y = sparse_ctx.dense_forward(x)
    target_y.backward(grad_y)
    target_grad_x = x.grad
    x.grad = None

    if compressed:
        x = sparse_ctx.get_converter('forward:y', 'x').convert(x.detach())
        x.requires_grad = True
        target_y = sparse_ctx.get_converter('forward:y', 'x').convert(target_y)
        grad_y = sparse_ctx.get_converter('backward:x', 'x').convert(grad_y)
        target_grad_x = sparse_ctx.get_converter('backward:x', 'x').convert(target_grad_x)

    def softmax_forward_backward(x, grad_y):
        y = SparseBatchSoftmaxFunc.apply(sparse_ctx, x)
        y.backward(grad_y)
        return y, x.grad

    check(softmax_forward_backward, [x, grad_y], [target_y, target_grad_x])
