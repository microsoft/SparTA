# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.testing import block_mask


BATCH_SIZE, H, W = 4, 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.float32(np.sqrt(W))


@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_softmax_forward_kernel(compressed: bool):
    torch.manual_seed(2022)
    mask = block_mask((H, W), block=BLOCK, sparsity=SPARSITY, device='cuda')
    x = torch.rand(size=(BATCH_SIZE, H, W), device='cuda')
    grad_y = torch.rand(size=(BATCH_SIZE, H, W), device='cuda')

    forward_kernel = SparTASparseSoftmaxForwardKernel(compressed=compressed)
    backward_kernel = SparTASparseSoftmaxBackwardKernel(compressed=compressed)

    config = {
        'BLOCK_SIZE_H_VALUE': BH,
        'BLOCK_SIZE_W_VALUE': BW,
        'ROW_TILE_VALUE': RT,
    }
    forward_kernel.set_shape(BATCH_SIZE, H, W)
    forward_kernel.set_masks({'x': mask})
    forward_kernel.compile(config)
    backward_kernel.set_shape(BATCH_SIZE, H, W)
    backward_kernel.set_masks({'grad_y': mask})
    backward_kernel.compile(config)

    target_y = forward_kernel.reference(x, T)
    target_grad_x = backward_kernel.reference(grad_y, target_y, T)

    forward_kernel.test([x, T], [target_y], 0, 1, False)
    backward_kernel.test([grad_y, target_y, T], [target_grad_x], 0, 1, False)
