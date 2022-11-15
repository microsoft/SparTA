# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.testing import block_mask, check


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
    forward_kernel.compile(config, {'x': mask})
    backward_kernel.set_shape(BATCH_SIZE, H, W)
    backward_kernel.compile(config, {'grad_y': mask})

    if compressed:
        x = forward_kernel.get_converter('x').convert(x)
        grad_y = backward_kernel.get_converter('grad_y').convert(grad_y)

    forward_inputs = [x, T]
    check(forward_kernel, forward_inputs, forward_kernel.reference(forward_inputs))

    y = forward_kernel.reference(forward_inputs)
    backward_inputs = [grad_y, y, T]
    check(backward_kernel, backward_inputs, backward_kernel.reference(backward_inputs))


if __name__ == '__main__':
    test_sparse_softmax_forward_kernel(False)
    print('uncompressed pass')
    test_sparse_softmax_forward_kernel(True)
    print('compressed pass')
