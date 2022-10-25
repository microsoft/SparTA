# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from turtle import backward
import unittest

import torch
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel
from sparta.testing import block_mask, test_latency, sparse_softmax_reference, sparse_softmax_backward_reference


BATCH_SIZE, H, W = 4, 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.sqrt(W)


def test_sparse_softmax_kernel(compressed: bool):
    c_str = '_c' if compressed else ''
    forward_kernel_name = f'sparta_sparse_softmax{c_str}'
    backward_kernel_name = f'sparta_sparse_softmax{c_str}_backward'

    torch.manual_seed(2022)
    mask = block_mask((H, W), block=BLOCK, sparsity=SPARSITY).cuda()
    x = torch.rand(size=(BATCH_SIZE, H, W)).cuda()
    grad_y = torch.rand(size=(BATCH_SIZE, H, W)).cuda()

    forward_kernel = SparTASparseSoftmaxKernel(compressed=compressed)
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

    target_y = sparse_softmax_reference(x, mask, temperature=T)
    target_grad_x = sparse_softmax_backward_reference(grad_y, target_y, mask, temperature=T)

    if compressed:
        x = forward_kernel.get_converter('x').convert(x)
        mask = forward_kernel.get_converter('mask').convert(mask)
        target_y = forward_kernel.get_converter('y').convert(target_y)
        grad_y = backward_kernel.get_converter('grad_y').convert(grad_y)
        target_grad_x = backward_kernel.get_converter('grad_x').convert(target_grad_x)

    forward_latency = test_latency(
        func=forward_kernel,
        inputs=[x, mask.to(torch.int32), np.float32(1 / T)],
        target_outputs=[target_y],
        num_warmups=10,
        num_iters=10,
    )
    print(f'{forward_kernel_name}: {forward_latency} ms')

    backward_latency = test_latency(
        func=backward_kernel,
        inputs=[grad_y, target_y, mask.to(torch.int32), np.float32(1 / T)],
        target_outputs=[target_grad_x],
        num_warmups=10,
        num_iters=10,
    )
    print(f'{backward_kernel_name}: {backward_latency} ms')


class TestSparseSoftmaxKernels(unittest.TestCase):

    def test_sparse_softmax_kernels(self):
        print('==================== Testing Sparse Softmax Kernels ====================')
        for compressed in [False, True]:
            test_sparse_softmax_kernel(compressed)


if __name__ == '__main__':
    unittest.main()
