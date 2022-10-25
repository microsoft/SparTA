# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import numpy as np

from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmax
from sparta.testing import block_mask, sparse_softmax_reference


BATCH_SIZE, H, W = 4, 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.sqrt(W)


def test_sparse_softmax_function(compressed: bool):
    c_str = '_c' if compressed else ''
    func_name = f'sparta_sparse_softmax{c_str}'

    torch.manual_seed(2022)
    mask = block_mask((H, W), block=BLOCK, sparsity=SPARSITY).cuda()
    x = torch.rand(size=(BATCH_SIZE, H, W)).cuda()
    grad_y = torch.rand(size=(BATCH_SIZE, H, W)).cuda()

    x.requires_grad = True
    target_y = sparse_softmax_reference(x, mask, temperature=T)

    kernel_names = ['forward:y', 'backward:x']
    sparse_ctx = SparseBatchSoftmaxCtx(compressed, T)
    sparse_ctx.set_shape(BATCH_SIZE, H, W)
    kernel_config = {
        'BLOCK_SIZE_H_VALUE': BH,
        'BLOCK_SIZE_W_VALUE': BW,
        'ROW_TILE_VALUE': RT,
    }
    sparse_ctx.build(
        config=dict(
            _impl=';'.join([f'{kernel_name}=sparta' for kernel_name in kernel_names]),
            **{
                f'{kernel_name};{param_name}': param_value
                for param_name, param_value in kernel_config.items()
                for kernel_name in kernel_names
            }
        ),
        mask={'x': mask},
    )

    target_y.backward(grad_y)
    target_grad_x = x.grad
    x.grad = None

    if compressed:
        x = sparse_ctx.get_converter('forward:y', 'x').convert(x.detach())
        x.requires_grad = True
        target_y = sparse_ctx.get_converter('forward:y', 'x').convert(target_y)
        grad_y = sparse_ctx.get_converter('backward:x', 'x').convert(grad_y)
        target_grad_x = sparse_ctx.get_converter('backward:x', 'x').convert(target_grad_x)

    print(func_name, end=': ')
    y = SparseBatchSoftmax.apply(sparse_ctx, x)
    torch.testing.assert_close(y, target_y)
    print('forward pass;', end=' ')

    y.backward(grad_y)
    torch.testing.assert_close(x.grad, target_grad_x)
    print('backward pass.')


class TestSparseSoftmaxFunctions(unittest.TestCase):

    def test_sparse_softmax_functions(self):
        print('==================== Testing Sparse Softmax Functions ====================')
        for compressed in [False, True]:
            test_sparse_softmax_function(compressed)


if __name__ == '__main__':
    unittest.main()
