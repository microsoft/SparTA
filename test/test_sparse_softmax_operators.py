# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import numpy as np

from sparta.nn import SparseSoftmax
from sparta.testing import block_mask, sparse_softmax_reference


BATCH_SIZE, DIMS = 1024, 512
BH, BW, RT = 32, 32, 4
BLOCK = (8, 8)
SPARSITY = 0.95
T = np.sqrt(DIMS)


def test_sparse_softmax_operator():
    print(f'sparse_softmax:', end=' ')

    dense_softmax = torch.nn.Softmax(dim=-1).cuda()

    torch.manual_seed(2022)
    mask = block_mask((BATCH_SIZE, DIMS), block=BLOCK, sparsity=SPARSITY).cuda()
    sample_input = torch.rand((BATCH_SIZE, DIMS), dtype=torch.float32).cuda()
    sample_grad = torch.rand((BATCH_SIZE, DIMS), dtype=torch.float32).cuda()

    sparse_softmax = SparseSoftmax(dense_softmax, mask=mask, temperature=T)
    kernel_names = ['forward:y', 'backward:x']
    kernel_config = {
        'BLOCK_SIZE_H_VALUE': BH,
        'BLOCK_SIZE_W_VALUE': BW,
        'ROW_TILE_VALUE': RT,
    }
    sparse_softmax.build(
        params=dict(
            _impl=';'.join([f'{kernel_name}=sparta' for kernel_name in kernel_names]),
            **{
                f'{kernel_name};{param_name}': param_value
                for param_name, param_value in kernel_config.items()
                for kernel_name in kernel_names
            }
        ),
        sample_inputs=[sample_input]
    )

    sample_input.requires_grad = True
    target_output = sparse_softmax_reference(sample_input, mask, temperature=T)
    output = sparse_softmax.forward(sample_input)
    print('forward pass;', end=' ')

    target_output.backward(sample_grad)
    target_grad_input = torch.clone(sample_input.grad)
    sample_input.grad *= 0
    output.backward(sample_grad)
    grad_input = sample_input.grad
    torch.testing.assert_close(grad_input, target_grad_input)
    print('backward pass.')


class TestSparseSoftmaxOperators(unittest.TestCase):

    def test_sparse_softmax_operators(self):
        print('==================== Testing Sparse Softmax Operators ====================')
        test_sparse_softmax_operator()


if __name__ == '__main__':
    unittest.main()
