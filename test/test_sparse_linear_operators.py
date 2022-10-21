# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.nn import SparseLinear
from sparta.testing import block_mask


M, K, N = 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


def test_sparse_linear_operator(sparse_type: str, biased: bool):
    b_str = '_b' if biased else ''
    print(f'sparse_linear_{sparse_type}{b_str}:', end=' ')

    dense_linear = torch.nn.Linear(K, N, bias=biased).cuda()

    torch.manual_seed(2022)
    sample_input = torch.rand((M, K), dtype=torch.float32).cuda()
    dense_weight = torch.rand((N, K), dtype=torch.float32).cuda()
    bias = torch.rand((N, ), dtype=torch.float32).cuda()
    sample_grad = torch.rand((M, N), dtype=torch.float32).cuda()

    if sparse_type == 'sdd':
        mask = block_mask((M, K), block=BLOCK, sparsity=SPARSITY).cuda()
        sample_input *= mask
        mask_dict = {'input_mask': mask}
    elif sparse_type == 'dsd':
        mask = block_mask((N, K), block=BLOCK, sparsity=SPARSITY).cuda()
        dense_weight *= mask
        mask_dict = {'weight_mask': mask}
    else:
        mask = block_mask((M, N), block=BLOCK, sparsity=SPARSITY).cuda()
        sample_grad *= mask
        mask_dict = {'output_mask': mask}
    if biased:
        dense_linear.load_state_dict({'weight': dense_weight, 'bias': bias})
    else:
        dense_linear.load_state_dict({'weight': dense_weight})

    sparse_linear = SparseLinear(dense_linear, **mask_dict)
    kernel_names = ['forward:C', 'backward:A', 'backward:B']
    kernel_config = {
        'BLOCK_SIZE_M_VALUE': BM,
        'BLOCK_SIZE_K_VALUE': BK,
        'BLOCK_SIZE_N_VALUE': BN,
        'THREAD_SIZE_M_VALUE': TM,
        'THREAD_SIZE_K_VALUE': TK,
        'THREAD_SIZE_N_VALUE': TN,
    }
    sparse_linear.build(
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
    target_output = dense_linear.forward(sample_input)
    output = sparse_linear.forward(sample_input)
    if sparse_type == 'dds':
        torch.testing.assert_close(output * mask, target_output * mask, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(output, target_output, rtol=1e-4, atol=1e-4)
    print('forward pass;', end=' ')

    target_output.backward(sample_grad)
    target_grad_input = torch.clone(sample_input.grad)
    target_grad_weight = torch.clone(dense_linear.weight.grad)
    sample_input.grad *= 0
    dense_linear.weight.grad *= 0
    output.backward(sample_grad)
    grad_input = sample_input.grad
    grad_weight = sparse_linear.weight.grad
    if sparse_type == 'sdd':
        torch.testing.assert_close(grad_input * mask, target_grad_input * mask, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(grad_input, target_grad_input, rtol=1e-4, atol=1e-4)
    if sparse_type == 'dsd':
        weight_converter = sparse_linear._sparse_ctx.get_converter('backward:B', 'B')
        target_grad_weight = weight_converter(target_grad_weight)
    torch.testing.assert_close(grad_weight, target_grad_weight, rtol=1e-4, atol=1e-4)
    print('backward pass.')


class TestSparseLinearOperators(unittest.TestCase):

    def test_sparse_linear_operators(self):
        print('==================== Testing Sparse Linear Operators ====================')
        for sparse_type in ['sdd', 'dsd', 'dds']:
            for biased in [False, True]:
                test_sparse_linear_operator(sparse_type, biased)


if __name__ == '__main__':
    unittest.main()
