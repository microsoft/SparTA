# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.nn import SparseLinear
from sparta.testing import block_mask, check


BATCH_SIZE, IN_DIMS, OUT_DIMS = 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


@pytest.mark.parametrize('mode', ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize('biased', [False, True])
def test_sparse_linear_operator(mode: str, biased: bool):
    dense_linear = torch.nn.Linear(IN_DIMS, OUT_DIMS, bias=biased).cuda()

    torch.manual_seed(2022)
    sample_input = torch.rand((BATCH_SIZE, IN_DIMS), dtype=torch.float32).cuda()
    dense_weight = torch.rand((OUT_DIMS, IN_DIMS), dtype=torch.float32).cuda()
    bias = torch.rand((OUT_DIMS, ), dtype=torch.float32).cuda()
    sample_grad = torch.rand((BATCH_SIZE, OUT_DIMS), dtype=torch.float32).cuda()

    if mode == 'sdd':
        mask = block_mask((BATCH_SIZE, IN_DIMS), block=BLOCK, sparsity=SPARSITY).cuda()
        sample_input *= mask
        mask_dict = {'input_mask': mask}
    elif mode == 'dsd':
        mask = block_mask((OUT_DIMS, IN_DIMS), block=BLOCK, sparsity=SPARSITY).cuda()
        dense_weight *= mask
        mask_dict = {'weight_mask': mask}
    else:
        mask = block_mask((BATCH_SIZE, OUT_DIMS), block=BLOCK, sparsity=SPARSITY).cuda()
        sample_grad *= mask
        mask_dict = {'output_mask': mask}
    if biased:
        dense_linear.load_state_dict({'weight': dense_weight, 'bias': bias})
    else:
        dense_linear.load_state_dict({'weight': dense_weight})

    sparse_linear = SparseLinear(dense_linear, **mask_dict)
    kernel_names = ['forward:C', 'backward:A', 'backward:B']
    sparse_linear.build(
        params={
            kernel_name: {
                '_impl': 'sparta',
                'BLOCK_SIZE_M_VALUE': BM,
                'BLOCK_SIZE_K_VALUE': BK,
                'BLOCK_SIZE_N_VALUE': BN,
                'THREAD_SIZE_M_VALUE': TM,
                'THREAD_SIZE_K_VALUE': TK,
                'THREAD_SIZE_N_VALUE': TN,
            }
            for kernel_name in kernel_names
        },
        sample_inputs=[sample_input],
    )

    sample_input.requires_grad = True
    target_output = dense_linear.forward(sample_input)

    if mode == 'dds':
        output_converter = sparse_linear._sparse_ctx.get_converter('forward:C', 'C')
        target_output *= output_converter.get_mask()

    target_output.backward(sample_grad)
    target_grad_input = sample_input.grad
    sample_input.grad = None
    target_grad_weight = dense_linear.weight.grad
    dense_linear.weight.grad = None
    if biased:
        target_grad_bias = dense_linear.bias.grad
        dense_linear.bias.grad = None

    if mode == 'sdd':
        input_converter = sparse_linear._sparse_ctx.get_converter('backward:A', 'A')
        target_grad_input *= input_converter.get_mask()
    elif mode == 'dsd':
        weight_converter = sparse_linear._sparse_ctx.get_converter('backward:B', 'B')
        target_grad_weight = weight_converter.convert(target_grad_weight)

    if biased:
        def linear_forward_backward(x: torch.Tensor, grad: torch.Tensor):
            y = sparse_linear.forward(x)
            y.backward(grad)
            return y, x.grad, sparse_linear.weight.grad, sparse_linear.bias.grad
        target_outputs = [target_output, target_grad_input, target_grad_weight, target_grad_bias]
    else:
        def linear_forward_backward(x: torch.Tensor, grad: torch.Tensor):
            y = sparse_linear.forward(x)
            y.backward(grad)
            return y, x.grad, sparse_linear.weight.grad
        target_outputs = [target_output, target_grad_input, target_grad_weight]

    check(linear_forward_backward, [sample_input, sample_grad], target_outputs)
