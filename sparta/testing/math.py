# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


def sparse_softmax_reference(x: torch.Tensor, mask: torch.Tensor):
    '''Sparse softmax reference function. Masked input values are treated as negative infinity.

    Args:
        x (torch.Tensor): The input tensor. We will calculate softmax along the last axis.
        mask (torch.Tensor): The mask tensor having the same shape with the input tensor.

    Returns:
        torch.Tensor: The output tensor having the same shape with the input tensor. Notice that
            the return value on completely masked rows will be 0.
    '''
    C_max = x.max(axis=-1).values.reshape((-1, 1))
    C_exp = torch.exp(x - C_max) * mask
    C_exp_sum = C_exp.sum(axis=-1).reshape((-1, 1)) + 1e-10
    return C_exp / C_exp_sum
