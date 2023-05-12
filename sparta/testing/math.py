# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np


def sparse_softmax_forward_reference(
    x: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.,
) -> torch.Tensor:
    """Sparse softmax reference function. Masked input values are treated as negative infinity.

    Args:
        x (torch.Tensor): The input tensor. We will calculate softmax along the last axis.
        mask (torch.Tensor): The mask tensor having the same shape with the input tensor.
        temperature: The temperature value of softmax.

    Returns:
        torch.Tensor: The output tensor having the same shape with the input tensor. Notice that
            the return value on completely masked rows will be 0.
    """
    C_max = x.max(axis=-1).values.unsqueeze(-1)
    C_exp = torch.exp((x - C_max) / temperature) * mask
    C_exp_sum = C_exp.sum(axis=-1).unsqueeze(-1) + 1e-10
    return C_exp / C_exp_sum


def sparse_softmax_backward_reference(
    grad: torch.Tensor,
    output: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.,
) -> torch.Tensor:
    """Sparse softmax backward reference function.

    Args:
        grad (torch.Tensor): The gradient of output tensor.
        output (torch.Tensor): The output tensor of sparse softmax forward function.
        mask (torch.Tensor): The mask tensor having the same shape with the grad and output tensor.
        temperature: The temperature value of softmax.

    Returns:
        torch.Tensor: The gradient of input tensor. The return value on masked positions will be 0.
    """
    masked_output = output * mask
    C_prod = grad * masked_output
    C_sum = C_prod.sum(axis=-1).unsqueeze(-1)
    return (C_prod - masked_output * C_sum) / temperature


def sparse_multi_head_attention_forward_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = np.nan,
) -> torch.Tensor:
    r"""Sparse multi-head attention reference function with batch size :math:`B`,
    head number :math:`H`, sourse sequence length :math:`N_{source}`,
    target sequence length :math:`N_{target}` and embed dimention :math:`E`.

    Args:
        query (torch.Tensor): The input query tensor of shape :math:`(B, H, N_{target}, E)`.
        key (torch.Tensor): The input key tensor of shape :math:`(B, H, N_{source}, E)`.
        value (torch.Tensor): The input value tensor of shape :math:`(B, H, N_{source}, E)`.
        mask (torch.Tensor): The mask tensor of shape :math:`(N_{target}, N_{source})`.
        temperature (float): The softmax temperature which is set to :math:`\sqrt{N_{source}}` by default.

    Returns:
        torch.Tensor: Sparse multi-head attention output of shape :math:`(B, H, N_{target}, E)`.
    """
    if np.isnan(temperature):
        temperature = np.sqrt(key.shape[-1])
    high_dims = ''.join([chr(ord('a') + i) for i in range(len(query.shape) - 2)])
    p = torch.einsum(f'{high_dims}mk, {high_dims}nk -> {high_dims}mn', query, key)
    s = sparse_softmax_forward_reference(p, mask, temperature)
    return torch.einsum(f'{high_dims}mn, {high_dims}nk -> {high_dims}mk', s, value)


def sparse_multi_head_attention_backward_reference(
    grad: torch.Tensor,
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = np.nan,
) -> torch.Tensor:
    r"""Sparse multi-head attention backward reference function.

    Args:
        grad (torch.Tensor): The gradient of output tensor. Shape: :math:`(B, H, N_{target}, E)`.
        output (torch.Tensor): The output tensor of forward function. Shape: :math:`(B, H, N_{target}, E)`.
        query (torch.Tensor): The input query tensor of forward function. Shape: :math:`(B, H, N_{target}, E)`.
        key (torch.Tensor): The input key tensor of forward function. Shape: :math:`(B, H, N_{source}, E)`.
        value (torch.Tensor): The input value tensor of forward function. Shape: :math:`(B, H, N_{source}, E)`.
        mask (torch.Tensor): The mask tensor. Shape :math:`(N_{target}, N_{source})`.
        temperature (float): The softmax temperature which is set to :math:`\sqrt{N_{source}}` by default.

    Returns:
        Tuple: The gradient of query, key and value respectively..
    """
    if np.isnan(temperature):
        temperature = np.sqrt(key.shape[-1])
    high_dims = ''.join([chr(ord('a') + i) for i in range(len(query.shape) - 2)])
    p = torch.einsum(f'{high_dims}mk, {high_dims}nk -> {high_dims}mn', query, key)
    s = sparse_softmax_forward_reference(p, mask, temperature)
    grad_v = torch.einsum(f'{high_dims}mn, {high_dims}mk -> {high_dims}nk', p, grad)
    grad_s = torch.einsum(f'{high_dims}nk, {high_dims}mk -> {high_dims}mn', value, grad)
    grad_p = sparse_softmax_backward_reference(grad_s, s, mask, temperature)
    grad_q = torch.einsum(f'{high_dims}nk, {high_dims}mn -> {high_dims}mk', key, grad_p)
    grad_k = torch.einsum(f'{high_dims}mk, {high_dims}mn -> {high_dims}nk', query, grad_p)
    return grad_q, grad_k, grad_v
