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


def sparse_multi_head_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sparse multi-head attention reference function of shape (B * H, Ns, Nt, E).
    Where B is the batch size, H is the head number, Ns is the sourse sequence length,
    Nt is the target sequence length, and E is the embed dimention.

    Args:
        query (torch.Tensor): The input query tensor of shape (B * H, Nt, E).
        key (torch.Tensor): The input key tensor of shape (B * H, Ns, E).
        value (torch.Tensor): The input value tensor of shape (B * H, Ns, E).
        mask (torch.Tensor): The mask tensor of shape (Nt, Ns).

    Returns:
        torch.Tensor: Sparse multi-head attention output of shape (B * H, Nt, E).
    """
    temperature = np.sqrt(query.shape[-1])
    qk = torch.einsum('bmk,bnk->bmn', query, key)
    sm = sparse_softmax_forward_reference(qk, mask, temperature)
    return torch.einsum('bmk,bkn->bmn', sm, value)
