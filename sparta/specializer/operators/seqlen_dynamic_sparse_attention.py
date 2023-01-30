# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

import seqlen_dynamic_sparse_attention_cpp


class SeqlenDynamicSparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        inter_result: torch.Tensor,
        seqlens: torch.Tensor,
        H: int
    ):
        # ctx.save_for_backward()
        return seqlen_dynamic_sparse_attention_cpp.forward(Q, K, V, inter_result, seqlens, H)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Not implemented yet
        pass


class SeqlenDynamicSparseAttention(torch.nn.Module):
    r"""The sequence-length sparse attention operator that support dynamic sparse patterns.
    Only support 32 x 64 block size now.

    Args:
        global_mode (bool): Whether all SeqlenDynamicSparseAttention instances share
            the same sparse pattern. If true, there is no need to convert the sparse
            pattern for each instance and the performance should be better.

    Shape:
        - Input1: :math:`(B, H, N, E)` where :math:`B` is the batch size,
            :math:`H` is the number of heads, :math:`N` is the max sequence length
            and :math:`E` is the embed dimension.
        - Input2: :math:`(B, H, N, E)`.
        - Input3: :math:`(B, H, N, E)`, same shape as the second input.
        - Output: :math:`(B, H, N, E)`, same shape as the first input.
    """

    global_seqlen: Optional[torch.Tensor] = None

    def __init__(self, global_mode=True):
        super().__init__()
        self.global_mode = global_mode
        self._inter_result: Optional[torch.Tensor] = None  # Tensor to store the internal results

    @staticmethod
    def set_global_seqlens(seqlens: torch.Tensor):
        """Set sequence lengths for global mode.

        Args:
            seqlens (torch.Tensor): An one-dimension tensor represents the effective sequence
                length of each batch.
        """
        assert isinstance(seqlens, torch.Tensor)
        assert seqlens.is_cuda
        assert seqlens.dtype == torch.int32, "only support int32 for seqlens"
        SeqlenDynamicSparseAttention.global_seqlen = seqlens

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor, 
        seqlens: Optional[torch.Tensor] = None,
    ):
        """Dynamic sequence-length sparse attention forward function.

        Args:
            Q (torch.Tensor): the output tensor of the projection linear layer of query.
            K (torch.Tensor): the output tensor of the projection linear layer of key.
            V (torch.Tensor): the output tensor of the projection linear layer of value.
            seqlens (torch.Tensor, optional): An one-dimension tensor represents the effective
                sequence length of each batch.

        Returns:
            torch.Tensor: The output tensor.
        """
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()

        if self.global_mode:
            seqlens = SeqlenDynamicSparseAttention.global_seqlen

        B, H, N, E = Q.shape
        assert B == seqlens.size(0)
        assert N % 32 == 0, 'max sequence length should be divisible by 32'
        assert E % 32 == 0, 'embed dimension size should be divisible by 32'

        inter_result_size = B * H * N * N
        if self._inter_result is None or self._inter_result.numel() < inter_result_size:
            self._inter_result = torch.zeros(inter_result_size, dtype=torch.float32, device=Q.device)

        return SeqlenDynamicSparseAttentionFunction.apply(Q, K, V, self._inter_result, seqlens, H)
