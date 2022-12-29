# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .sparse_opbase import SparseOPBase
from sparta.codegen.template.sparse_attention import *
from sparta.common.utils import *

import seqlen_dynamic_sparse_attention_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



class SeqlenDynamicSparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        inter_result,
        seqlens,
        head_num
    ):
        # ctx.save_for_backward(
        # )

        return seqlen_dynamic_sparse_attention_cpp.forward(
            Q,
            K,
            V,
            inter_result,
            seqlens,
            head_num
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Not implemented yet
        pass


class SeqlenDynamicSparseAttention(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module
    global_seqlen = None

    @staticmethod
    def set_global_seqlens(seqlens):
        # seqlens is an one-dimension tensor with size of [Batchsize]
        # each element in the tensor represents the effective sequence
        # length of current instance
        assert isinstance(seqlens, torch.Tensor)
        assert seqlens.is_cuda
        assert seqlens.dtype == torch.int32, "only support int32 type"
        SeqlenDynamicSparseAttention.global_seqlen = seqlens
        

    def __init__(self, global_mode=True):
        """
        Parameters
        ----------
        HEAD_NUM: int
            The number of heads of the sparse attention
        max_seq_length: int
            The maximum length of the input sequence
        global_mode: bool
            If use the global sparse pattern, if true, then all the sparse_attention
            instance share the same sparse pattern to get the better performance
        """
        super(SeqlenDynamicSparseAttention, self).__init__()
        self.global_mode = global_mode
        # currently only support 32 x 64
        self.inter_result = None  # tensor to store the internal results

    def forward(self, Q, K, V, seqlens=None):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()
        if self.global_mode is not True:
            assert isinstance(seqlens, torch.Tensor)
            assert seqlens.size(0) == Q.size(0)
        else:
            seqlens = SeqlenDynamicSparseAttention.global_seqlen
        # need create val each time
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        # Shape of tensor Q should be {Batchsize, sequence length, hidden dim}
        batch_size = Q.size(0)
        head_num =  Q.size(1)
        max_seq_len = Q.size(2)
        hidden_dim = Q.size(3)
        err_msg = 'Currently, seq_len and hidden_dim should be divisible by 32'
        assert max_seq_len % 32 == 0, err_msg
        assert hidden_dim % 32 == 0
        if self.inter_result is None or self.inter_result.numel() < batch_size * head_num * max_seq_len * max_seq_len:
            self.inter_result = torch.zeros(batch_size * head_num * max_seq_len * max_seq_len,
                          dtype=torch.float32, device=Q.device)
        result = SeqlenDynamicSparseAttentionFunction.apply(Q, K, V,
                                                      self.inter_result,
                                                      seqlens,
                                                      head_num)

        return result

    def reference_forward(self, Q, K, V, attention_mask):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        add_mask = torch.zeros(attention_mask.size()).to(Q.device)
        add_mask[attention_mask == 0] = float(-inf)
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        added = torch.add(dots, add_mask)
        attn = added.softmax(dim=-1)
        nan_pos = torch.isnan(attn)
        attn[nan_pos] = 0.0
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        return ref_out
