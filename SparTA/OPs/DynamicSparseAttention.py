# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .SparseOPBase import SparseOPBase
from .Template.SparseAttention import *
from SparTA.Common.Utils import *
from .BcsrConverter import BcsrConverter
import dynamic_sparse_attention
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



class DynamicSparseAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        inter_result,
        row_ptr,
        col_idx,
        row_pos,
        val_mask,
        block_nnz,
        head_num
    ):
        # ctx.save_for_backward(
        # )

        return dynamic_sparse_attention.forward(
            Q,
            K,
            V,
            inter_result,
            row_ptr,
            col_idx,
            row_pos,
            val_mask,
            block_nnz,
            head_num
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass


class DynamicSparseAttention(SparseOPBase):
    """
    The Sparse Attention module that support the dynamic sparse pattern.
    """
    # if all the sparse attention share the same sparse pattern, then
    # no need to convert the sparse pattern for each module. Set the global_mode
    # to be true when initialize the module
    global_sparse_pattern = None
    global_bcsr_row = None
    global_bcsr_col = None
    global_bcsr_row_pos = None
    global_bcsr_val_mask = None
    global_block_nnz = None
    global_converter = BcsrConverter()
    global_block_h = 32
    global_block_w = 32

    @staticmethod
    def set_global_sparse_pattern(sparse_pattern):
        assert isinstance(sparse_pattern, torch.Tensor)
        assert sparse_pattern.dtype == torch.int32, "only support int32 type"
        DynamicSparseAttention.global_sparse_pattern = sparse_pattern
        n_row = DynamicSparseAttention.global_sparse_pattern.size(0) // DynamicSparseAttention.global_block_h
        DynamicSparseAttention.global_bcsr_row, DynamicSparseAttention.global_bcsr_col, DynamicSparseAttention.global_bcsr_row_pos, \
            DynamicSparseAttention.global_bcsr_val_mask = DynamicSparseAttention.global_converter(
                DynamicSparseAttention.global_sparse_pattern, DynamicSparseAttention.global_sparse_pattern.to(torch.float), DynamicSparseAttention.global_block_h, DynamicSparseAttention.global_block_w)
        # import pdb; pdb.set_trace()
        DynamicSparseAttention.global_block_nnz = DynamicSparseAttention.global_bcsr_row[n_row].item()

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
        super(DynamicSparseAttention, self).__init__()
        self.global_mode = global_mode
        # currently only support 32 x 64
        self.block_size_h = 32
        self.block_size_w = 32
        self.converter = BcsrConverter()
        self.inter_result = None  # tensor to store the internal results

    def forward(self, Q, K, V, sparse_mask=None):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        sparse_
        """

        if self.global_mode is not True:
            assert isinstance(sparse_mask, torch.Tensor)
            csr_row, csr_col, csr_row_pos, csr_value_mask = self.converter(
                sparse_mask, sparse_mask.to(torch.float), self.block_size_h, self.block_size_w)
            n_row =  sparse_mask.size(0) // self.block_h
            block_nnz = csr_row[n_row].item()
        else:
            csr_row, csr_col, csr_row_pos, csr_value_mask = DynamicSparseAttention.global_bcsr_row, DynamicSparseAttention.global_bcsr_col, DynamicSparseAttention.global_bcsr_row_pos, DynamicSparseAttention.global_bcsr_val_mask
            sparse_mask = DynamicSparseAttention.global_sparse_pattern
            block_nnz = DynamicSparseAttention.global_block_nnz
        # need create val each time
        assert isinstance(Q, torch.Tensor)
        assert isinstance(K, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        # Shape of tensor Q should be {Batchsize, sequence length, hidden dim}
        batch_size = Q.size(0)
        head_num =  Q.size(1)
        seq_len = Q.size(2)
        hidden_dim = Q.size(3)
        err_msg = 'Currently, seq_len and hidden_dim should be divisible by 32'
        assert seq_len % 32 == 0, err_msg
        assert hidden_dim % 32 == 0
        assert seq_len == sparse_mask.size(0), "input sequence length dose not match the given sparse pattern"
        sparse_val_size = block_nnz * self.block_size_h * self.block_size_w
        if self.inter_result is None or self.inter_result.numel() < batch_size * head_num * block_nnz * self.block_h * self.block_w:
            self.inter_result = torch.zeros(batch_size * head_num * sparse_val_size,
                          dtype=torch.float32, device=Q.device)
        result = DynamicSparseAttentionFunction.apply(Q, K, V,
                                                      self.inter_result,
                                                      csr_row,
                                                      csr_col,
                                                      csr_row_pos,
                                                      csr_value_mask,
                                                      block_nnz,
                                                      head_num)

        return result

    def reference_forward(self, Q, K, V, sparse_pattern=None):
        """
        Calculate the reference result the sparse attention to test the correctness.
        """
        if self.global_mode:
            out_mask = DynamicSparseAttention.global_sparse_pattern
        else:
            assert isinstance(sparse_pattern, torch.Tensor)
            out_mask = sparse_pattern
        add_mask = torch.zeros(out_mask.size()).to(Q.device)
        add_mask[out_mask == 0] = float(-inf)
        dots = torch.einsum('b h m k, b h n k -> b h m n', Q, K)
        added = torch.add(dots, add_mask)
        attn = added.softmax(dim=-1)
        nan_pos = torch.isnan(attn)
        attn[nan_pos] = 0.0
        ref_out = torch.einsum('b h m n, b h n k -> b h m k', attn, V)

        return ref_out
