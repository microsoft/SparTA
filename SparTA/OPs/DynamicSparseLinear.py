# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
from cmath import inf
import torch
import types
import logging

from .SparseOPBase import SparseOPBase
from .Template.SparseAttention import *
from SparTA.Common.Utils import *
from .BcsrConverter import BcsrConverter
import dynamic_sparse_linear
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class DynamicSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                activation,
                row_ptr,
                col_idx,
                val,
                bias,
                grad_csr_row,
                grad_csr_col,
                grad_csr_val,
                ori_weight,
                M,
                K,
                N,
                block_h,
                block_w):
        ctx.save_for_backward(
            activation,
            grad_csr_row,
            grad_csr_col,
            grad_csr_val,
            bias,
            M, K, N,
            block_h, block_w
        )
        return dynamic_sparse_linear.forward(activation, row_ptr, col_idx, val, bias, M.item(), K.item(), N.item(), block_h.item(), block_w.item())

    @staticmethod
    def backward(ctx, *grad_out):
        assert len(grad_out) == 1
        (activation, grad_csr_row, grad_csr_col, grad_csr_val, bias, M, K, N, block_h, block_w) =  ctx.saved_tensors
        a_grad, w_grad = dynamic_sparse_linear.backward(activation, grad_csr_row, grad_csr_col, grad_csr_val, grad_out[0], M.item(), N.item(), K.item(), block_h.item(), block_w.item())
        return a_grad, None, None, None, torch.zeros_like(bias), None, None, None, w_grad, None, None, None, None, None

class DynamicSparseLinear(SparseOPBase):
    def __init__(self, ori_linear):
        super(DynamicSparseLinear, self).__init__()
        assert isinstance(ori_linear, torch.nn.Linear)
        self.ori_linear = ori_linear.cuda()
        self.mask = torch.ones_like(ori_linear.weight, dtype=torch.int32)
        self.converter = BcsrConverter()
        self.block_h = 64
        self.block_w = 32
        self.csr_row, self.csr_col, self.csr_row_pos, self.csr_val = None, None, None, None
        self.grad_csr_row, self.grad_csr_col, self.grad_csr_row_pos, self.grad_csr_val = None, None, None, None
        self.update_mask(self.mask)

    def update_mask(self, mask):
        self.mask = mask
        self.csr_row, self.csr_col, self.csr_row_pos, self.csr_val = self.converter(
            self.mask, self.ori_linear.weight, self.block_h, self.block_w)
        self.grad_csr_row, self.grad_csr_col, self.grad_csr_row_pos, self.grad_csr_val = self.converter(
            self.mask.t(), self.ori_linear.weight.t(), self.block_h, self.block_w)

    def forward(self, activation):
        N = self.ori_linear.weight.size(0)
        K = self.ori_linear.weight.size(1)
        M = activation.numel() // K
        return DynamicSparseLinearFunction.apply(activation, self.csr_row, self.csr_col,
                                                 self.csr_val, self.ori_linear.bias, self.grad_csr_row,
                                                 self.grad_csr_col, self.grad_csr_val, self.ori_linear.weight,
                                                 torch.tensor(M, dtype=torch.int32), torch.tensor(K, dtype=torch.int32),
                                                 torch.tensor(N, dtype=torch.int32), torch.tensor(self.block_h, dtype=torch.int32),
                                                 torch.tensor(self.block_w, dtype=torch.int32))
