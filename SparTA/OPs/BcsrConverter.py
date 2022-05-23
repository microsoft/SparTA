# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
import torch
import types
import logging
from torch.utils.cpp_extension import load as module_load
from .SparseOPBase import SparseOPBase
from SparTA.Common.Utils import *
import convert_bcsr
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class BcsrConverter(SparseOPBase):
    """
    The Sparse Attention module.
    """

    def __init__(self):
        super(BcsrConverter, self).__init__()
        self.csr_row = None
        self.csr_col = None
        self.csr_value = None
        self.csr_row_pos = None

    def forward(self, sparse_pattern, dense_values, block_size_h, block_size_w):
        """
        Q, K, V are the output tensors of the corresponding
        projection linear layers.
        """
        assert(isinstance(sparse_pattern, torch.Tensor))
        assert(isinstance(dense_values, torch.Tensor))
        # currently only support on the cuda devices
        assert(sparse_pattern.is_cuda)
        assert(dense_values.is_cuda)
        self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value = convert_bcsr.forward(sparse_pattern, dense_values, block_size_h, block_size_w)
        return self.csr_row, self.csr_col, self.csr_row_pos, self.csr_value

