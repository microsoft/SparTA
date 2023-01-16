# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import abc
from typing import Any, Dict

import torch
import jinja2
import numpy as np

from sparta import __env_ready__
if __env_ready__:
    # we may need to dry run without GPU (e.g., for document generation)
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule

from sparta.tesa import TeSAIndexes, TeSAFunctionContext


KERNEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kernels')


class BCSIndexes(TeSAIndexes):

    def __init__(
        self,
        function_context: BCSFunctionContext,
        raw_mask: torch.Tensor,
        block_nnz: int,
        block_H: int,
        block_W: int,
    ):
        super().__init__(function_context, raw_mask)
        self.nnz = np.int32(block_nnz)
        # Shape variables
        self.block_H = block_H
        self.block_W = block_W
        self.H, self.W = raw_mask.shape
        self.row_num = self.H // self.block_H
        self.col_num = self.W // self.block_W
        self.block_nnz = block_nnz

    @abc.abstractmethod
    def get_block_mask(self) -> torch.Tensor:
        """Rebuild block mask using BCSR/BCSC indexes"""

    def get_mask(self) -> torch.Tensor:
        mask = self.get_block_mask().unsqueeze(0).unsqueeze(0)
        mask = mask.tile((self.block_H, self.block_W, 1, 1)).swapaxes(1, 2)
        return mask.reshape(self.raw_mask.shape).contiguous()

    @abc.abstractmethod
    def convert(self, dense: torch.Tensor) -> torch.Tensor:
        """Convert dense tensor to compressed sparse value."""
        # TODO: convert_uint8()

    @abc.abstractmethod
    def inverse(self, sparse_val: torch.Tensor) -> torch.Tensor:
        """Inversely convert compressed sparse value to dense tensor."""

    @abc.abstractmethod
    def _row_sum(self, sparse_val: torch.Tensor) -> torch.Tensor:
        """Calculate sum value along rows."""

    @abc.abstractmethod
    def _col_sum(self, sparse_val: torch.Tensor) -> torch.Tensor:
        """Calculate sum value along columns."""

    def sum(self, sparse_val: torch.Tensor, axis: int = -1) -> torch.Tensor:
        inverse_axis = -axis if axis < 0 else len(sparse_val.shape) - axis
        if inverse_axis == 1:
            return self._row_sum(sparse_val)
        elif inverse_axis == 2:
            return self._col_sum(sparse_val)
        else:
            raise ValueError(f'axis #{axis} is not sparsed, please use torch.sum() instead.')


class BCSRIndexes(BCSIndexes):

    def __init__(
        self,
        function_context: BCSFunctionContext,
        raw_mask: torch.Tensor,
        row_ptr: torch.Tensor,
        BCSR_idx: torch.Tensor,
        block_nnz: int,
        block_H: int,
        block_W: int,
    ):
        super().__init__(function_context, raw_mask, block_nnz, block_H, block_W)
        self.row_ptr = row_ptr
        self.BCSR_idx = BCSR_idx

    def get_block_mask(self):
        block_mask_val = torch.ones(
            size=(self.block_nnz, ),
            dtype=torch.uint8,
            device=self.row_ptr.device
        )
        col_idx = self.BCSR_idx.bitwise_and(0xffff)
        return torch.sparse_csr_tensor(self.row_ptr, col_idx, block_mask_val).to_dense()

    def convert(self, dense: torch.Tensor):
        return self.function_context.convert(
            dense, self.BCSR_idx,
            self.block_nnz, self.H, self.W
        )

    def inverse(self, sparse_val: torch.Tensor):
        return self.function_context.inverse(
            sparse_val, self.BCSR_idx,
            self.block_nnz, self.H, self.W
        )

    def _row_sum(self, sparse_val: torch.Tensor):
        return self.function_context.row_sum(
            sparse_val, self.row_ptr, self.nnz,
            self.row_num, self.H,
        )

    def _col_sum(self, sparse_val: torch.Tensor):
        return self.inverse(sparse_val).sum(-2)


class BCSCIndexes(BCSIndexes):

    def __init__(
        self,
        function_context: BCSFunctionContext,
        raw_mask: torch.Tensor,
        col_ptr: torch.Tensor,
        BCSC_idx: torch.Tensor,
        block_nnz: int,
        block_H: int,
        block_W: int,
    ):
        super().__init__(function_context, raw_mask, block_nnz, block_H, block_W)
        self.col_ptr = col_ptr
        self.BCSC_idx = BCSC_idx

    def get_block_mask(self):
        block_mask_val = torch.ones(
            size=(self.block_nnz, ),
            dtype=torch.uint8,
            device=self.col_ptr.device
        )
        row_idx = self.BCSC_idx.bitwise_and(0xffff)
        return torch.sparse_csr_tensor(self.col_ptr, row_idx, block_mask_val).to_dense().T

    def convert(self, dense: torch.Tensor):
        return self.function_context.convert(
            dense, self.BCSC_idx,
            self.block_nnz, self.H, self.W
        )

    def inverse(self, sparse_val: torch.Tensor):
        return self.function_context.inverse(
            sparse_val, self.BCSC_idx,
            self.block_nnz, self.H, self.W
        )

    def _row_sum(self, sparse_val: torch.Tensor):
        return self.inverse(sparse_val).sum(-1)

    def _col_sum(self, sparse_val: torch.Tensor):
        return self.inverse(sparse_val).sum(-2)


class BCSRCIndexes(BCSRIndexes):

    def __init__(
        self,
        function_context: BCSFunctionContext,
        raw_mask: torch.Tensor,
        row_ptr: torch.Tensor,
        col_ptr: torch.Tensor,
        BCSR_idx: torch.Tensor,
        BCSC_idx: torch.Tensor,
        block_nnz: int,
        block_H: int,
        block_W: int,
    ):
        super().__init__(
            function_context, raw_mask, row_ptr, BCSR_idx,
            block_nnz, block_H, block_W,
        )
        self.col_ptr = col_ptr
        self.BCSC_idx = BCSC_idx

    def _col_sum(self, sparse_val: torch.Tensor):
        return self.function_context.col_sum(
            sparse_val, self.col_ptr, self.BCSC_idx, self.nnz,
            self.col_num, self.W,
        )


_extra_buffer_cache: Dict[str, torch.Tensor] = {}


class BCSFunctionContext(TeSAFunctionContext):

    def __init__(self, block_H: int, block_W: int, BCSR: bool, BCSC: bool):
        assert block_H in [4, 8, 16, 32, 64, 128, 256]
        assert block_W in [4, 8, 16, 32, 64, 128, 256]
        self._block_H = block_H
        self._block_W = block_W
        self._block_size = block_H * block_W
        self._index_block_dim = (self._block_size // min(block_W, 16), 1, 1)
        self._convert_block_dim = (min(self._block_size // 4, 256), 1, 1)
        self._sum_block_dim = (min(self._block_size, 256), 1, 1)
        with open(os.path.join(KERNEL_DIR, 'block_compressed.cu.j2')) as f:
            kernel_code = jinja2.Template(f.read()).render({
                'BH': block_H, 'BW': block_W, 'BCSR': BCSR, 'BCSC': BCSC,
            })
        source_module = SourceModule(kernel_code, options=['-O3'])
        self.index_1_kernel = source_module.get_function('bcs_index_1')
        self.index_2_kernel = source_module.get_function('bcs_index_2')
        self.convert_kernel = source_module.get_function('dense_to_bcs_val')
        self.inverse_kernel = source_module.get_function('bcs_val_to_dense')
        if BCSR:
            self.row_sum_kernel = source_module.get_function('bcsr_val_sum_row')
            if BCSC:
                self.col_sum_kernel = source_module.get_function('bcsr_val_sum_col')
            else:
                self.build_indexes = self._build_BCSR_indexes
        else:
            self.build_indexes = self._build_BCSC_indexes

    def _get_index_extra_buffer(self, row_num: int, col_num: int, device: Any):
        extra_buffer_id = f'{row_num},{col_num}'
        if extra_buffer_id in _extra_buffer_cache:
            extra_buffer = _extra_buffer_cache[extra_buffer_id]
            extra_buffer.fill_(0)
        else:
            buffer_size = row_num + col_num + row_num * col_num * 3
            extra_buffer = torch.zeros((buffer_size, ), dtype=torch.int32, device=device)
            _extra_buffer_cache[extra_buffer_id] = extra_buffer
        return extra_buffer

    def build_indexes(self, mask: torch.Tensor) -> BCSRCIndexes:
        H, W = mask.shape
        row_num = H // self._block_H
        col_num = W // self._block_W
        extra_buffer = self._get_index_extra_buffer(row_num, col_num, mask.device)
        row_ptr = torch.zeros((row_num + 1, ), dtype=torch.int32, device='cuda')
        col_ptr = torch.zeros((col_num + 1, ), dtype=torch.int32, device='cuda')
        self.index_1_kernel(
            mask, row_ptr, col_ptr, extra_buffer, np.int32(H), np.int32(W),
            block=self._index_block_dim, grid=(col_num, row_num, 1)
        )
        row_ptr = torch.cumsum(row_ptr, dim=0, dtype=torch.int32)
        col_ptr = torch.cumsum(col_ptr, dim=0, dtype=torch.int32)
        block_nnz = row_ptr[-1].item()
        BCSR_idx = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        BCSC_idx = torch.empty((block_nnz, ), dtype=torch.int64, device='cuda')
        self.index_2_kernel(
            row_ptr, col_ptr, BCSR_idx, BCSC_idx, extra_buffer,
            block=(1, 1, 1), grid=(col_num, row_num, 1)
        )
        return BCSRCIndexes(
            self, mask, row_ptr, col_ptr, BCSR_idx, BCSC_idx,
            block_nnz, self._block_H, self._block_W,
        )

    def _build_BCSR_indexes(self, mask: torch.Tensor) -> BCSRIndexes:
        H, W = mask.shape
        row_num = H // self._block_H
        col_num = W // self._block_W
        extra_buffer = self._get_index_extra_buffer(row_num, col_num, mask.device)
        row_ptr = torch.zeros((row_num + 1, ), dtype=torch.int32, device='cuda')
        self.index_1_kernel(
            mask, row_ptr, extra_buffer, np.int32(H), np.int32(W),
            block=self._index_block_dim, grid=(col_num, row_num, 1)
        )
        row_ptr = torch.cumsum(row_ptr, dim=0, dtype=torch.int32)
        block_nnz = row_ptr[-1].item()
        BCSR_idx = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        self.index_2_kernel(
            row_ptr, BCSR_idx, extra_buffer,
            block=(1, 1, 1), grid=(col_num, row_num, 1)
        )
        return BCSRIndexes(
            self, mask, row_ptr, BCSR_idx,
            block_nnz, self._block_H, self._block_W,
        )

    def _build_BCSC_indexes(self, mask: torch.Tensor) -> BCSCIndexes:
        H, W = mask.shape
        row_num = H // self._block_H
        col_num = W // self._block_W
        extra_buffer = self._get_index_extra_buffer(row_num, col_num, mask.device)
        col_ptr = torch.zeros((col_num + 1, ), dtype=torch.int32, device='cuda')
        self.index_1_kernel(
            mask, col_ptr, extra_buffer, np.int32(H), np.int32(W),
            block=self._index_block_dim, grid=(col_num, row_num, 1)
        )
        col_ptr = torch.cumsum(col_ptr, dim=0, dtype=torch.int32)
        block_nnz = col_ptr[-1].item()
        BCSC_idx = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        self.index_2_kernel(
            col_ptr, BCSC_idx, extra_buffer,
            block=(1, 1, 1), grid=(col_num, row_num, 1)
        )
        return BCSCIndexes(
            self, mask, col_ptr, BCSC_idx,
            block_nnz, self._block_H, self._block_W,
        )

    def convert(
        self,
        dense: torch.Tensor,
        index: torch.Tensor,
        block_nnz: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        dense_shape = dense.shape
        assert dense_shape[-2] == H
        assert dense_shape[-1] == W
        assert dense.dtype == torch.float32
        batch_size = int(np.prod(dense_shape[:-2]))
        sparse_val = torch.empty(
            size=(batch_size, block_nnz * self._block_size),
            dtype=dense.dtype,
            device=dense.device,
        )
        self.convert_kernel(
            dense, sparse_val, index,
            np.int32(block_nnz), np.int32(H), np.int32(W),
            block=self._convert_block_dim, grid=(block_nnz, batch_size, 1),
        )
        return sparse_val.reshape(dense_shape[:-2] + (-1, ))

    def inverse(
        self,
        sparse_val: torch.Tensor,
        index: torch.Tensor,
        block_nnz: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        sparse_shape = sparse_val.shape
        assert sparse_shape[-1] == block_nnz * self._block_size
        assert sparse_val.dtype == torch.float32
        batch_size = int(np.prod(sparse_shape[:-1]))
        dense = torch.zeros(
            size=(batch_size, H, W),
            dtype=sparse_val.dtype,
            device=sparse_val.device,
        )
        self.inverse_kernel(
            sparse_val, dense, index,
            np.int32(block_nnz), np.int32(H), np.int32(W),
            block=self._convert_block_dim, grid=(block_nnz, batch_size, 1),
        )
        return dense.reshape(sparse_shape[:-1] + (H, W))

    def row_sum(
        self,
        sparse_val: torch.Tensor,
        row_ptr: torch.Tensor,
        nnz: np.int32,
        row_num: int,
        H: int,
    ):
        sparse_shape = sparse_val.shape
        assert sparse_val.dtype == torch.float32
        batch_size = int(np.prod(sparse_shape[:-1]))
        result = torch.zeros(
            size=(batch_size, H),
            dtype=sparse_val.dtype,
            device=sparse_val.device,
        )
        self.row_sum_kernel(
            sparse_val, result, row_ptr, nnz, np.int32(H),
            block=self._sum_block_dim, grid=(row_num, batch_size, 1),
        )
        return result.reshape(sparse_shape[:-1] + (H, ))

    def col_sum(
        self,
        sparse_val: torch.Tensor,
        col_ptr: torch.Tensor,
        BCSC_index: torch.Tensor,
        nnz: np.int32,
        col_num: int,
        W: int,
    ):
        sparse_shape = sparse_val.shape
        assert sparse_val.dtype == torch.float32
        batch_size = int(np.prod(sparse_shape[:-1]))
        result = torch.zeros(
            size=(batch_size, W),
            dtype=sparse_val.dtype,
            device=sparse_val.device,
        )
        self.col_sum_kernel(
            sparse_val, result, col_ptr, BCSC_index, nnz, np.int32(W),
            block=self._sum_block_dim, grid=(col_num, batch_size, 1),
        )
        return result.reshape(sparse_shape[:-1] + (W, ))


_bcsr_function_cache: Dict[str, BCSFunctionContext] = {}


def get_bcs_function(block_H: int, block_W: int, BCSR: bool, BCSC: bool):
    bcsr_function_id = f'{block_H},{block_W},{BCSR},{BCSC}'
    if bcsr_function_id not in _bcsr_function_cache:
        _bcsr_function_cache[bcsr_function_id] = BCSFunctionContext(block_H, block_W, BCSR, BCSC)
    return _bcsr_function_cache[bcsr_function_id]
