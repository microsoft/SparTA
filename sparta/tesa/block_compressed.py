# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
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


class BCSRIndexes(TeSAIndexes):

    def __init__(
        self,
        function_context: BCSRFunctionContext,
        raw_mask: torch.Tensor,
        row_ptr: torch.Tensor,
        col_ptr: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        BCSC_order: torch.Tensor,
        block_nnz: int,
        block_H: int,
        block_W: int,
    ):
        super().__init__(function_context, raw_mask)
        # BCSR/BCSC index tensors & int32 parameters for op kernels
        self.row_ptr = row_ptr
        self.col_ptr = col_ptr
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.BCSC_order = BCSC_order
        self.nnz = np.int32(block_nnz)
        # Shape variables
        self.block_H = block_H
        self.block_W = block_W
        self.H, self.W = raw_mask.shape
        self.row_num = self.H // self.block_H
        self.col_num = self.W // self.block_W
        self.block_nnz = block_nnz

    def get_block_mask(self):
        block_mask_val = torch.ones(
            size=(self.block_nnz, ),
            dtype=torch.uint8,
            device=self.col_idx.device
        )
        return torch.sparse_csr_tensor(self.row_ptr, self.col_idx, block_mask_val)

    def get_mask(self) -> torch.Tensor:
        mask = self.get_block_mask().unsqueeze(0).unsqueeze(0)
        mask = mask.tile((self.block_H, self.block_W, 1, 1)).swapaxes(1, 2)
        return mask.reshape(self.raw_mask.shape)

    def convert(self, dense: torch.Tensor):
        return self.function_context.convert(
            dense, self.row_idx, self.col_idx,
            self.block_nnz, self.H, self.W
        )

    def inverse(self, sparse_val: torch.Tensor):
        return self.function_context.inverse(
            sparse_val, self.row_idx, self.col_idx,
            self.block_nnz, self.H, self.W
        )

    def sum(self, sparse_val: torch.Tensor, axis: int = -1) -> torch.Tensor:
        inverse_axis = -axis if axis < 0 else len(sparse_val.shape) - axis
        if inverse_axis == 1:
            return self.function_context.row_sum(
                sparse_val, self.row_ptr, self.nnz,
                self.row_num, self.H,
            )
        elif inverse_axis == 2:
            return self.function_context.col_sum(
                sparse_val, self.col_ptr, self.BCSC_order, self.nnz,
                self.col_num, self.W,
            )
        else:
            raise ValueError(f'axis #{axis} is not sparsed, please use torch.sum() instead.')


_extra_buffer_cache: Dict[str, torch.Tensor] = {}


class BCSRFunctionContext(TeSAFunctionContext):

    def __init__(self, block_H: int, block_W: int):
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
                'BH': block_H, 'BW': block_W,
            })
        source_module = SourceModule(kernel_code, options=['-O3'])
        self.index_1_kernel = source_module.get_function('bcsr_index_1')
        self.index_2_kernel = source_module.get_function('bcsr_index_2')
        self.convert_kernel = source_module.get_function('dense_to_bcsr_val')
        self.inverse_kernel = source_module.get_function('bcsr_val_to_dense')
        self.row_sum_kernel = source_module.get_function('bcsr_val_sum_row')
        self.col_sum_kernel = source_module.get_function('bcsr_val_sum_col')
    
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

    def build_indexes(self, mask: torch.Tensor) -> BCSRIndexes:
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
        row_idx = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        col_idx = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        BCSC_order = torch.empty((block_nnz, ), dtype=torch.int32, device='cuda')
        self.index_2_kernel(
            row_ptr, col_ptr, extra_buffer, row_idx, col_idx, BCSC_order,
            block=(1, 1, 1), grid=(col_num, row_num, 1)
        )
        return BCSRIndexes(
            self, mask,
            row_ptr, col_ptr, row_idx, col_idx, BCSC_order,
            block_nnz, self._block_H, self._block_W,
        )

    def convert(
        self,
        dense: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
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
            dense, sparse_val, row_idx, col_idx,
            np.int32(block_nnz), np.int32(H), np.int32(W),
            block=self._convert_block_dim, grid=(block_nnz, batch_size, 1),
        )
        return sparse_val.reshape(dense_shape[:-2] + (-1, ))

    def inverse(
        self,
        sparse_val: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
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
            sparse_val, dense, row_idx, col_idx,
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
        BCSC_order: torch.Tensor,
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
            sparse_val, result, col_ptr, BCSC_order, nnz, np.int32(W),
            block=self._sum_block_dim, grid=(col_num, batch_size, 1),
        )
        return result.reshape(sparse_shape[:-1] + (W, ))


_bcsr_function_cache: Dict[str, BCSRFunctionContext] = {}


def get_bcsr_function(block_H: int, block_W: int):
    bcsr_function_id = f'{block_H},{block_W}'
    if bcsr_function_id not in _bcsr_function_cache:
        _bcsr_function_cache[bcsr_function_id] = BCSRFunctionContext(block_H, block_W)
    return _bcsr_function_cache[bcsr_function_id]
