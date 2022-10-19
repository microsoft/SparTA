# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Dict, List, Tuple, Callable

import torch
import numpy as np

class TeSAConverter(Callable):

    def __init__(self):
        self._attrs: Dict[str, torch.Tensor] = {}

    def _set_attr(self, name: str, value: torch.Tensor):
        self._attrs[name] = value

    def get_attr(self, name: str):
        return self._attrs[name]

    def get_attrs(self, names: List[str]):
        return [self._attrs[name] for name in names]

    def get_attr_names(self):
        return self._attrs.keys()

    @abc.abstractmethod
    def convert(self, dense: torch.Tensor) -> torch.Tensor:
        '''Convert dense tensor to compressed sparse value.'''

    @abc.abstractmethod
    def get_mask(self) -> torch.Tensor:
        '''Get the mask actually used.'''

    def __call__(self, dense: torch.Tensor) -> torch.Tensor:
        return self.convert(dense)


class BCSR(TeSAConverter):

    def __init__(self, mask: torch.Tensor, size: Tuple[int, int], block_size: Tuple[int, int]):
        super().__init__()
        self._H, self._W = size
        self._BH, self._BW = block_size
        row_num = self._H // self._BH
        col_num = self._W // self._BW
        self._block_mask = mask.reshape((row_num, self._BH, col_num, self._BW))
        self._block_mask = self._block_mask.swapaxes(1, 2).any(dim=-1).any(dim=-1)
        attr_names = ('row_idx', 'col_idx', 'row_ptr', 'col_ptr', 'nnz')
        for name, value in zip(attr_names, self.read_block_mask(self._block_mask)):
            self._set_attr(name, torch.tensor(value, dtype=torch.int32))
        row_idx = self.get_attr('row_idx')
        col_idx = self.get_attr('col_idx')
        self._H_order = torch.argsort(row_idx * col_num + col_idx)
        self._V_order = torch.argsort(col_idx * row_num + row_idx)
        self.is_H_main = torch.all(torch.diff(self._H_order) >= 0)

    def get_mask(self):
        mask = self._block_mask.reshape((self._H // self._BH, 1, self._W // self._BW, 1))
        return mask.tile((1, self._BH, 1, self._BW)).reshape((self._H, self._W))

    @abc.abstractmethod
    def read_block_mask(
        self, block_mask: torch.Tensor
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        '''Convert block mask to BCSR parameters.'''

    def convert(self, dense: torch.Tensor):
        # TODO: use CUDA kernel
        dense_shape = dense.shape
        assert dense_shape[-2] == self._H
        assert dense_shape[-1] == self._W
        batch_size = int(np.prod(dense_shape[:-2]))
        sparse_shape = [batch_size, -1]
        dense = dense.reshape((batch_size, self._H, self._W))
        sparse_val = []
        for batch in range(batch_size):
            for row, col in zip(self.get_attr('row_idx'), self.get_attr('col_idx')):
                block_start_i = row * self._BH
                block_end_i = block_start_i + self._BH
                block_start_j = col * self._BW
                block_end_j = block_start_j + self._BW
                block = dense[batch, block_start_i:block_end_i, block_start_j:block_end_j]
                sparse_val.append(block.flatten())
        # Note: torch.stack() may lose data here
        return torch.stack(sparse_val).reshape(sparse_shape).contiguous()

    def swapaxes(self, sparse_val: torch.Tensor):
        # TODO: use CUDA kernel
        sparse_shape = sparse_val.shape
        batch_size = int(np.prod(sparse_shape[:-1]))
        val = sparse_val.reshape((batch_size, -1, self._BH * self._BW))
        if self.is_H_main:
            val = val[:, self._V_order, :]
        else:
            val = val[:, self._H_order, :]
        return val.reshape(sparse_shape).contiguous()

    def sum(self, sparse_val: torch.Tensor, axis: int):
        sparse_shape = sparse_val.shape
        batch_size = int(np.prod(sparse_shape[:-1]))
        sparse_val = sparse_val.detach().reshape((batch_size, -1, self._BH, self._BW))
        if axis > 0:
            axis -= len(sparse_shape) + 1
        assert axis in [-1, -2], 'invalid axis'
        sparse_val = sparse_val.sum(dim=axis)
        if axis == -1:
            sum_val = torch.zeros(size=(batch_size, self._H), device=sparse_val.device)
            for k, row in enumerate(self.get_attr('row_idx')):
                segment_start = row * self._BH
                segment_end = segment_start + self._BH
                sum_val[:, segment_start:segment_end] += sparse_val[:, k, :]
        else:
            sum_val = torch.zeros(size=(batch_size, self._W), device=sparse_val.device)
            for k, col in enumerate(self.get_attr('col_idx')):
                segment_start = col * self._BW
                segment_end = segment_start + self._BW
                sum_val[:, segment_start:segment_end] += sparse_val[:, k, :]
        return sum_val


class BCSRH(BCSR):

    def read_block_mask(self, block_mask: torch.Tensor):
        row_idx, col_idx, row_ptr, col_ptr = [], [], [0], []
        for block_i in range(block_mask.shape[0]):
            for block_j in range(block_mask.shape[1]):
                if block_mask[block_i, block_j].item():
                    row_idx.append(block_i)
                    col_idx.append(block_j)
            row_ptr.append(len(col_idx))
        nnz = row_ptr[-1:]
        return row_idx, col_idx, row_ptr, col_ptr, nnz


class BCSRV(BCSR):

    def read_block_mask(self, block_mask: torch.Tensor):
        row_idx, col_idx, row_ptr, col_ptr = [], [], [], [0]
        for block_j in range(block_mask.shape[1]):
            for block_i in range(block_mask.shape[0]):
                if block_mask[block_i, block_j].item():
                    row_idx.append(block_i)
                    col_idx.append(block_j)
            col_ptr.append(len(row_idx))
        nnz = col_ptr[-1:]
        return row_idx, col_idx, row_ptr, col_ptr, nnz
