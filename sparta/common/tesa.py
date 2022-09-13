# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Dict, Tuple, Iterable, Optional, Union

import numpy as np
import torch


class TeSABase(abc.ABC):

    def __init__(self, **kwargs):
        try:
            self.dense, self.sparse = self._import_dense_data(**kwargs)
        except TypeError:
            self.dense, self.sparse = self._import_sparse_data(**kwargs)

    @staticmethod
    @abc.abstractmethod
    def desc(**kwargs) -> Dict[str, Dict]:
        '''
        Describe TeSA components
        '''

    @abc.abstractmethod
    def _import_dense_data(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Calculate TeSA variables by the dense matrix
        '''

    @abc.abstractmethod
    def _import_sparse_data(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Reconstruct the dense matrix using TeSA variables
        '''


class BCSR(TeSABase):

    @staticmethod
    def _select_vars(data: Dict, mode: str = 'H'):
        if mode.startswith('H'):
            keys = ['val', 'row_ptr', 'col_idx']
        elif mode.startswith('V'):
            keys = ['val', 'col_ptr', 'row_idx']
        else:
            keys = ['val', 'row_idx', 'col_idx', 'nnz']
        return {k: data[k] for k in keys}

    def _import_dense_data(
        self, dense: np.ndarray, block_size: Iterable[int], mode: str = 'H',
        size: Optional[Iterable[int]] = None, mask: Optional[Union[np.ndarray, float, int]] = None,
    ):
        block_height, block_width = block_size
        height, width = dense.shape if size is None else size
        row_num = height // block_height
        col_num = width // block_width

        if mask is None:
            mask = dense.reshape(row_num, block_height, col_num, block_width).swapaxes(1, 2)
            mask: np.ndarray = np.abs(mask).sum(axis=(2, 3)) > 0
        elif (isinstance(mask, float) or isinstance(mask, int)) and 0 <= mask <= 1:
            mask = np.random.uniform(size=(row_num, col_num)) < mask
        elif isinstance(mask, np.ndarray) and mask.shape == (height, width):
            mask = mask.reshape(row_num, block_height, col_num, block_width).swapaxes(1, 2)
            mask: np.ndarray = np.abs(mask).sum(axis=(2, 3)) > 0
        elif isinstance(mask, np.ndarray) and mask.shape == (row_num, col_num):
            mask = mask.astype(bool)
        else:
            raise ValueError('BCSR mask invalid')

        val, row_idx, col_idx, row_ptr, col_ptr = [], [], [], [0], [0]

        def read_block(block_i, block_j, val, row_idx, col_idx):
            block_start_i = block_i * block_height
            block_end_i = block_start_i + block_height
            block_start_j = block_j * block_width
            block_end_j = block_start_j + block_width
            if mask[block_i, block_j]:
                block = dense[block_start_i:block_end_i, block_start_j:block_end_j]
                val.append(block.flatten())
                row_idx.append(block_i)
                col_idx.append(block_j)
            else:
                dense[block_start_i:block_end_i, block_start_j:block_end_j] = 0
            return val, row_idx, col_idx

        if mode.startswith('H'):
            for block_i in range(row_num):
                for block_j in range(col_num):
                    val, row_idx, col_idx = read_block(block_i, block_j, val, row_idx, col_idx)
                row_ptr.append(len(col_idx))
        else:
            for block_j in range(col_num):
                for block_i in range(row_num):
                    val, row_idx, col_idx = read_block(block_i, block_j, val, row_idx, col_idx)
                col_ptr.append(len(row_idx))

        val = dense if mode.endswith('D') else np.concatenate(val)

        sparse = {
            'val': np.array(val).astype(dense.dtype),
            'nnz': np.array([len(row_idx)]).astype(np.int32),
            'row_idx': np.array(row_idx).astype(np.int32),
            'col_idx': np.array(col_idx).astype(np.int32),
            'row_ptr': np.array(row_ptr).astype(np.int32),
            'col_ptr': np.array(col_ptr).astype(np.int32),
        }
        return dense, BCSR._select_vars(sparse, mode)

    def _import_sparse_data(
        self, val: np.ndarray, size: Iterable[int], block_size: Iterable[int],
        row_idx: Optional[np.ndarray] = None, col_idx: Optional[np.ndarray] = None,
        row_ptr: Optional[np.ndarray] = None, col_ptr: Optional[np.ndarray] = None,
        nnz: Optional[int] = None
    ):
        block_height, block_width = block_size
        height, width = size
        row_num = height // block_height
        col_num = width // block_width

        def check_int_arr(arr: Optional[np.ndarray], key: str):
            if arr is not None:
                if len(arr.shape) != 1:
                    raise ValueError(f'BCSR {key} should be an 1D-array')
                if not str(arr.dtype).startswith('int'):
                    raise ValueError(f'BCSR {key} should be an interger array')

        check_int_arr(row_idx, 'row_idx')
        check_int_arr(col_idx, 'col_idx')
        check_int_arr(row_ptr, 'row_ptr')
        check_int_arr(col_ptr, 'col_ptr')

        if row_idx is None:
            if col_idx is None or row_ptr is None:
                raise ValueError('BCSR variable combination unrecognized')
            if nnz is None:
                nnz = col_idx.size
            elif col_idx.size != nnz:
                ValueError('BCSR col_idx.size should be equal to nnz')
            if row_ptr[-1] != nnz:
                raise ValueError('BCSR row_ptr[-1] should be equal to nnz')
            if row_ptr.size != row_num + 1:
                raise ValueError('BCSR row_ptr.size should be equal to row_num')
            row_cnt = [row_ptr[k + 1] - row_ptr[k] for k in range(row_num)]
            row_idx = np.concatenate([[k] * n for k, n in enumerate(row_cnt)])
            mode = 'H'
        elif col_idx is None:
            if row_idx is None or col_ptr is None:
                raise ValueError('BCSR variable combination unrecognized')
            if nnz is None:
                nnz = row_idx.size
            elif row_idx.size != nnz:
                ValueError('BCSR row_idx.size should be equal to nnz')
            if col_ptr[-1] != nnz:
                raise ValueError('BCSR col_ptr[-1] should be equal to nnz')
            if col_ptr.size != col_num + 1:
                raise ValueError('BCSR col_ptr.size should be equal to row_num')
            col_cnt = [col_ptr[k + 1] - col_ptr[k] for k in range(col_num)]
            col_idx = np.concatenate([[k] * n for k, n in enumerate(col_cnt)])
            mode = 'V'
        else:
            mode = 'X'

        if len(val.shape) == 1:
            if nnz * block_width * block_height != val.size:
                raise ValueError('BCSR variable size mismatches')
            block_flatten_size = block_height * block_width
            dense = np.zeros((height, width)).astype(val.dtype)
            block_i = 0
            block_cnt = 0
            block_start = 0
            for block_i, block_j in zip(row_idx, col_idx):
                row_start = block_i * block_height
                col_start = block_j * block_width
                block = val[block_start:block_start + block_flatten_size]
                block = block.reshape((block_width, block_height))
                dense[row_start:row_start + block_height, col_start:col_start + block_width] = block
                block_start += block_flatten_size
                block_cnt += 1
        elif len(val.shape) == 2:
            if val.shape != (height, width):
                raise ValueError('BCSR variable size mismatches')
            dense = val
            mode += 'D'
        else:
            raise ValueError('BCSR variable size mismatches')

        sparse = {
            'val': val,
            'nnz': np.array([nnz]).astype(np.int32),
            'row_idx': row_idx.astype(np.int32),
            'col_idx': col_idx.astype(np.int32),
            'row_ptr': row_ptr.astype(np.int32),
            'col_ptr': col_ptr.astype(np.int32),
        }
        return dense, BCSR._select_vars(sparse, mode)

    @staticmethod
    def desc(mode: str = 'H'):
        return BCSR._select_vars({
            'val': {'role': 'data'},
            'row_idx': {'role': 'tesa', 'type': 'int'},
            'col_idx': {'role': 'tesa', 'type': 'int'},
            'row_ptr': {'role': 'tesa', 'type': 'int'},
            'col_ptr': {'role': 'tesa', 'type': 'int'},
            'nnz': {'role': 'tesa', 'type': 'int'},
        }, mode)


class BCSRObj:

    def __init__(self, mode: str = 'H', block_size: Tuple[int] = None) -> None:
        self.mode = mode
        self.block_size = block_size

    def to_sparse(self, dense: torch.tensor, mask: torch.tensor):
        sp = BCSR(
            dense=dense.cpu().numpy() if dense.is_cuda else dense.numpy(),
            mask=mask.cpu().numpy() if mask.is_cuda else mask.numpy(),
            block_size=self.block_size
        ).sparse
        val = torch.from_numpy(sp['val']).cuda()
        ptr = torch.from_numpy(sp['row_ptr']).cuda()
        idx = torch.from_numpy(sp['col_idx']).cuda()
        return val, ptr, idx
