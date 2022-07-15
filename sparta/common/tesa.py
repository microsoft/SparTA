# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import numpy as np


class TeSABase(abc.ABC):

    def __init__(self, data: any, **kwargs):
        self._config = kwargs
        if (isinstance(data, np.ndarray)):
            self._dense = data
        elif (isinstance(data, dict)):
            self._verify_tesa(data)
            self._tesa = data
        else:
            raise TypeError(f'{type(data)} could not be fed into {type(self)}\'s initializer')

    def dense(self) -> 'np.ndarray':
        if not hasattr(self, '_dense') and hasattr(self, '_tesa'):
            self._dense = self._convert_tesa_to_dense(self._tesa)
        return self._dense

    def tesa(self) -> dict[str, 'np.ndarray']:
        if not hasattr(self, '_tesa') and hasattr(self, '_dense'):
            self._tesa = self._convert_dense_to_tesa(self._dense)
        return self._tesa

    @staticmethod
    @abc.abstractmethod
    def desc() -> dict[str, dict]:
        '''
        Describe TeSA components
        '''

    @abc.abstractmethod
    def _verify_tesa(self, tesa: dict[str, 'np.ndarray']):
        '''
        Raise an exception if the TeSA input is invalid.
        '''

    @abc.abstractmethod
    def _convert_tesa_to_dense(self, tesa: dict[str, 'np.ndarray']) -> 'np.ndarray':
        '''
        Convert a TeSA to a dense tensor.
        '''

    @abc.abstractmethod
    def _convert_dense_to_tesa(self, dense: 'np.ndarray', **kwargs) -> dict[str, 'np.ndarray']:
        '''
        Convert a dense tensor to a TeSA.
        '''


class BCSR(TeSABase):

    @staticmethod
    def desc():
        return {
            'val': {},
            'row': {'type': 'int'},
            'col': {'type': 'int'},
        }

    def _verify_tesa(self, tesa: dict[str, 'np.ndarray']):
        try:
            block_width = self._config['block_width']
            block_height = self._config['block_height']
        except KeyError:
            raise ValueError('BCSR block size arguments missed')
        for key in ['val', 'row', 'col']:
            if key not in tesa.keys():
                raise ValueError(f'BCSR {key} missed')
            if not isinstance(tesa[key], np.ndarray):
                raise ValueError(f'BCSR {key} should be a numpy array')
            if len(tesa[key].shape) != 1:
                raise ValueError(f'BCSR {key} should be an 1D-array')
        for key in ['row', 'col']:
            if not str(tesa[key].dtype).startswith('int'):
                raise ValueError(f'BCSR {key} should be intergers')
        if tesa['row'][-1] != tesa['col'].size:
            raise ValueError('BCSR variable size mismatches')
        if tesa['col'].size * block_width * block_height != tesa['val'].size:
            raise ValueError('BCSR variable size mismatches')

    def _convert_tesa_to_dense(self, tesa: dict[str, 'np.ndarray']) -> 'np.ndarray':
        try:
            block_width = self._config['block_width']
            block_height = self._config['block_height']
        except KeyError:
            raise ValueError('BCSR block size arguments missed')
        col_num = np.max(tesa['col']) + 1
        row_num = tesa['row'].size - 1
        height = block_height * row_num
        width = block_width * col_num
        block_size = block_height * block_width
        dense = np.zeros((height, width)).astype(tesa['val'].dtype)
        block_i = 0
        block_cnt = 0
        block_start = 0
        for block_j in tesa['col']:
            if block_cnt == tesa['row'][block_i]:
                block_i += 1
            col_start = block_j * block_width
            row_start = (block_i - 1) * block_height
            block = tesa['val'][block_start:block_start + block_size]
            block = block.reshape((block_width, block_height)).T  # TODO: Check
            dense[row_start:row_start + block_height, col_start:col_start + block_width] = block
            block_start += block_size
            block_cnt += 1
        return dense

    def _convert_dense_to_tesa(self, dense: 'np.ndarray') -> dict[str, 'np.ndarray']:
        try:
            block_width = self._config['block_width']
            block_height = self._config['block_height']
        except KeyError:
            raise ValueError('BCSR block size arguments missed')
        height, width = dense.shape
        row_num = height // block_height
        col_num = width // block_width
        if 'mask' in self._config.keys():
            if isinstance(self._config['mask'], np.ndarray) and self._config['mask'].shape == (row_num, col_num):
                mask = self._config['mask'].astype(bool)
            else:
                raise ValueError('BCSR mask invalid')
        else:
            mask = dense.reshape(row_num, block_height, col_num, block_width).swapaxes(1,2)
            mask = np.abs(mask).sum(axis=(2, 3)) > 0
        col = []
        row = []
        val = np.array([])
        for block_i in range(row_num):
            block_start_i = block_i * block_height
            block_end_i = block_start_i + block_height
            for block_j in range(col_num):
                block_start_j = block_j * block_width
                block_end_j = block_start_j + block_width
                if mask[block_i, block_j]:
                    col.append(block_j)
                    block = dense[block_start_i:block_end_i, block_start_j:block_end_j].T  # TODO: Check
                    val = np.concatenate([val, block.flatten()])
                else:
                    dense[block_start_i:block_end_i, block_start_j:block_end_j] = 0
            row.append(len(col))
        row = [0] + row
        return {
            'col': np.array(col).astype(np.int32),
            'row': np.array(row).astype(np.int32),
            'val': val.astype(dense.dtype)
        }
