# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import textwrap
import importlib.resources as res
from typing import Any, Dict, Tuple

import torch
import jinja2
import numpy as np
import pandas as pd

from sparta.tuning import TunableItemCfg
from sparta.specializer.kernels import templates, look_up_tables
from sparta.specializer.kernels.kernel_base import KernelBase


def _get_softmax_lut(impl: str, direction: str):
    major, minor = torch.cuda.get_device_capability()
    try:
        lut_file = f'softmax.{direction}.{impl}.{major}{minor}.csv'
        lut_text = res.read_text(look_up_tables, lut_file)
    except FileNotFoundError:
        lut_file = f'softmax.{direction}.{impl}.default.csv'
        lut_text = res.read_text(look_up_tables, lut_file)
    return pd.read_csv(io.StringIO(lut_text))


_SOFTMAX_LUTS = {
    'sparta': {
        'forward': _get_softmax_lut('sparta', 'forward'),
        'backward': _get_softmax_lut('sparta', 'backward'),
    },
}


class SparseSoftmaxKernel(KernelBase):

    __algo__: str = ''
    __direction__: str = ''

    def __init__(self, compressed: bool, batched: bool, dtype: str = 'float'):
        self._compressed = compressed
        self._batched = batched
        self._dtype = dtype
        self._lut = _SOFTMAX_LUTS[self.__algo__][self.__direction__]
        super().__init__()

    def _add_parameters(self):
        self._add_parameter('BATCHED', value=self._batched)
        self._add_parameter('COMPRESSED', value=self._compressed)

    def get_block_shape(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        return BH, BW


class SparseSoftmaxForwardKernel(SparseSoftmaxKernel):

    __direction__ = 'forward'


class SparseSoftmaxBackwardKernel(SparseSoftmaxKernel):

    __direction__ = 'backward'


class SparTASoftmaxKernel(SparseSoftmaxKernel):

    __algo__ = 'sparta'

    def __init__(self, compressed: bool, batched: bool, dtype: str = 'float'):
        super().__init__(compressed, batched, dtype)
        self._lut: pd.DataFrame = None

    def _add_parameters(self):
        super()._add_parameters()
        self._add_parameter(
            'BLOCK_SIZE_H_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128])
        )
        self._add_parameter(
            'BLOCK_SIZE_W_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128])
        )
        self._add_parameter('ROW_TILE_VALUE')
        self._add_parameter('MAX_W_VALUE', value=1024)

    def threads_per_block(self) -> Tuple[int]:
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        RT = self.get_parameter('ROW_TILE_VALUE')
        return (RT * min(BW, 32), 1, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        BH = params['BLOCK_SIZE_H_VALUE']
        BW = params['BLOCK_SIZE_W_VALUE']
        assert BH & (BH - 1) == 0
        assert BW & (BW - 1) == 0
        if 'ROW_TILE_VALUE' in params:
            RT = params['ROW_TILE_VALUE']
            assert BH >= RT
        else:
            BH, BW = self.get_block_shape()
            BH_filter = self._lut['BH'] == BH
            BW_filter = self._lut['BW'] == BW
            row = self._lut[BH_filter & BW_filter]
            assert len(row) > 0, f'block shape ({BH}, {BW}) not found in LUT'
            row = row.reset_index(drop=True).iloc[0, :]
            assert float(row['latency']) < float('inf'), f'block shape ({BH}, {BW}) is invalid'
            self.set_parameter('ROW_TILE_VALUE', int(row['RT']))
            self.estimated_latency_per_flop = row['latency'] / BH / BW

    def get_kernel_code(self):
        template_file = f'{self.__algo__}_sparse_softmax_{self.__direction__}.cuh.j2'
        kernel_template = res.read_text(templates, template_file)
        return jinja2.Template(kernel_template).render(self.get_parameters())


class SparTASparseSoftmaxForwardKernel(SparseSoftmaxForwardKernel, SparTASoftmaxKernel):

    def set_kernel_call(self, shape: Tuple[int, int, int], sparse_attr: Any):
        batch, H, W = shape
        H_32, W_32 = np.int32(H), np.int32(W)
        BH, BW = self.get_block_shape()
        RT = self.get_parameter('ROW_TILE_VALUE')
        block = self.threads_per_block()
        row_num = H // RT
        raw_func = self._kernel
        zeros = torch.zeros

        func_code = jinja2.Template(textwrap.dedent('''
            def softmax_forward_func(x, mask, T):
                {% if BATCHED %}
                batch = x.shape[0]
                    {% if COMPRESSED %}
                y = zeros((batch, sparse_attr.indexes.block_nnz * BH * BW), device=x.device)
                    {% else %}
                y = zeros((batch, H, W), device=x.device)
                    {% endif %}
                {% else %}
                    {% if COMPRESSED %}
                y = zeros((sparse_attr.indexes.block_nnz * BH * BW), device=x.device)
                    {% else %}
                y = zeros((H, W), device=x.device)
                    {% endif %}
                {% endif %}
                raw_func(
                    x.detach(), mask, T, y,
                    sparse_attr.indexes.row_ptr, sparse_attr.indexes.BCSR_idx,
                    H_32, W_32,
                    block=block,
                    grid=(row_num, {% if BATCHED %}batch{% else %}1{% endif %}, 1),
                )
                return y
        ''')).render(self.get_parameters())

        exec(func_code, locals())
        self._func = locals()['softmax_forward_func']


class SparTASparseSoftmaxBackwardKernel(SparseSoftmaxBackwardKernel, SparTASoftmaxKernel):

    def set_kernel_call(self, shape: Tuple[int, int, int], sparse_attr: Any):
        batch, H, W = shape
        H_32, W_32 = np.int32(H), np.int32(W)
        BH, BW = self.get_block_shape()
        RT = self.get_parameter('ROW_TILE_VALUE')
        block = self.threads_per_block()
        row_num = H // RT
        raw_func = self._kernel
        zeros = torch.zeros

        func_code = jinja2.Template(textwrap.dedent('''
            def softmax_backward_func(gy, y, mask, T):
                {% if BATCHED %}
                batch = gy.shape[0]
                    {% if COMPRESSED %}
                gx = zeros((batch, sparse_attr.indexes.block_nnz * BH * BW), device=gy.device)
                    {% else %}
                gx = zeros((batch, H, W), device=gy.device)
                    {% endif %}
                {% else %}
                    {% if COMPRESSED %}
                gx = zeros((sparse_attr.indexes.block_nnz * BH * BW), device=gy.device)
                    {% else %}
                gx = zeros((H, W), device=gy.device)
                    {% endif %}
                {% endif %}
                raw_func(
                    gy, y.detach(), mask, T, gx,
                    sparse_attr.indexes.row_ptr, sparse_attr.indexes.BCSR_idx,
                    H_32, W_32,
                    block=block,
                    grid=(row_num, {% if BATCHED %}batch{% else %}1{% endif %}, 1),
                )
                return gx
        ''')).render(self.get_parameters())

        exec(func_code, locals())
        self._func = locals()['softmax_backward_func']
