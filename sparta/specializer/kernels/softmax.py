# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
from typing import Tuple

import jinja2

from sparta.common.tesa import BCSRH
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels import KernelBase


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'templates')


class SparseSoftmaxKernel(KernelBase):

    def __init__(self, dtype: str = 'float', compressed: bool = False):
        self._compressed = compressed
        self._dtype = dtype
        super().__init__()

    def _set_tesa(self):
        self.tesa_type['x'] = BCSRH
        self.tesa_attrs['x'] = ['row_ptr', 'col_idx']
        self.tesa_type['mask'] = BCSRH
        self.tesa_attrs['mask'] = []
        self.tesa_type['y'] = BCSRH
        self.tesa_attrs['y'] = []

    def _add_parameters(self):
        self._add_parameter('BATCH_SIZE')
        self._add_parameter('GLOBAL_H_VALUE')
        self._add_parameter('GLOBAL_W_VALUE')
        self._add_parameter('COMPRESSED', value=self._compressed)

    def set_shape(self, batch_size: int, H: int, W: int):
        self.set_parameter('BATCH_SIZE', batch_size)
        self.set_parameter('GLOBAL_H_VALUE', H)
        self.set_parameter('GLOBAL_W_VALUE', W)

    def get_shape(self):
        batch_size = self.get_parameter('BATCH_SIZE')
        H = self.get_parameter('GLOBAL_H_VALUE')
        W = self.get_parameter('GLOBAL_W_VALUE')
        return batch_size, H, W

    @abc.abstractmethod
    def get_block_shape(self) -> Tuple[int, int]:
        '''Get BH and BW.'''

    def _pre_compile(self):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        input_mask = [True, False, False, True, True]

        converter = self.tesa_type['x'](
            mask=self.get_mask('x'),
            size=(H, W),
            block_size=(BH, BW),
        )
        fixed_inputs = converter.get_attrs(self.tesa_attrs['x'])
        self._converters['x'] = converter
        self._converters['mask'] = converter
        self._converters['y'] = converter

        if self._compressed:
            output_shapes = [(batch_size, converter.get_attr('nnz').item() * BH * BW)]
        else:
            output_shapes = [(batch_size, H, W)]
        return input_mask, fixed_inputs, output_shapes


class SparTASparseSoftmaxKernel(SparseSoftmaxKernel):

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
            search_space=TunableItemCfg('choice', [32, 64, 128])
        )
        self._add_parameter(
            'ROW_TILE_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [2, 4, 8, 16])
        )

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, 'sparta_sparse_softmax.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def get_block_shape(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        return BH, BW

    def blocks_per_grid(self):
        batch_size, H, W = self.get_shape()
        T = self.get_parameter('ROW_TILE_VALUE')
        return (H // T, batch_size)

    def threads_per_block(self) -> Tuple[int]:
        T = self.get_parameter('ROW_TILE_VALUE')
        return (T * 32, )


class SparseSoftmaxBackwardKernel(KernelBase):

    def __init__(self, dtype: str = 'float', compressed: bool = False):
        self._compressed = compressed
        self._dtype = dtype
        super().__init__()

    def _set_tesa(self):
        self.tesa_type['grad_y'] = BCSRH
        self.tesa_attrs['grad_y'] = ['row_ptr', 'col_idx']
        self.tesa_type['y'] = BCSRH
        self.tesa_attrs['y'] = []
        self.tesa_type['mask'] = BCSRH
        self.tesa_attrs['mask'] = []
        self.tesa_type['grad_x'] = BCSRH
        self.tesa_attrs['grad_x'] = []

    def _add_parameters(self):
        self._add_parameter('BATCH_SIZE')
        self._add_parameter('GLOBAL_H_VALUE')
        self._add_parameter('GLOBAL_W_VALUE')
        self._add_parameter('COMPRESSED', value=self._compressed)

    def set_shape(self, batch_size: int, H: int, W: int):
        self.set_parameter('BATCH_SIZE', batch_size)
        self.set_parameter('GLOBAL_H_VALUE', H)
        self.set_parameter('GLOBAL_W_VALUE', W)

    def get_shape(self):
        batch_size = self.get_parameter('BATCH_SIZE')
        H = self.get_parameter('GLOBAL_H_VALUE')
        W = self.get_parameter('GLOBAL_W_VALUE')
        return batch_size, H, W

    @abc.abstractmethod
    def get_block_shape(self) -> Tuple[int, int]:
        '''Get BH and BW.'''

    def _pre_compile(self):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        input_mask = [True, False, False, True, True, True]

        converter = self.tesa_type['grad_y'](
            mask=self.get_mask('grad_y'),
            size=(H, W),
            block_size=(BH, BW),
        )
        fixed_inputs = converter.get_attrs(self.tesa_attrs['grad_y'])
        self._converters['grad_y'] = converter
        self._converters['y'] = converter
        self._converters['mask'] = converter
        self._converters['grad_x'] = converter

        if self._compressed:
            output_shapes = [(batch_size, converter.get_attr('nnz').item() * BH * BW)]
        else:
            output_shapes = [(batch_size, H, W)]
        return input_mask, fixed_inputs, output_shapes


class SparTASparseSoftmaxBackwardKernel(SparseSoftmaxBackwardKernel):

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
            search_space=TunableItemCfg('choice', [32, 64, 128])
        )
        self._add_parameter(
            'ROW_TILE_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [2, 4, 8, 16])
        )

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, 'sparta_sparse_softmax_backward.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def get_block_shape(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        return BH, BW

    def blocks_per_grid(self):
        batch_size, H, W = self.get_shape()
        T = self.get_parameter('ROW_TILE_VALUE')
        return (H // T, batch_size)

    def threads_per_block(self) -> Tuple[int]:
        T = self.get_parameter('ROW_TILE_VALUE')
        return (T * 32, )
