# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Tuple
import importlib.resources as res

import torch
import jinja2
import numpy as np

from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels import templates
from sparta.specializer.kernels.kernel_base import KernelBase, PortConfig
from sparta.testing import sparse_softmax_forward_reference, sparse_softmax_backward_reference


class SparseSoftmaxKernel(KernelBase):

    __algo__: str = ''
    __direction__: str = ''

    def __init__(self, compressed: bool = False, dtype: str = 'float'):
        self._compressed = compressed
        self._dtype = dtype
        super().__init__()

    def _add_parameters(self):
        self._add_parameter('BATCH_SIZE')
        self._add_parameter('GLOBAL_H_VALUE')
        self._add_parameter('GLOBAL_W_VALUE')
        self._add_parameter('COMPRESSED', value=self._compressed)

    def set_parameters(self, params: Dict[str, Any]):
        super().set_parameters(params)
        sparse_port = self.ports['y']
        BH, BW = self.get_block_shape()
        sparse_port.set_block_size(BH, BW)

    def set_shape(self, batch_size: int, H: int, W: int):
        self.set_parameter('BATCH_SIZE', batch_size)
        self.set_parameter('GLOBAL_H_VALUE', H)
        self.set_parameter('GLOBAL_W_VALUE', W)

    def get_shape(self):
        batch_size = self.get_parameter('BATCH_SIZE')
        H = self.get_parameter('GLOBAL_H_VALUE')
        W = self.get_parameter('GLOBAL_W_VALUE')
        return batch_size, H, W

    def get_block_shape(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        return BH, BW


class SparseSoftmaxForwardKernel(SparseSoftmaxKernel):

    __direction__ = 'forward'

    def _set_ports(self):
        self.ports['x'] = PortConfig(name='x', is_input=True, is_sparse=True, BCSR=True)
        self.ports['y'] = PortConfig(name='y', is_input=False, is_sparse=True, BCSR=True)
        self.ports['x'].connect(self, 'y')

    def update_func(self):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        indexes = self.ports['y'].indexes
        row_ptr = indexes.row_ptr
        BCSR_idx = indexes.BCSR_idx
        if self._compressed:
            shape = (batch_size, indexes.block_nnz * BH * BW)
        else:
            shape = (batch_size, H, W)
        mask = indexes.raw_mask
        if self._compressed:
            mask = indexes.convert(mask.to(torch.float32)).to(torch.uint8)
        block = self.threads_per_block()
        grid = self.blocks_per_grid()
        raw_func = self._kernel

        def softmax_forward_func(x: torch.Tensor, T: np.int32):
            y = torch.zeros(shape, device=x.device)
            raw_func(x, row_ptr, BCSR_idx, mask, T, y, block=block, grid=grid)
            return y

        self._func = softmax_forward_func

    def _convert_data(self, inputs, outputs):
        mask = self.ports['y'].mask
        inputs[0] = (inputs[0].reshape(self.get_shape()) * mask).detach()
        outputs[0] = (outputs[0].reshape(self.get_shape()) * mask).detach()
        if self._compressed:
            indexes = self.ports['y'].indexes
            inputs[0] = indexes.convert(inputs[0])
            outputs[0] = indexes.convert(outputs[0])

    def reference(self, *args):
        x, T = args
        mask = self.ports['y'].mask
        y = sparse_softmax_forward_reference(x, mask, 1 / T)
        return y


class SparseSoftmaxBackwardKernel(SparseSoftmaxKernel):

    __direction__ = 'backward'

    def _set_ports(self):
        self.ports['grad_y'] = PortConfig(name='grad_y', is_input=True, is_sparse=True, BCSR=True)
        self.ports['y'] = PortConfig(name='y', is_input=True, is_sparse=True, BCSR=True)
        self.ports['grad_x'] = PortConfig(name='grad_x', is_input=False, is_sparse=True, BCSR=True)
        self.ports['grad_y'].connect(self, 'y')
        self.ports['grad_y'].connect(self, 'grad_x')

    def update_func(self):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        indexes = self.ports['y'].indexes
        row_ptr = indexes.row_ptr
        BCSR_idx = indexes.BCSR_idx
        if self._compressed:
            shape = (batch_size, indexes.block_nnz * BH * BW)
        else:
            shape = (batch_size, H, W)
        mask = indexes.raw_mask
        if self._compressed:
            mask = indexes.convert(mask.to(torch.float32)).to(torch.uint8)
        block = self.threads_per_block()
        grid = self.blocks_per_grid()
        raw_func = self._kernel

        def softmax_backward_func(grad_y: torch.Tensor, y: torch.Tensor, T: np.int32):
            x = torch.zeros(shape, device=grad_y.device)
            raw_func(grad_y, row_ptr, BCSR_idx, y, mask, T, x, block=block, grid=grid)
            return x

        self._func = softmax_backward_func

    def _convert_data(self, inputs, outputs):
        mask = self.ports['y'].mask
        inputs[0] = (inputs[0].reshape(self.get_shape()) * mask).detach()
        inputs[1] = (inputs[1].reshape(self.get_shape()) * mask).detach()
        outputs[0] = (outputs[0].reshape(self.get_shape()) * mask).detach()
        if self._compressed:
            indexes = self.ports['y'].indexes
            inputs[0] = indexes.convert(inputs[0])
            inputs[1] = indexes.convert(inputs[1])
            outputs[0] = indexes.convert(outputs[0])

    def reference(self, *args):
        grad_y, y, T = args
        mask = self.ports['y'].indexes.raw_mask
        grad_x = sparse_softmax_backward_reference(grad_y, y, mask, 1 / T)
        return grad_x


class SparTASoftmaxKernel(SparseSoftmaxKernel):

    __algo__ = 'sparta'

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
            search_space=TunableItemCfg('choice', [16, 32, 64, 128])
        )
        self._add_parameter(
            'ROW_TILE_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [1, 2, 4, 8, 16])
        )

    def blocks_per_grid(self):
        batch_size, H, W = self.get_shape()
        RT = self.get_parameter('ROW_TILE_VALUE')
        return (H // RT, batch_size, 1)

    def threads_per_block(self) -> Tuple[int]:
        RT = self.get_parameter('ROW_TILE_VALUE')
        return (RT * 32, 1, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        BH = params['BLOCK_SIZE_H_VALUE']
        BW = params['BLOCK_SIZE_W_VALUE']
        RT = params['ROW_TILE_VALUE']
        assert BH > RT

    def get_kernel_code(self):
        template_file = f'{self.__algo__}_sparse_softmax_{self.__direction__}.cuh.j2'
        kernel_template = res.read_text(templates, template_file)
        return jinja2.Template(kernel_template).render(self.get_parameters())


class SparTASparseSoftmaxForwardKernel(SparseSoftmaxForwardKernel, SparTASoftmaxKernel):

    pass


class SparTASparseSoftmaxBackwardKernel(SparseSoftmaxBackwardKernel, SparTASoftmaxKernel):

    pass
