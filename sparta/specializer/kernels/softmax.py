# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List, Tuple, Callable

import torch
import jinja2

from sparta.common.tesa import BCSRH
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels import KernelBase, PortConfig
from sparta.testing import sparse_softmax_forward_reference, sparse_softmax_backward_reference


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'templates')


class SparseSoftmaxKernel(KernelBase):

    def __init__(self, compressed: bool = False, dtype: str = 'float'):
        self._compressed = compressed
        self._dtype = dtype
        super().__init__()

    def _set_ports(self):
        for port in self.ports.values():
            port.set_tesa(BCSRH, [
                'GLOBAL_H_VALUE', 'GLOBAL_W_VALUE',
                'BLOCK_SIZE_H_VALUE', 'BLOCK_SIZE_W_VALUE'
            ])

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

    def get_block_shape(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        return BH, BW


class SparseSoftmaxForwardKernel(SparseSoftmaxKernel):

    def set_port_params(self):
        self.ports['x'].set_params(self.get_parameters())

    def _set_ports(self):
        self.ports['x'] = PortConfig(name='x', is_input=True)
        self.ports['y'] = PortConfig(name='y', is_input=False)
        super()._set_ports()
        self.ports['x'].connect(self.ports['y'])

    def _set_func_call(self, kernel_func_call: Callable):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        converter = self.get_converter('x')
        row_ptr = converter.get_attr('row_ptr')
        col_idx = converter.get_attr('col_idx')
        block_nnz = converter.get_attr('nnz').item()
        shape = (batch_size, block_nnz * BH * BW) if self._compressed else (batch_size, H, W)
        mask = self.get_mask('x').unsqueeze(0).tile([batch_size, 1, 1]).to(torch.float32)
        if self._compressed:
            mask = self.get_converter('x').convert(mask)
        block = self.threads_per_block()
        grid = self.blocks_per_grid()

        def softmax_forward_func(x, T):
            y = torch.zeros(shape, device=x.device)
            kernel_func_call(x, row_ptr, col_idx, mask, T, y, block=block, grid=grid)
            return y

        return softmax_forward_func

    def _convert_data(self, inputs, outputs):
        inputs[0] = inputs[0].reshape(self.get_shape())
        outputs[0] = outputs[0].reshape(self.get_shape())
        if self._compressed:
            inputs[0] = self.get_converter('x').convert(inputs[0])
            outputs[0] = self.get_converter('y').convert(outputs[0])

    def reference(self, *args):
        x, T = args
        mask = self.get_mask('x')
        y = sparse_softmax_forward_reference(x, mask, 1 / T)
        return y


class SparseSoftmaxBackwardKernel(SparseSoftmaxKernel):

    def set_port_params(self):
        self.ports['grad_y'].set_params(self.get_parameters())

    def _set_ports(self):
        self.ports['grad_y'] = PortConfig(name='grad_y', is_input=True)
        self.ports['y'] = PortConfig(name='y', is_input=True)
        self.ports['grad_x'] = PortConfig(name='grad_x', is_input=False)
        super()._set_ports()
        self.ports['grad_y'].connect(self.ports['y'])
        self.ports['grad_y'].connect(self.ports['grad_x'])

    def _set_func_call(self, kernel_func_call: Callable):
        batch_size, H, W = self.get_shape()
        BH, BW = self.get_block_shape()

        converter = self.get_converter('grad_y')
        row_ptr = converter.get_attr('row_ptr')
        col_idx = converter.get_attr('col_idx')
        block_nnz = converter.get_attr('nnz').item()
        shape = (batch_size, block_nnz * BH * BW) if self._compressed else (batch_size, H, W)
        mask = self.get_mask('grad_y').unsqueeze(0).tile([batch_size, 1, 1]).to(torch.float32)
        if self._compressed:
            mask = self.get_converter('grad_y').convert(mask)
        block = self.threads_per_block()
        grid = self.blocks_per_grid()

        def softmax_backward_func(grad_y, y, T):
            x = torch.zeros(shape, device=grad_y.device)
            kernel_func_call(grad_y, row_ptr, col_idx, y, mask, T, x, block=block, grid=grid)
            return x

        return softmax_backward_func

    def _convert_data(self, inputs, outputs):
        inputs[0] = inputs[0].reshape(self.get_shape())
        inputs[1] = inputs[1].reshape(self.get_shape())
        outputs[0] = outputs[0].reshape(self.get_shape())
        if self._compressed:
            inputs[0] = self.get_converter('grad_y').convert(inputs[0])
            inputs[1] = self.get_converter('y').convert(inputs[1])
            outputs[0] = self.get_converter('grad_x').convert(outputs[0])

    def reference(self, *args):
        grad_y, y, T = args
        mask = self.get_mask('grad_y')
        grad_x = sparse_softmax_backward_reference(grad_y, y, mask, 1 / T)
        return grad_x


class SparTASoftmaxKernel(SparseSoftmaxKernel):

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


class SparTASparseSoftmaxForwardKernel(SparseSoftmaxForwardKernel, SparTASoftmaxKernel):

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, 'sparta_sparse_softmax_forward.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())


class SparTASparseSoftmaxBackwardKernel(SparseSoftmaxBackwardKernel, SparTASoftmaxKernel):

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, 'sparta_sparse_softmax_backward.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())
