# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
from typing import Tuple, Dict

import jinja2
import numpy as np

from sparta.specializer.kernels.kernel_base import KernelBase
from sparta.common.tuning import TunableItemCfg


TEMPLATE_DIR = os.path.join(os.path.split(
    os.path.realpath(__file__))[0], "templates")


class SoftmaxKernelBase(KernelBase):

    def __init__(self, batch_size: int = 1, dtype: str = 'float', compressed: bool = True):
        self._dtype = dtype
        self._batch_size = batch_size
        self._compressed = compressed
        super().__init__()

    def get_kernel_name(self) -> str:
        c_str = '_c' if self._compressed else ''
        return f'sparse_softmax{c_str}'

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        '''
        Get CUDA code of the kernel
        '''

    def add_parameters(self):
        self.add_parameter("GLOBAL_H_VALUE")
        self.add_parameter("GLOBAL_W_VALUE")
        self.add_parameter("COMPRESSED", value=self._compressed)

    @abc.abstractmethod
    def check_parameters(self):
        '''
        Check if parameters are valid
        '''

    def add_ports(self):
        self.add_input('C_in', self._dtype, 'BCSR')
        self.add_input('C_mask', 'int', 'BCSR', default_val=1)
        self.add_output('C_out', self._dtype, 'BCSR')

    @abc.abstractmethod
    def set_ports_shape(self):
        H = self.get_parameter('GLOBAL_H_VALUE')
        W = self.get_parameter('GLOBAL_W_VALUE')
        self.set_input_shape('C_in', (self._batch_size, H, W))
        self.set_input_shape('C_mask', (self._batch_size, H, W))
        self.set_output_shape('C_out', (self._batch_size, H, W))

    @abc.abstractmethod
    def set_ports_layout(self):
        '''
        Set layout configs of inputs and outputs using determined parameters
        '''

    @abc.abstractmethod
    def blocks_per_grid(self) -> Tuple[int]:
        '''
        Get launch config: number of blocks per grid
        '''

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
        '''
        Get launch config: number of threads per block
        '''

    def calc_target_outputs(self) -> Dict[str, np.ndarray]:
        C_in = np.concatenate(self.get_input('C_in').dense())
        C_mask = np.concatenate(self.get_input('C_mask').dense())
        C_max = C_in.max(axis=-1).reshape((-1, 1))
        C_exp = np.exp(C_in - C_max) * C_mask
        C_exp_sum = C_exp.sum(axis=-1).reshape((-1, 1)) + 1e-10
        C_out = C_exp / C_exp_sum
        C_out = C_out.reshape((self._batch_size, -1, C_in.shape[1])).astype(C_in.dtype)
        self.set_target_output('C_out', C_out)


class SparTATemplateSparseSoftmaxKernel(SoftmaxKernelBase):

    def get_kernel_name(self) -> str:
        return f'sparta_{super().get_kernel_name()}'

    def get_kernel_code(self) -> str:
        with open(os.path.join(TEMPLATE_DIR, f'sparta_sparse_softmax.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def add_parameters(self):
        super().add_parameters()
        self.add_parameter("BLOCK_SIZE_H_VALUE", is_tunable=True, search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128]))
        self.add_parameter("BLOCK_SIZE_W_VALUE", is_tunable=True, search_space=TunableItemCfg('choice', [32, 64, 128]))
        self.add_parameter("ROW_TILE_VALUE", is_tunable=True, search_space=TunableItemCfg('choice', [2, 4, 8, 16]))

    def check_parameters(self):
        H = self.get_parameter('GLOBAL_H_VALUE')
        W = self.get_parameter('GLOBAL_W_VALUE')
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        T = self.get_parameter('ROW_TILE_VALUE')
        assert np.log2(BH) % 1 == 0
        assert np.log2(BW) % 1 == 0
        assert np.log2(T) % 1 == 0
        assert H % BH == 0
        assert W % BW == 0
        assert BH % T == 0
        assert H > BH
        assert W > BW
        assert BH > T

    def set_ports_layout(self):
        BH = self.get_parameter('BLOCK_SIZE_H_VALUE')
        BW = self.get_parameter('BLOCK_SIZE_W_VALUE')
        self.set_input_layout('C_in', {
            'mode': 'H' if self._compressed else 'HD',
            'block_size': [BH, BW],
        })
        self.set_input_layout('C_mask', self.get_input('C_in'))
        self.set_output_layout('C_out', self.get_input('C_in'))

    def blocks_per_grid(self) -> Tuple[int]:
        H = self.get_parameter('GLOBAL_H_VALUE')
        T = self.get_parameter('ROW_TILE_VALUE')
        return (H // T, self._batch_size)

    def threads_per_block(self) -> Tuple[int]:
        T = self.get_parameter('ROW_TILE_VALUE')
        return (T * 32, )
