# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import abc
from typing import Tuple

import jinja2

from sparta.common.tesa import BCSRH, BCSRV
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels import KernelBase


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'templates')


class SparseMatMulKernel(KernelBase):

    def __init__(
        self, sparse_type: str, dtype: str = 'float', biased: bool = True, 
        transpose_A: bool = False, transpose_B: bool = True, 
        compressed: bool = True, bcsr_main: str = 'H'
    ):
        if sparse_type not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse type: {sparse_type}')
        self._biased = biased
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._compressed = compressed
        self._bcsr_main = bcsr_main
        self._stype = sparse_type
        self._dtype = dtype
        super().__init__()

    def set_tesa(self):
        if self._stype == 'sdd':
            if self._transpose_A:
                self.tesa_type['A'] = BCSRV
                self.tesa_attrs['A'] = ['col_ptr', 'row_idx']
            else:
                self.tesa_type['A'] = BCSRH
                self.tesa_attrs['A'] = ['row_ptr', 'col_idx']
        elif self._stype == 'dsd':
            if self._transpose_B:
                self.tesa_type['B'] = BCSRH
                self.tesa_attrs['B'] = ['row_ptr', 'col_idx']
            else:
                self.tesa_type['B'] = BCSRV
                self.tesa_attrs['B'] = ['col_ptr', 'row_idx']
        elif self._stype == 'dds':
            if self._bcsr_main == 'H':
                self.tesa_type['C'] = BCSRH
            else:
                self.tesa_type['C'] = BCSRV
            self.tesa_attrs['C'] = ['row_idx', 'col_idx', 'nnz']

    def add_parameters(self):
        self.add_parameter('BATCH_SIZE')
        self.add_parameter('GLOBAL_M_VALUE')
        self.add_parameter('GLOBAL_K_VALUE')
        self.add_parameter('GLOBAL_N_VALUE')
        self.add_parameter('BIASED', value=self._biased)
        self.add_parameter('TRANSPOSE_A', value=self._transpose_A)
        self.add_parameter('TRANSPOSE_B', value=self._transpose_B)
        self.add_parameter('COMPRESSED', value=self._compressed)

    def set_shape(self, batch_size: int, M: int, K: int, N: int):
        self.set_parameter('BATCH_SIZE', batch_size)
        self.set_parameter('GLOBAL_M_VALUE', M)
        self.set_parameter('GLOBAL_K_VALUE', K)
        self.set_parameter('GLOBAL_N_VALUE', N)

    def get_shape(self):
        batch_size = self.get_parameter('BATCH_SIZE')
        M = self.get_parameter('GLOBAL_M_VALUE')
        K = self.get_parameter('GLOBAL_K_VALUE')
        N = self.get_parameter('GLOBAL_N_VALUE')
        return batch_size, M, K, N

    @abc.abstractmethod
    def get_block_shape(self) -> Tuple[int, int, int]:
        '''Get BM, BK and BN.'''

    def blocks_per_grid(self):
        batch_size, M, K, N = self.get_shape()
        if self._stype == 'dds':
            return (self.get_converter('C').get_attr('nnz').item(), batch_size)
        else:
            BM, BK, BN = self.get_block_shape()
            return (N // BN, M // BM, batch_size)

    def pre_compile(self):
        batch_size, M, K, N = self.get_shape()
        BM, BK, BN = self.get_block_shape()
        input_mask = [True] * 3 if self._biased else [True] * 2
        if self._stype == 'sdd':
            sparse_tensor = 'A'
            sparse_tensor_size = (K, M) if self._transpose_A else (M, K)
            block_size = (BK, BM) if self._transpose_A else (BM, BK)
            input_mask[1:1] = [False] * 2
        elif self._stype == 'dsd':
            sparse_tensor = 'B'
            sparse_tensor_size = (N, K) if self._transpose_B else (K, N)
            block_size = (BN, BK) if self._transpose_B else (BK, BN)
            input_mask[2:2] = [False] * 2
        else:
            sparse_tensor = 'C'
            sparse_tensor_size = (M, N)
            block_size = (BM, BN)
            input_mask[3:3] = [False] * 3

        converter = self.tesa_type[sparse_tensor](
            mask=self.get_mask(sparse_tensor),
            size=sparse_tensor_size,
            block_size=block_size
        )
        fixed_inputs = converter.get_attrs(self.tesa_attrs[sparse_tensor])
        self._converters[sparse_tensor] = converter

        if self._stype == 'dds' and self._compressed:
            output_shapes = [(batch_size, fixed_inputs[-1].item() * BM * BN)]
        else:
            output_shapes = [(batch_size, M, N)]
        return input_mask, fixed_inputs, output_shapes


class SparTASparseMatMulKernel(SparseMatMulKernel):

    def add_parameters(self):
        super().add_parameters()
        for dim in ['M', 'N', 'K']:
            self.add_parameter(
                f'BLOCK_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [16, 32, 64])
            )
            self.add_parameter(
                f'THREAD_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [4, 8])
            )

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, f'sparta_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def get_block_shape(self):
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BK = self.get_parameter('BLOCK_SIZE_K_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        return BM, BK, BN

    def get_thread_shape(self):
        TM = self.get_parameter('THREAD_SIZE_M_VALUE')
        TK = self.get_parameter('THREAD_SIZE_K_VALUE')
        TN = self.get_parameter('THREAD_SIZE_N_VALUE')
        return TM, TK, TN

    def threads_per_block(self):
        BM, BK, BN = self.get_block_shape()
        TM, TK, TN = self.get_thread_shape()
        return (BN // TN, BM // TM)


class OpenAISparseMatMulKernel(SparseMatMulKernel):

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, f'openai_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def get_block_shape(self):
        return 32, 64, 32

    def threads_per_block(self):
        return (256, )
