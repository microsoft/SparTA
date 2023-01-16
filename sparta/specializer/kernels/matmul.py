# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch
import jinja2
import numpy as np

from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels.kernel_base import KernelBase, PortConfig


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')


class SparseMatMulKernel(KernelBase):

    __algo__: str = ''

    def __init__(
        self,
        mode: str,
        dtype: str = 'float',
        biased: bool = True, 
        transpose_A: bool = False,
        transpose_B: bool = True, 
        compressed: bool = True,
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse type: {mode}')
        self._biased = biased
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._compressed = compressed
        self._bcs_mode = ''
        self._mode = mode
        self._dtype = dtype
        self._sparse_port = ''
        self._sparse_block_H = ''
        self._sparse_block_W = ''
        self._tesa_vars = []
        super().__init__()

    def _set_ports(self):
        self.ports['A'] = PortConfig(name='A', is_input=True)
        self.ports['B'] = PortConfig(name='B', is_input=True)
        if self._biased:
            self.ports['bias'] = PortConfig(name='bias', is_input=True)
        self.ports['C'] = PortConfig(name='C', is_input=False)

        if self._mode == 'sdd':
            self._sparse_port = 'A'
            if self._transpose_A:
                self._sparse_block_H = 'BLOCK_SIZE_K_VALUE'
                self._sparse_block_W = 'BLOCK_SIZE_M_VALUE'
                self._bcs_mode = 'BCSC'
            else:
                self._sparse_block_H = 'BLOCK_SIZE_M_VALUE'
                self._sparse_block_W = 'BLOCK_SIZE_K_VALUE'
                self._bcs_mode = 'BCSR'
        elif self._mode == 'dsd':
            self._sparse_port = 'B'
            if self._transpose_B:
                self._sparse_block_H = 'BLOCK_SIZE_N_VALUE'
                self._sparse_block_W = 'BLOCK_SIZE_K_VALUE'
                self._bcs_mode = 'BCSR'
            else:
                self._sparse_block_H = 'BLOCK_SIZE_K_VALUE'
                self._sparse_block_W = 'BLOCK_SIZE_N_VALUE'
                self._bcs_mode = 'BCSC'
        elif self._mode == 'dds':
            self._sparse_port = 'C'
            self._sparse_block_H = 'BLOCK_SIZE_M_VALUE'
            self._sparse_block_W = 'BLOCK_SIZE_N_VALUE'
            self._bcs_mode = 'BCSR'

        BCSR = self._bcs_mode == 'BCSR'
        BCSC = self._bcs_mode == 'BCSC'
        self.ports[self._sparse_port].set_sparse(BCSR, BCSC)

        if BCSR:
            self._tesa_vars = ['row_ptr', 'BCSR_idx', 'nnz']
        elif BCSC:
            self._tesa_vars = ['col_ptr', 'BCSC_idx', 'nnz']
        else:
            raise ValueError('failed to initialize SparseMatMulKernel')
        if self._mode == 'dds':
            self._tesa_vars = self._tesa_vars[1:]

    def _add_parameters(self):
        self._add_parameter('BATCH_SIZE')
        self._add_parameter('GLOBAL_M_VALUE')
        self._add_parameter('GLOBAL_K_VALUE')
        self._add_parameter('GLOBAL_N_VALUE')
        self._add_parameter('BIASED', value=self._biased)
        self._add_parameter('TRANSPOSE_A', value=self._transpose_A)
        self._add_parameter('TRANSPOSE_B', value=self._transpose_B)
        self._add_parameter('COMPRESSED', value=self._compressed)
        self._add_parameter('BCSR')
        self._add_parameter('BCSC')

    def set_parameters(self, params: Dict[str, Any]):
        super().set_parameters(params)
        sparse_port = self.ports[self._sparse_port]
        if self._bcs_mode == 'BCSR':
            self.set_parameter('BCSR', True)
            self.set_parameter('BCSC', False)
        elif self._bcs_mode == 'BCSC':
            self.set_parameter('BCSR', sparse_port.BCSR)
            self.set_parameter('BCSC', True)
        BH = self.get_parameter(self._sparse_block_H)
        BW = self.get_parameter(self._sparse_block_W)
        sparse_port.set_block_size(BH, BW)

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

    def get_block_shape(self):
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BK = self.get_parameter('BLOCK_SIZE_K_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        return BM, BK, BN

    def blocks_per_grid(self):
        batch_size, M, K, N = self.get_shape()
        if self._mode == 'dds':
            return (self.ports['C'].indexes.block_nnz, batch_size, 1)
        else:
            BM, BK, BN = self.get_block_shape()
            return (N // BN, M // BM, batch_size)

    def _set_func_call(self, kernel_func_call: Callable):
        batch_size, M, K, N = self.get_shape()
        BM, BK, BN = self.get_block_shape()

        indexes = self.ports[self._sparse_port].indexes
        if self._mode == 'dds' and self._compressed:
            C_shape = (batch_size, indexes.block_nnz * BM * BN)
        else:
            C_shape = (batch_size, M, N)

        M_32 = np.int32(M)
        K_32 = np.int32(K)
        N_32 = np.int32(N)
        tesa_vars = [getattr(indexes, x) for x in self._tesa_vars]
        block = self.threads_per_block()
        grid = self.blocks_per_grid()

        if self._biased:
            def matmul_func(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
                C = torch.zeros(C_shape, device=A.device)
                kernel_func_call(
                    A, B, bias, C, *tesa_vars,
                    M_32, K_32, N_32,
                    block=block, grid=grid
                )
                return C
        else:
            def matmul_func(A: torch.Tensor, B: torch.Tensor):
                C = torch.zeros(C_shape, device=A.device)
                kernel_func_call(
                    A, B, C, *tesa_vars,
                    M_32, K_32, N_32,
                    block=block, grid=grid
                )
                return C

        return matmul_func

    def get_kernel_code(self):
        template_file = f'{self.__algo__}_sparse_matmul_{self._mode}.cuh.j2'
        with open(os.path.join(TEMPLATE_DIR, template_file)) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def _convert_data(self, inputs, outputs):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].detach()
        outputs[0] = outputs[0].detach()
        if self._compressed:
            if self._sparse_port == 'A':
                inputs[0] = self.ports['A'].indexes.convert(inputs[0])
            elif self._sparse_port == 'B':
                inputs[1] = self.ports['B'].indexes.convert(inputs[1])
            elif self._sparse_port == 'C':
                outputs[0] = self.ports['C'].indexes.convert(outputs[0])

    def reference_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_str = 'bkm' if self._transpose_A else 'bmk'
        B_str = 'bnk' if self._transpose_B else 'bkn'
        return torch.einsum(f'{A_str},{B_str}->bmn', A, B)

    def reference_bias(self, C: torch.Tensor, bias: torch.Tensor):
        return C + bias.unsqueeze(1)

    def reference(self, *args):
        C = self.reference_matmul(args[0], args[1])
        if self._biased:
            C = self.reference_bias(C, args[2])
        if self._mode == 'dds' and self.ready:
            C = C * self.ports['C'].indexes.get_mask()  # DDS known issue
        return C


class SparTASparseMatMulKernel(SparseMatMulKernel):

    __algo__ = 'sparta'

    def _add_parameters(self):
        super()._add_parameters()
        for dim in ['M', 'K', 'N']:
            self._add_parameter(
                f'BLOCK_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [16, 32, 64])
            )
            self._add_parameter(
                f'THREAD_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [4, 8])
            )

    def get_thread_shape(self):
        TM = self.get_parameter('THREAD_SIZE_M_VALUE')
        TK = self.get_parameter('THREAD_SIZE_K_VALUE')
        TN = self.get_parameter('THREAD_SIZE_N_VALUE')
        return TM, TK, TN

    def threads_per_block(self):
        BM, BK, BN = self.get_block_shape()
        TM, TK, TN = self.get_thread_shape()
        return (BN // TN, BM // TM, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        BM = params['BLOCK_SIZE_M_VALUE']
        BK = params['BLOCK_SIZE_K_VALUE']
        BN = params['BLOCK_SIZE_N_VALUE']
        TM = params['THREAD_SIZE_M_VALUE']
        TK = params['THREAD_SIZE_K_VALUE']
        TN = params['THREAD_SIZE_N_VALUE']
        assert BM > TM
        assert BK > TK
        assert BN > TN
        A_thread_per_rows = (BM if self._transpose_A else BK) // 4
        B_thread_per_rows = (BK if self._transpose_A else BN) // 4
        threads_per_block = (BM // TM) * (BN // TN)
        assert threads_per_block >= A_thread_per_rows
        assert threads_per_block >= B_thread_per_rows


class OpenAISparseMatMulKernel(SparseMatMulKernel):

    __algo__ = 'openai'

    def _add_parameters(self):
        super()._add_parameters()
        self._add_parameter(
            'BLOCK_SIZE_M_VALUE',
            value=32,
            is_tunable=True,
            search_space=TunableItemCfg('choice', [32]),
        )
        self._add_parameter(
            'BLOCK_SIZE_K_VALUE',
            value=64,
            is_tunable=True,
            search_space=TunableItemCfg('choice', [64]),
        )
        self._add_parameter(
            'BLOCK_SIZE_N_VALUE',
            value=32,
            is_tunable=True,
            search_space=TunableItemCfg('choice', [32]),
        )

    def threads_per_block(self):
        return (256, 1, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        if 'BLOCK_SIZE_M_VALUE' in params:
            assert params['BLOCK_SIZE_M_VALUE'] == 32
        if 'BLOCK_SIZE_K_VALUE' in params:
            assert params['BLOCK_SIZE_K_VALUE'] == 64
        if 'BLOCK_SIZE_N_VALUE' in params:
            assert params['BLOCK_SIZE_N_VALUE'] == 32
