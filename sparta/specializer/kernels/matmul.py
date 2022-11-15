# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List, Tuple, Callable

import torch
import jinja2

from sparta.common.tesa import BCSRH, BCSRV
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels import KernelBase, PortConfig


TEMPLATE_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'templates')


def get_matmul_func_call(
    func: Callable, stype: str, biased: bool, C_shape: Tuple, tesa_vars: List[torch.Tensor],
    block: Tuple[int, int, int], grid: Tuple[int, int, int]
):
    if stype == 'sdd':
        if biased:
            def matmul_func(A, B, bias):
                C = torch.zeros(C_shape, device=A.device)
                func(A, *tesa_vars, B, bias, C, block=block, grid=grid)
                return C
        else:
            def matmul_func(A, B):
                C = torch.zeros(C_shape, device=A.device)
                func(A, *tesa_vars, B, C, block=block, grid=grid)
                return C
    elif stype == 'dsd':
        if biased:
            def matmul_func(A, B, bias):
                C = torch.zeros(C_shape, device=A.device)
                func(A, B, *tesa_vars, bias, C, block=block, grid=grid)
                return C
        else:
            def matmul_func(A, B):
                C = torch.zeros(C_shape, device=A.device)
                func(A, B, *tesa_vars, C, block=block, grid=grid)
                return C
    elif stype == 'dds':
        if biased:
            def matmul_func(A, B, bias):
                C = torch.zeros(C_shape, device=A.device)
                func(A, B, bias, *tesa_vars, C, block=block, grid=grid)
                return C
        else:
            def matmul_func(A, B):
                C = torch.zeros(C_shape, device=A.device)
                func(A, B, *tesa_vars, C, block=block, grid=grid)
                return C
    else:
        raise ValueError(f'invalid stype "{stype}"')
    return matmul_func


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

    def _set_ports(self):
        self.ports['A'] = PortConfig(name='A', is_input=True)
        self.ports['B'] = PortConfig(name='B', is_input=True)
        if self._biased:
            self.ports['bias'] = PortConfig(name='bias', is_input=True)
        self.ports['C'] = PortConfig(name='C', is_input=False)

        def shape_to_params(H: str, W: str):
            return [
                f'GLOBAL_{H}_VALUE', f'GLOBAL_{W}_VALUE',
                f'BLOCK_SIZE_{H}_VALUE', f'BLOCK_SIZE_{W}_VALUE'
            ]

        if self._stype == 'sdd':
            if self._transpose_A:
                self.ports['A'].set_tesa(BCSRV, shape_to_params('K', 'M'))
            else:
                self.ports['A'].set_tesa(BCSRH, shape_to_params('M', 'K'))
        elif self._stype == 'dsd':
            if self._transpose_B:
                self.ports['B'].set_tesa(BCSRH, shape_to_params('N', 'K'))
            else:
                self.ports['B'].set_tesa(BCSRV, shape_to_params('K', 'N'))
        elif self._stype == 'dds':
            if self._bcsr_main == 'H':
                self.ports['C'].set_tesa(BCSRH, shape_to_params('M', 'N'))
            else:
                self.ports['C'].set_tesa(BCSRV, shape_to_params('M', 'N'))

    def _add_parameters(self):
        self._add_parameter('BATCH_SIZE')
        self._add_parameter('GLOBAL_M_VALUE')
        self._add_parameter('GLOBAL_K_VALUE')
        self._add_parameter('GLOBAL_N_VALUE')
        self._add_parameter('BIASED', value=self._biased)
        self._add_parameter('TRANSPOSE_A', value=self._transpose_A)
        self._add_parameter('TRANSPOSE_B', value=self._transpose_B)
        self._add_parameter('COMPRESSED', value=self._compressed)

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
        if self._stype == 'dds':
            return (self.get_converter('C').get_attr('nnz').item(), batch_size, 1)
        else:
            BM, BK, BN = self.get_block_shape()
            return (N // BN, M // BM, batch_size)

    def _set_func_call(self, kernel_func_call: Callable):
        batch_size, M, K, N = self.get_shape()
        BM, BK, BN = self.get_block_shape()
        if self._stype == 'sdd':
            converter = self.get_converter('A')
            tesa_attrs = ['col_ptr', 'row_idx'] if self._transpose_A else ['row_ptr', 'col_idx']
        elif self._stype == 'dsd':
            converter = self.get_converter('B')
            tesa_attrs = ['row_ptr', 'col_idx'] if self._transpose_B else ['col_ptr', 'row_idx']
        else:
            converter = self.get_converter('C')
            tesa_attrs = ['row_idx', 'col_idx', 'nnz']
        if self._stype == 'dds' and self._compressed:
            C_shape = (batch_size, converter.get_attr('nnz').item() * BM * BN)
        else:
            C_shape = (batch_size, M, N)
        tesa_vars = converter.get_attrs(tesa_attrs)
        return get_matmul_func_call(
            func=kernel_func_call,
            stype=self._stype,
            biased=self._biased,
            C_shape=C_shape,
            tesa_vars=tesa_vars,
            grid=self.blocks_per_grid(),
            block=self.threads_per_block(),
        )

    def reference(self, inputs: List[torch.Tensor]):
        A = inputs[0]
        B = inputs[1]
        if self._stype == 'sdd':
            if self._compressed:
                A = self.get_converter('A').inverse(A)
            A *= self.get_mask('A')
        elif self._stype == 'dsd':
            if self._compressed:
                B = self.get_converter('B').inverse(B)
            B *= self.get_mask('B')
        A_str = 'bkm' if self._transpose_A else 'bmk'
        B_str = 'bnk' if self._transpose_B else 'bkn'
        C = torch.einsum(f'{A_str}, {B_str} -> bmn', A, B)
        if self._biased:
            C += inputs[2].unsqueeze(1)
        if self._stype == 'dds':
            C *= self.get_converter('C').get_mask()  # DDS known issue
            if self._compressed:
                C = self.get_converter('C').convert(C)
        return [C]


class SparTASparseMatMulKernel(SparseMatMulKernel):

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

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, f'sparta_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

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

    def _add_parameters(self):
        super()._add_parameters()
        self._add_parameter('BLOCK_SIZE_M_VALUE', value=32)
        self._add_parameter('BLOCK_SIZE_K_VALUE', value=64)
        self._add_parameter('BLOCK_SIZE_N_VALUE', value=32)

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, f'openai_sparse_matmul_{self._stype}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def threads_per_block(self):
        return (256, 1, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        assert len(params.keys()) == 0, 'The OpenAI sparse matmul kernel has no tubable parameters.'
