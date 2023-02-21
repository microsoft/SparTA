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


def _get_matmul_lut(impl: str):
    major, minor = torch.cuda.get_device_capability()
    try:
        lut_file = f'matmul.{impl}.{major}{minor}.csv'
        lut_text = res.read_text(look_up_tables, lut_file)
    except FileNotFoundError:
        lut_file = f'matmul.{impl}.default.csv'
        lut_text = res.read_text(look_up_tables, lut_file)
    return pd.read_csv(io.StringIO(lut_text))


_MATMUL_LUTS = {
    'sparta': _get_matmul_lut('sparta'),
    'openai': _get_matmul_lut('openai'),
}


class SparseMatMulKernel(KernelBase):

    __algo__: str = ''

    def __init__(
        self,
        mode: str,
        biased: bool, 
        transpose_A: bool,
        transpose_B: bool, 
        compressed: bool,
        batched: bool,
        dtype: str = 'float',
    ):
        self._biased = biased
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._compressed = compressed
        self._batched = batched
        self._mode = mode
        self._dtype = dtype

        lut = _MATMUL_LUTS[self.__algo__]
        mode_filter = lut['mode'] == self._mode
        trans_A_filter = lut['trans_A'] == self._transpose_A
        trans_B_filter = lut['trans_B'] == self._transpose_B
        self._lut = lut[mode_filter & trans_A_filter & trans_B_filter]

        super().__init__()

    def _add_parameters(self):
        self._add_parameter('MODE', value=self._mode)
        self._add_parameter('BIASED', value=self._biased)
        self._add_parameter('BATCHED', value=self._batched)
        self._add_parameter('TRANSPOSE_A', value=self._transpose_A)
        self._add_parameter('TRANSPOSE_B', value=self._transpose_B)
        self._add_parameter('COMPRESSED', value=self._compressed)
        self._add_parameter('BCSR')
        self._add_parameter('BCSC')

    def get_block_shape(self):
        BM = self.get_parameter('BLOCK_SIZE_M_VALUE')
        BK = self.get_parameter('BLOCK_SIZE_K_VALUE')
        BN = self.get_parameter('BLOCK_SIZE_N_VALUE')
        return BM, BK, BN

    def set_kernel_call(self, shape: Tuple[int, int, int, int], sparse_attr: Any):
        batch, M, K, N = shape
        M_32, K_32, N_32 = np.int32(M), np.int32(K), np.int32(N)
        BM, BK, BN = self.get_block_shape()
        row_num, col_num = M // BM, N // BN
        block = self.threads_per_block()
        raw_func = self._kernel
        zeros = torch.zeros
        int32 = np.int32

        func_code = jinja2.Template(textwrap.dedent('''
            def matmul_func(A, B{% if BIASED %}, bias{% endif %}):
                {% if MODE == "sdd" %}
                {% if BATCHED %}batch, {% endif %}{% if TRANSPOSE_B %}N, _{% else %}_, N{% endif %} = B.shape
                N_32 = int32(N)
                col_num = N // BN
                {% elif MODE == 'dsd' %}
                {% if BATCHED %}batch, {% endif %}{% if TRANSPOSE_A %}_, M{% else %}M, _{% endif %} = A.shape
                M_32 = int32(M)
                row_num = M // BM
                {% else %}
                {% if BATCHED %}batch, {% endif %}{% if TRANSPOSE_A %}K, _{% else %}_, K{% endif %} = A.shape
                K_32 = int32(K)
                {% endif %}
                {% if MODE == 'dds' and COMPRESSED %}
                C = zeros(({% if BATCHED %}batch, {% endif %}sparse_attr.indexes.block_nnz * BM * BN), device=A.device)
                {% else %}
                C = zeros(({% if BATCHED %}batch, {% endif %}M, N), device=A.device)
                {% endif %}
                raw_func(
                    A.detach(), B.detach(), {% if BIASED %}bias.detach(), {% endif %}C,
                    {% if (MODE == "sdd" and TRANSPOSE_A) or (MODE == "dsd" and not TRANSPOSE_B) %}
                    sparse_attr.indexes.col_ptr, sparse_attr.indexes.BCSC_idx, sparse_attr.indexes.nnz,
                    {% elif (MODE == "sdd" and not TRANSPOSE_A) or (MODE == "dsd" and TRANSPOSE_B) %}
                    sparse_attr.indexes.row_ptr, sparse_attr.indexes.BCSR_idx, sparse_attr.indexes.nnz,
                    {% else %}
                    sparse_attr.indexes.BCSR_idx, sparse_attr.indexes.nnz,
                    {% endif %}
                    M_32, K_32, N_32,
                    block=block,
                    {% if MODE == 'dds' %}
                    grid=(sparse_attr.indexes.block_nnz, {% if BATCHED %}batch{% else %}1{% endif %}, 1),
                    {% else %}
                    grid=(col_num, row_num, {% if BATCHED %}batch{% else %}1{% endif %}),
                    {% endif %}
                )
                return C
        ''')).render(self.get_parameters())

        exec(func_code, locals())
        self._func = locals()['matmul_func']

    def get_kernel_code(self):
        template_file = f'{self.__algo__}_sparse_matmul_{self._mode}.cuh.j2'
        kernel_template = res.read_text(templates, template_file)
        return jinja2.Template(kernel_template).render(self.get_parameters())


class SparTASparseMatMulKernel(SparseMatMulKernel):

    __algo__ = 'sparta'

    def _add_parameters(self):
        super()._add_parameters()
        for dim in ['M', 'K', 'N']:
            self._add_parameter(
                f'BLOCK_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [8, 16, 32, 64]),
            )
            self._add_parameter(
                f'THREAD_SIZE_{dim}_VALUE',
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
        assert BM >= 4
        assert BN >= 4
        assert BK >= 4
        assert BM & (BM - 1) == 0
        assert BK & (BK - 1) == 0
        assert BN & (BN - 1) == 0
        if all([f'THREAD_SIZE_{dim}_VALUE' in params for dim in ['M', 'K', 'N']]):
            TM = params['THREAD_SIZE_M_VALUE']
            TK = params['THREAD_SIZE_K_VALUE']
            TN = params['THREAD_SIZE_N_VALUE']
            assert BM >= TM
            assert BK >= TK
            assert BN >= TN
            assert TM & (TM - 1) == 0
            assert TK & (TK - 1) == 0
            assert TN & (TN - 1) == 0
            A_threads_per_row = (BM if self._transpose_A else BK) // 4
            B_threads_per_row = (BK if self._transpose_B else BN) // 4
            threads_per_block = (BM // TM) * (BN // TN)
            assert threads_per_block >= A_threads_per_row
            assert threads_per_block >= B_threads_per_row
            A_tile_row_stride = threads_per_block // A_threads_per_row
            B_tile_row_stride = threads_per_block // B_threads_per_row
            assert A_tile_row_stride <= (BK if self._transpose_A else BM)
            assert B_tile_row_stride <= (BN if self._transpose_B else BK)
        else:
            row = self._lut[(self._lut['BM'] == BM) & (self._lut['BK'] == BK) & (self._lut['BN'] == BN)]
            assert len(row) > 0, f'block shape ({BM}, {BK}, {BN}) not found in LUT'
            row = row.reset_index(drop=True).iloc[0, :]
            assert float(row['latency']) < float('inf'), f'block shape ({BM}, {BK}, {BN}) is invalid'
            TM, TK, TN = row['TM'], row['TK'], row['TN']
            self.set_parameter('THREAD_SIZE_M_VALUE', int(TM))
            self.set_parameter('THREAD_SIZE_K_VALUE', int(TK))
            self.set_parameter('THREAD_SIZE_N_VALUE', int(TN))
            self.estimated_latency_per_flop = row['latency'] / 4096 / 4096 / 4096


class OpenAISparseMatMulKernel(SparseMatMulKernel):

    __algo__ = 'openai'

    def _add_parameters(self):
        super()._add_parameters()
        for dim, val in zip(['M', 'K', 'N'], [32, 64, 32]):
            self._add_parameter(
                f'BLOCK_SIZE_{dim}_VALUE',
                value=val,
                is_tunable=True,
                search_space=TunableItemCfg('choice', [val]),
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
        row = self._lut.reset_index(drop=True).iloc[0, :]
        self.estimated_latency_per_flop = row['latency'] / 32 / 64 / 32
