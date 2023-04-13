# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import textwrap
import importlib.resources as res
from typing import Any, Dict, Tuple, Optional

import torch
import jinja2
import numpy as np
import pandas as pd

from sparta.tuning import TunableItemCfg
from sparta.kernels import KernelBase, SparsityAttr, templates, look_up_tables
from sparta.testing import sparse_multi_head_attention_reference


class FlashSparseAttentionForwardKernel(KernelBase):

    __lut_shape__ = (64 * 12, 1024, 1024, 64)  # BxH, Nt, Ns, D
    __algo__ = 'flash'
    __direction__ = 'forward'

    def __init__(self, buffer: torch.Tensor, dtype: str = 'float'):
        self._buffer = buffer
        self._dtype = dtype
        super().__init__()

    def _add_parameters(self):
        self._add_parameter(
            f'GLOBAL_SIZE_D_VALUE',
        )
        for dim in ['S', 'T']:
            self._add_parameter(
                f'BLOCK_SIZE_{dim}_VALUE',
                is_tunable=True,
                search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128, 256]),
            )
            self._add_parameter(
                f'THREAD_SIZE_{dim}_VALUE',
            )
        self.attr = SparsityAttr(self, 'BLOCK_SIZE_T_VALUE', 'BLOCK_SIZE_S_VALUE', BCSR=False, BCSC=True)

    def _check_parameters(self, params: Dict[str, Any]):
        Bt = params['BLOCK_SIZE_T_VALUE']
        Bs = params['BLOCK_SIZE_S_VALUE']
        assert Bt >= 4
        assert Bs >= 4
        assert Bt & (Bt - 1) == 0
        assert Bs & (Bs - 1) == 0
        Tt = params['THREAD_SIZE_T_VALUE']
        Ts = params['THREAD_SIZE_S_VALUE']
        assert Bt >= Tt
        assert Bs >= Ts
        assert Tt & (Tt - 1) == 0
        assert Ts & (Ts - 1) == 0

    def _check_shape(self, Nt: int, Ns: int, D: int):
        Bt, Bs = self.get_block_shape()
        Tt, Ts = self.get_thread_shape()
        assert D & (D - 1) == 0  # TODO: pad
        threads_per_block = Bs // Ts * Bt // Tt
        smem_threads_D = D // 4
        assert threads_per_block >= smem_threads_D
        smem_threads_N = threads_per_block // smem_threads_D
        assert smem_threads_N <= Bt
        assert smem_threads_N <= Bs
        assert Bs // Ts <= 32
        assert Bs // Ts >= 4
        assert D * Ts >= Bs

    def get_block_shape(self):
        Bt = self.get_parameter('BLOCK_SIZE_T_VALUE')
        Bs = self.get_parameter('BLOCK_SIZE_S_VALUE')
        return Bt, Bs

    def get_thread_shape(self):
        Tt = self.get_parameter('THREAD_SIZE_T_VALUE')
        Ts = self.get_parameter('THREAD_SIZE_S_VALUE')
        return Tt, Ts

    def threads_per_block(self):
        Bt, Bs = self.get_block_shape()
        Tt, Ts = self.get_thread_shape()
        return (Bs // Ts, Bt // Tt, 1)

    def set_kernel_call(self, shape: Tuple[int, int, int, int]):
        batch, Nt, Ns, D = shape
        self._check_shape(Nt, Ns, D)
        Ns_32, Nt_32, D_32 = np.int32(Ns), np.int32(Nt), np.int32(D)
        block = self.threads_per_block()

        def attn_func(Q, K, V):
            O = torch.zeros_like(Q)
            self._buffer.fill_(0)  # TODO: try BCSR and delete this
            self._kernel(
                Q, K, V, O, self._buffer,
                self.attr.indexes.BCSC_idx,
                Ns_32, Nt_32,  # D_32,
                self.attr.indexes.nnz,
                block=block,
                grid=(Q.shape[0], 1, 1),
            )
            return O

        self._func = attn_func

    def get_kernel_code(self):
        template_file = f'{self.__algo__}_sparse_attention_{self.__direction__}.cuh.j2'
        kernel_template = res.read_text(templates, template_file)
        with open('tmp.cu', 'w') as f:
            f.write(jinja2.Template(kernel_template).render(self.get_parameters()))
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def reference(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        sparse: bool = False,
    ):
        return sparse_multi_head_attention_reference(Q, K, V, self.attr.mask)

    def compile(self, params: Dict[str, Any], shape: Tuple):
        params['GLOBAL_SIZE_D_VALUE'] = shape[-1]
        super().compile(params, shape)
