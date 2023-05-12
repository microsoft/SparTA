# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import abc
import warnings
import importlib.resources as res
from typing import Any, Dict, Tuple

import torch
import jinja2
import numpy as np
import pandas as pd

from sparta import __env_ready__
if __env_ready__:
    from pycuda.driver import function_attribute
    from pycuda.compiler import SourceModule

from sparta.tuning import TunableItemCfg
from sparta.kernels import KernelBase, SparsityAttr, templates, look_up_tables
from sparta.testing import sparse_multi_head_attention_forward_reference, sparse_multi_head_attention_backward_reference


class FlashSparseAttentionKernel(KernelBase):

    __lut_shape__ = (64 * 12, 1024, 1024, 64)  # BxH, Nt, Ns, D
    __algo__ = 'flash'
    __dtype__ = ''
    __direction__ = ''

    def __init__(self, buffer: torch.Tensor):
        self._buffer = buffer
        super().__init__()

    def _add_parameters(self):
        self._add_parameter('GLOBAL_SIZE_D_VALUE')
        self._add_parameter(
            'BLOCK_SIZE_S_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128, 256]),
        )
        self._add_parameter(
            'BLOCK_SIZE_T_VALUE',
            is_tunable=True,
            search_space=TunableItemCfg('choice', [8, 16, 32, 64, 128, 256]),
        )
        self.attr = SparsityAttr(self, 'BLOCK_SIZE_T_VALUE', 'BLOCK_SIZE_S_VALUE', BCSR=False, BCSC=True)

    @abc.abstractmethod
    def _check_shape(self, Nt: int, Ns: int, D: int):
        """Check if input shape is valid."""

    def get_block_shape(self):
        Bt = self.get_parameter('BLOCK_SIZE_T_VALUE')
        Bs = self.get_parameter('BLOCK_SIZE_S_VALUE')
        return Bt, Bs

    @abc.abstractmethod
    def _calc_shared_mem_size(self, Bs: int, Bt: int, D: int):
        """Calc shared memory size in bytes."""

    def get_kernel_code(self):
        self._buffer.to()
        template_file = f'{self.__algo__}_sparse_attention_{self.__direction__}_{self.__dtype__}.cuh.j2'
        kernel_template = res.read_text(templates, template_file)
        with open('tmp.cu', 'w') as f:
            f.write(jinja2.Template(kernel_template).render(self.get_parameters()))
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def compile(self, params: Dict[str, Any], shape: Tuple):
        params['GLOBAL_SIZE_D_VALUE'] = shape[-1]
        super().compile(params, shape)


class FlashSparseAttentionFP32Kernel(FlashSparseAttentionKernel):

    __dtype__ = 'float32'

    def _add_parameters(self):
        super()._add_parameters()
        self._add_parameter('THREAD_SIZE_S_VALUE')
        self._add_parameter('THREAD_SIZE_T_VALUE')
        self._add_parameter('THREAD_SIZE_D_VALUE')

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
        Tt, Ts, Td = self.get_thread_shape()
        assert D & (D - 1) == 0  # TODO: pad
        threads_per_block = Bs // Ts * Bt // Tt
        smem_threads_D = D // 4
        assert threads_per_block >= smem_threads_D
        smem_threads_N = threads_per_block // smem_threads_D
        assert smem_threads_N <= Bt
        assert smem_threads_N <= Bs
        assert Bs // Ts <= 32
        assert Bs // Ts >= 4
        assert D // Td >= Bs // Ts
        if self.__direction__ == 'backward':
            assert D // Td >= Bt // Tt

    def get_thread_shape(self):
        Tt = self.get_parameter('THREAD_SIZE_T_VALUE')
        Ts = self.get_parameter('THREAD_SIZE_S_VALUE')
        Td = self.get_parameter('THREAD_SIZE_D_VALUE')
        return Tt, Ts, Td

    def threads_per_block(self):
        Bt, Bs = self.get_block_shape()
        Tt, Ts, _ = self.get_thread_shape()
        return (Bs // Ts, Bt // Tt, 1)


class FlashSparseAttentionFP16Kernel(FlashSparseAttentionKernel):

    __dtype__ = 'float16'

    def _add_parameters(self):
        super()._add_parameters()
        self._add_parameter('THREADS_PER_BLOCK')
        self._add_parameter('TS_WARP_SIZE_M_VALUE')
        self._add_parameter('TD_WARP_SIZE_M_VALUE')
        if self.__direction__ == 'backward':
            self._add_parameter('SD_WARP_SIZE_M_VALUE')

    def _check_parameters(self, params: Dict[str, Any]):
        Bs = params['BLOCK_SIZE_S_VALUE']
        Bt = params['BLOCK_SIZE_T_VALUE']
        assert Bs & (Bs - 1) == 0
        assert Bt & (Bt - 1) == 0
        assert Bs >= 16
        assert Bt >= 16
        threads_per_block = params['THREADS_PER_BLOCK']
        assert threads_per_block in [32, 64, 128, 256]
        Wn1 = params['TS_WARP_SIZE_M_VALUE']
        assert Wn1 in [8, 16, 32]
        Wn2 = params['TD_WARP_SIZE_M_VALUE']
        assert Wn2 in [8, 16, 32]
        if self.__direction__ == 'backward':
            Wn3 = params['SD_WARP_SIZE_M_VALUE']
            assert Wn3 in [8, 16, 32]

    def _check_shape(self, Nt: int, Ns: int, D: int):
        Bt, Bs = self.get_block_shape()
        assert D & (D - 1) == 0  # TODO: pad
        assert D >= 16
        threads_per_block = self.get_parameter('THREADS_PER_BLOCK')
        smem_threads_D = D // 8
        assert threads_per_block >= smem_threads_D
        smem_threads_N = threads_per_block // smem_threads_D
        assert smem_threads_N <= Bt
        assert smem_threads_N <= Bs
        thread_size = Bs * Bt // threads_per_block
        assert Bs // thread_size <= 32
        assert 8 <= thread_size <= Bs

    def threads_per_block(self):
        return (self.get_parameter('THREADS_PER_BLOCK'), 1, 1)

    def _build_kernel(self, kernel_code: str):
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(
                kernel_code,
                options=[
                    '-std=c++14',
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                ],
                no_extern_c=True,
            )
        return source_module.get_function(kernel_name)


class FlashSparseAttentionForwardKernel(FlashSparseAttentionKernel):

    __direction__ = 'forward'

    def set_kernel_call(self, shape: Tuple[int, int, int, int]):
        batch, Nt, Ns, D = shape
        self._check_shape(Nt, Ns, D)
        Ns_32, Nt_32, D_32 = np.int32(Ns), np.int32(Nt), np.int32(D)
        block = self.threads_per_block()
        Bt, Bs = self.get_block_shape()
        shared = self._calc_shared_mem_size(Bs, Bt, D)
        self._kernel.set_attribute(function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, shared)

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
                shared=shared,
            )
            return O

        self._func = attn_func

    def reference(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        sparse: bool = False,
    ):
        return sparse_multi_head_attention_forward_reference(Q, K, V, self.attr.mask)


class FlashSparseAttentionBackwardKernel(FlashSparseAttentionKernel):

    __direction__ = 'backward'

    def set_kernel_call(self, shape: Tuple[int, int, int, int]):
        batch, Nt, Ns, D = shape
        self._check_shape(Nt, Ns, D)
        Ns_32, Nt_32, D_32 = np.int32(Ns), np.int32(Nt), np.int32(D)
        block = self.threads_per_block()
        Bt, Bs = self.get_block_shape()
        shared = self._calc_shared_mem_size(Bs, Bt, D)
        self._kernel.set_attribute(function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, shared)

        def attn_func(grad, O, Q, K, V):
            grad_Q = torch.zeros_like(Q)
            grad_K = torch.zeros_like(Q)
            grad_V = torch.zeros_like(Q)
            self._kernel(
                Q, K, V, O, grad_Q, grad_K, grad_V, grad, self._buffer,
                self.attr.indexes.BCSC_idx,
                Ns_32, Nt_32,  # D_32,
                self.attr.indexes.nnz,
                block=block,
                grid=(Q.shape[0], 1, 1),
                shared=shared,
            )
            return grad_Q, grad_K, grad_V

        self._func = attn_func

    def reference(
        self,
        grad: torch.Tensor,
        O: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        sparse: bool = False,
    ):
        return sparse_multi_head_attention_backward_reference(grad, O, Q, K, V, self.attr.mask)


class FlashSparseAttentionFP32ForwardKernel(FlashSparseAttentionFP32Kernel, FlashSparseAttentionForwardKernel):

    def _calc_shared_mem_size(self, Bs: int, Bt: int, D: int):
        shared = 4 * (Bt * D + 2 * Bs * D)
        return shared


class FlashSparseAttentionFP32BackwardKernel(FlashSparseAttentionFP32Kernel, FlashSparseAttentionBackwardKernel):

    def _calc_shared_mem_size(self, Bs: int, Bt: int, D: int):
        shared = 4 * (2 * Bt * D + 4 * Bs * D)
        return shared


class FlashSparseAttentionFP16ForwardKernel(FlashSparseAttentionFP16Kernel, FlashSparseAttentionForwardKernel):

    def _calc_shared_mem_size(self, Bs: int, Bt: int, D: int):
        shared = 2 * (Bt * (D + 8) + 2 * (Bs * (D + 8)) + Bt * (Bs + 8))
        return shared


class FlashSparseAttentionFP16BackwardKernel(FlashSparseAttentionFP16Kernel, FlashSparseAttentionBackwardKernel):

    def _calc_shared_mem_size(self, Bs: int, Bt: int, D: int):
        shared = 2 * (2 * Bt * (D + 8) + 4 * (Bs * (D + 8)) + 2 * Bt * (Bs + 8))
        return shared
