# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch
import jinja2

from sparta.common.tesa import BCSR, BCSC
from sparta.common.tuning import TunableItemCfg
from sparta.specializer.kernels.kernel_base import KernelBase, PortConfig


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')


def get_matmul_func_call(
    func: Callable,
    biased: bool,
    C_shape: Tuple,
    tesa_vars: List[torch.Tensor],
    block: Tuple[int, int, int],
    grid: Tuple[int, int, int],
    sparse_port: str,
    reorder_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    ptr, idx, nnz = tesa_vars
    if biased:
        def matmul_func(A, B, bias):
            C = torch.zeros(C_shape, device=A.device)
            func(A, B, bias, C, ptr, idx, nnz, block=block, grid=grid)
            return C
        if sparse_port == 'A' and reorder_func is not None:
            combined_func = lambda A, B, bias: matmul_func(reorder_func(A), B, bias)
        elif sparse_port == 'B' and reorder_func is not None:
            combined_func = lambda A, B, bias: matmul_func(A, reorder_func(B), bias)
        elif sparse_port == 'C' and reorder_func is not None:
            combined_func = lambda A, B, bias: reorder_func(matmul_func(A, B, bias))
        else:
            combined_func = matmul_func
    else:
        def matmul_func(A, B):
            C = torch.zeros(C_shape, device=A.device)
            func(A, B, C, ptr, idx, nnz, block=block, grid=grid)
            return C
        if sparse_port == 'A' and reorder_func is not None:
            combined_func = lambda A, B: matmul_func(reorder_func(A), B)
        elif sparse_port == 'B' and reorder_func is not None:
            combined_func = lambda A, B: matmul_func(A, reorder_func(B))
        elif sparse_port == 'C' and reorder_func is not None:
            combined_func = lambda A, B: reorder_func(matmul_func(A, B))
        else:
            combined_func = matmul_func
    return combined_func


class SparseMatMulKernel(KernelBase):

    def __init__(
        self,
        mode: str,
        dtype: str = 'float',
        biased: bool = True, 
        transpose_A: bool = False,
        transpose_B: bool = True, 
        compressed: bool = True,
        bcs_mode: Optional[str] = None,
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse type: {mode}')
        self._biased = biased
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._compressed = compressed
        self._bcs_mode = bcs_mode
        self._mode = mode
        self._dtype = dtype
        super().__init__()

    def _set_ports(self):
        self.ports['A'] = PortConfig(name='A', is_input=True)
        self.ports['B'] = PortConfig(name='B', is_input=True)
        if self._biased:
            self.ports['bias'] = PortConfig(name='bias', is_input=True)
        self.ports['C'] = PortConfig(name='C', is_input=False)

        def shape_to_params(H: str, W: str):
            return [f'BLOCK_SIZE_{H}_VALUE', f'BLOCK_SIZE_{W}_VALUE']

        if self._mode == 'sdd':
            sparse_port_name = 'A'
            if self._transpose_A:
                tesa_params = shape_to_params('K', 'M')
                real_tesa_type = BCSC
            else:
                tesa_params = shape_to_params('M', 'K')
                real_tesa_type = BCSR
        elif self._mode == 'dsd':
            sparse_port_name = 'B'
            if self._transpose_B:
                tesa_params = shape_to_params('N', 'K')
                real_tesa_type = BCSR
            else:
                tesa_params = shape_to_params('K', 'N')
                real_tesa_type = BCSC
        elif self._mode == 'dds':
            sparse_port_name = 'C'
            tesa_params = shape_to_params('M', 'N')
            real_tesa_type = {'BCSR': BCSR, 'BCSC': BCSC, None: BCSR}[self._bcs_mode]
        tesa_type = {'BCSR': BCSR, 'BCSC': BCSC, None: real_tesa_type}[self._bcs_mode]
        self.ports[sparse_port_name].set_tesa(tesa_type, tesa_params, real_tesa_type)

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
        if self._mode == 'dds':
            return (self.get_converter('C').get_attr('nnz').item(), batch_size, 1)
        else:
            BM, BK, BN = self.get_block_shape()
            return (N // BN, M // BM, batch_size)

    def _set_func_call(self, kernel_func_call: Callable):
        batch_size, M, K, N = self.get_shape()
        BM, BK, BN = self.get_block_shape()
        if self._mode == 'sdd':
            sparse_port = self.ports['A']
            tesa_attrs = ['col_ptr', 'row_idx'] if self._transpose_A else ['row_ptr', 'col_idx']
        elif self._mode == 'dsd':
            sparse_port = self.ports['B']
            tesa_attrs = ['row_ptr', 'col_idx'] if self._transpose_B else ['col_ptr', 'row_idx']
        else:
            sparse_port = self.ports['C']
            tesa_attrs = ['row_idx', 'col_idx']
        tesa_attrs.append('nnz')
        converter = sparse_port.converter
        if self._mode == 'dds' and self._compressed:
            C_shape = (batch_size, converter.get_attr('nnz').item() * BM * BN)
        else:
            C_shape = (batch_size, M, N)
        tesa_vars = converter.get_attrs(tesa_attrs)
        reorder_func = None
        if self._compressed:
            R_to_C = converter.reorder_BCSR_to_BCSC
            C_to_R = converter.reorder_BCSC_to_BCSR
            if sparse_port.real_tesa_type is BCSR and sparse_port.tesa_type is BCSC:
                reorder_func = R_to_C if self._mode == 'dds' else C_to_R
            elif sparse_port.real_tesa_type is BCSC and sparse_port.tesa_type is BCSR:
                reorder_func = C_to_R if self._mode == 'dds' else R_to_C
        return get_matmul_func_call(
            func=kernel_func_call,
            biased=self._biased,
            C_shape=C_shape,
            tesa_vars=tesa_vars,
            grid=self.blocks_per_grid(),
            block=self.threads_per_block(),
            sparse_port=sparse_port.name,
            reorder_func=reorder_func,
        )

    def _convert_data(self, inputs, outputs):
        if self._compressed:
            sparse_tensor = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[self._mode]
            data = {'A': inputs[0], 'B': inputs[1], 'C': outputs[0]}
            converter = self.get_converter(sparse_tensor)
            sparse_port = self.ports[sparse_tensor]
            data[sparse_tensor] = converter.convert(data[sparse_tensor])
            if sparse_port.real_tesa_type is BCSR and sparse_port.tesa_type is BCSC:
                data[sparse_tensor] = converter.reorder_BCSR_to_BCSC(data[sparse_tensor])
            elif sparse_port.real_tesa_type is BCSC and sparse_port.tesa_type is BCSR:
                data[sparse_tensor] = converter.reorder_BCSC_to_BCSR(data[sparse_tensor])
            inputs[0], inputs[1], outputs[0] = data['A'], data['B'], data['C']

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
            C = C * self.get_converter('C').get_mask()  # DDS known issue
        return C


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
        with open(os.path.join(TEMPLATE_DIR, f'sparta_sparse_matmul_{self._mode}.cuh.j2')) as f:
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

    def get_kernel_code(self):
        with open(os.path.join(TEMPLATE_DIR, f'openai_sparse_matmul_{self._mode}.cuh.j2')) as f:
            kernel_template = f.read()
        return jinja2.Template(kernel_template).render(self.get_parameters())

    def threads_per_block(self):
        return (256, 1, 1)

    def _check_parameters(self, params: Dict[str, Any]):
        if 'BLOCK_SIZE_M_VALUE' in params:
            assert params['BLOCK_SIZE_M_VALUE'] == 32
        if 'BLOCK_SIZE_K_VALUE' in params:
            assert params['BLOCK_SIZE_K_VALUE'] == 64
        if 'BLOCK_SIZE_N_VALUE' in params:
            assert params['BLOCK_SIZE_N_VALUE'] == 32
