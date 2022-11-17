# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch

from sparta.specializer.kernels import SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseBatchMatMulCtx(SparseCtxBase):

    def __init__(
        self, sparse_type: str, transpose_A: bool, transpose_B: bool,
        biased: bool, compressed: bool
    ):
        super().__init__()

        self._biased = biased
        self._compressed = compressed
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._sparse_type = sparse_type

        def select(x: str, source: str, target: str):
            return target[source.find(x)]

        def rearange(s: str, source_order: str, target_order: str):
            return ''.join(select(x, source_order, s) for x in target_order)

        def calc_tesa_shape(sparse_type: str, trans_A: bool, trans_B: bool):
            if sparse_type == 'sdd':
                return ('K', 'M') if trans_A else ('M', 'K')
            elif sparse_type == 'dsd':
                return ('N', 'K') if trans_B else ('K', 'N')
            else:
                return ('M', 'N')

        sparse_tensor = select('s', sparse_type, 'ABC')
        if sparse_tensor == 'A' and transpose_A:
            bcsr_main = 'V'
        elif sparse_tensor == 'B' and not transpose_B:
            bcsr_main = 'V'
        else:
            bcsr_main = 'H'

        self._tesa_shapes: Dict[str, Tuple[str, str]] = {}
        for kernel_name, bias, target_order, trans_A, trans_B in zip(
            ['forward:C', 'backward:A', 'backward:B'],
            [biased, False, False],
            ['ABC', 'BCA' if transpose_A else 'CBA', 'CAB' if transpose_B else 'ACB'],
            [transpose_A, transpose_A and transpose_B, not transpose_A or transpose_B],
            [transpose_B, transpose_A or not transpose_B, transpose_A and transpose_B],
        ):
            s_type = rearange(sparse_type, 'ABC', target_order)
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={
                    'sparta': SparTASparseMatMulKernel,
                    'openai': OpenAISparseMatMulKernel,
                },
                args={
                    'biased': bias,
                    'bcsr_main': bcsr_main,
                    'compressed': compressed,
                    'transpose_A': trans_A,
                    'transpose_B': trans_B,
                    'sparse_type': s_type,
                },
                mask_map={sparse_tensor: select(sparse_tensor, target_order, 'ABC')},
            )
            self._tesa_shapes[kernel_name] = calc_tesa_shape(s_type, trans_A, trans_B)

    def set_shape(self, batch_size: int, M: int, K: int, N: int):
        self._kernels['forward:C'].set_shape(batch_size, M, K, N)
        if self._transpose_A:
            self._kernels['backward:A'].set_shape(batch_size, K, N, M)
        else:
            self._kernels['backward:A'].set_shape(batch_size, M, N, K)
        if self._transpose_B:
            self._kernels['backward:B'].set_shape(batch_size, N, M, K)
        else:
            self._kernels['backward:B'].set_shape(batch_size, K, M, N)

    def get_conditions(self, impls: Dict[str, str]):
        if self._compressed and len(impls) > 1:
            conditions = [[], []]
            fixed_dims = [None, None]
            for kernel_name, impl in impls.items():
                for k, dim in enumerate(self._tesa_shapes[kernel_name]):
                    if impl == 'sparta':
                        v = f'{kernel_name};BLOCK_SIZE_{dim}_VALUE'
                    else:  # impl == 'openai'
                        v = {'M': 32, 'K': 64, 'N': 32}[dim]
                        if fixed_dims[k] is None:
                            fixed_dims[k] = v
                        elif fixed_dims[k] != v:
                            return None
                    conditions[k].append(v)
            return conditions
        else:
            return []

    def _split_graph(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None
    ):
        funcs, inputs = [], []
        for kernel_name in kernels:
            if kernel_name == 'forward:C':
                funcs.append(self.forward_C)
                forward_inputs = [sample_inputs['A'], sample_inputs['B']]
                if self._biased:
                    forward_inputs.append(sample_inputs['bias'])
                inputs.append(forward_inputs)
            elif kernel_name == 'backward:A':
                funcs.append(self.backward_A)
                inputs.append([sample_grad, sample_inputs['B']])
            elif kernel_name == 'backward:B':
                funcs.append(self.backward_B)
                inputs.append([sample_grad, sample_inputs['A']])
        return funcs, inputs

    def build(self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        super().build(config, mask)
        forward_kernel = self._kernels['forward:C'].active_kernel()
        if forward_kernel is not None:
            if self._biased:
                self.forward_C = lambda A, B, bias: forward_kernel(A, B, bias)
            else:
                self.forward_C = lambda A, B: forward_kernel(A, B)
            if self._sparse_type == 'dds' and self._compressed:
                C_converter = self._kernels['forward:C'].get_converter('C')
                self.backward_bias = lambda grad_C: C_converter.sum(grad_C, axis=-2)
            else:
                self.backward_bias = lambda grad_C: grad_C.sum(-2)
        backward_A_kernel = self._kernels['backward:A'].active_kernel()
        if backward_A_kernel is not None:
            if self._transpose_A:
                self.backward_A = lambda grad_C, B: backward_A_kernel(B, grad_C)
            else:
                self.backward_A = lambda grad_C, B: backward_A_kernel(grad_C, B)
        backward_B_kernel = self._kernels['backward:B'].active_kernel()
        if backward_B_kernel is not None:
            if self._transpose_B:
                self.backward_B = lambda grad_C, A: backward_B_kernel(grad_C, A)
            else:
                self.backward_B = lambda grad_C, A: backward_B_kernel(A, grad_C)


class SparseBatchMatMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, sparta_ctx: SparseBatchMatMulCtx,
        input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ):
        input = input.detach()
        weight = weight.detach()
        ctx.save_for_backward(input, weight, bias)
        ctx.sparta_ctx = sparta_ctx
        if bias is None:
            return sparta_ctx.forward_C(input, weight)
        else:
            return sparta_ctx.forward_C(input, weight, bias.detach())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_output = grad_output.detach()
        if ctx.needs_input_grad[1]:
            grad_input = ctx.sparta_ctx.backward_A(grad_output, weight)
        if ctx.needs_input_grad[2]:
            grad_weight = ctx.sparta_ctx.backward_B(grad_output, input)
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = ctx.sparta_ctx.backward_bias(grad_output)
        return None, grad_input, grad_weight, grad_bias
