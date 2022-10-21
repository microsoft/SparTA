# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional

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

        sparse_tensor = select('s', sparse_type, 'ABC')
        backward_A_order = 'BCA' if transpose_A else 'CBA'
        backward_B_order = 'CAB' if transpose_B else 'ACB'
        if sparse_tensor == 'A' and transpose_A:
            bcsr_main = 'V'
        elif sparse_tensor == 'B' and not transpose_B:
            bcsr_main = 'V'
        else:
            bcsr_main = 'H'

        self._kernels['forward:C'] = KernelPlaceholder(
            name='forward:C',
            cat='forward:C',
            impls={
                'sparta': SparTASparseMatMulKernel,
                'openai': OpenAISparseMatMulKernel,
            },
            args={
                'biased': biased,
                'bcsr_main': bcsr_main,
                'compressed': compressed,
                'transpose_A': transpose_A,
                'transpose_B': transpose_B,
                'sparse_type': sparse_type,
            },
            mask_map={sparse_tensor: sparse_tensor},
        )
        self._kernels['backward:A'] = KernelPlaceholder(
            name='backward:A',
            cat='backward:A',
            impls={
                'sparta': SparTASparseMatMulKernel,
                'openai': OpenAISparseMatMulKernel,
            },
            args={
                'biased': False,
                'bcsr_main': bcsr_main,
                'compressed': compressed,
                'transpose_A': transpose_A and transpose_B,
                'transpose_B': transpose_A or not transpose_B,
                'sparse_type': rearange(sparse_type, 'ABC', backward_A_order),
            },
            mask_map={sparse_tensor: select(sparse_tensor, backward_A_order, 'ABC')},
        )
        self._kernels['backward:B'] = KernelPlaceholder(
            name='backward:B',
            cat='backward:B',
            impls={
                'sparta': SparTASparseMatMulKernel,
                'openai': OpenAISparseMatMulKernel,
            },
            args={
                'biased': False,
                'bcsr_main': bcsr_main,
                'compressed': compressed,
                'transpose_A': not transpose_A or transpose_B,
                'transpose_B': transpose_A and transpose_B,
                'sparse_type': rearange(sparse_type, 'ABC', backward_B_order),
            },
            mask_map={sparse_tensor: select(sparse_tensor, backward_B_order, 'ABC')},
        )

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
        return []

    def forward_C(self, *args):
        return self._kernels['forward:C'].active_kernel()(*args)

    def backward_A(self, grad_C: torch.Tensor, B: torch.Tensor):
        if self._compressed:
            if self._sparse_type == 'dsd':
                B = self._kernels['forward:C'].get_converter('B').swapaxes(B)
        if self._transpose_A:
            return self._kernels['backward:A'].active_kernel()(B, grad_C)
        else:
            return self._kernels['backward:A'].active_kernel()(grad_C, B)

    def backward_B(self, grad_C: torch.Tensor, A: torch.Tensor):
        if self._compressed:
            if self._sparse_type == 'sdd':
                A = self._kernels['forward:C'].get_converter('A').swapaxes(A)
            if self._sparse_type == 'dds':
                grad_C = self._kernels['forward:C'].get_converter('C').swapaxes(grad_C)
        if self._transpose_B:
            return self._kernels['backward:B'].active_kernel()(grad_C, A)
        else:
            return self._kernels['backward:B'].active_kernel()(A, grad_C)

    def backward_bias(self, grad_C: torch.Tensor):
        if self._sparse_type == 'dds' and self._compressed:
            return self._kernels['forward:C'].get_converter('C').sum(grad_C, axis=-2)
        else:
            return grad_C.sum(-2)


class SparseBatchMatMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, sparta_ctx: SparseBatchMatMulCtx,
        input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ):
        ctx.save_for_backward(input, weight, bias)
        ctx.sparta_ctx = sparta_ctx
        # output = input.mm(weight.t())
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        # return output
        if bias is None:
            return sparta_ctx.forward_C(input.detach(), weight.detach())
        else:
            return sparta_ctx.forward_C(input.detach(), weight.detach(), bias.detach())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[1]:
            # grad_input = grad_output.mm(weight)
            grad_input = ctx.sparta_ctx.backward_A(grad_output.detach(), weight.detach())
        if ctx.needs_input_grad[2]:
            # grad_weight = grad_output.t().mm(input)
            grad_weight = ctx.sparta_ctx.backward_B(grad_output.detach(), input.detach())
        if bias is not None and ctx.needs_input_grad[3]:
            # grad_bias = grad_output.sum(-2)
            grad_bias = ctx.sparta_ctx.backward_bias(grad_output.detach())

        return None, grad_input, grad_weight, grad_bias
