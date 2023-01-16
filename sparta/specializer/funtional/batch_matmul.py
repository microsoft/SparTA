# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch

from sparta.specializer.kernels import SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseBatchMatMulCtx(SparseCtxBase):

    def __init__(
        self,
        mode: str,
        transpose_A: bool,
        transpose_B: bool,
        biased: bool,
        compressed: bool,
    ):
        super().__init__()

        self._biased = biased
        self._compressed = compressed
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._mode = mode

        def select(x: str, source: str, target: str):
            return target[source.find(x)]

        def rearange(s: str, source_order: str, target_order: str):
            return ''.join(select(x, source_order, s) for x in target_order)

        def calc_tesa_shape(mode: str, trans_A: bool, trans_B: bool):
            if mode == 'sdd':
                return ('K', 'M') if trans_A else ('M', 'K')
            elif mode == 'dsd':
                return ('N', 'K') if trans_B else ('K', 'N')
            else:
                return ('M', 'N')

        sparse_tensor = select('s', mode, 'ABC')
        self.sparse_ports[sparse_tensor] = []

        self._tesa_shapes: Dict[str, Tuple[str, str]] = {}
        for kernel_name, bias, target_order, trans_A, trans_B in zip(
            ['forward:C', 'backward:A', 'backward:B'],
            [biased, False, False],
            ['ABC', 'BCA' if transpose_A else 'CBA', 'CAB' if transpose_B else 'ACB'],
            [transpose_A, transpose_A and transpose_B, not transpose_A or transpose_B],
            [transpose_B, transpose_A or not transpose_B, transpose_A and transpose_B],
        ):
            s_type = rearange(mode, 'ABC', target_order)
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={
                    'sparta': SparTASparseMatMulKernel,
                    'openai': OpenAISparseMatMulKernel,
                },
                args={
                    'biased': bias,
                    'compressed': compressed,
                    'transpose_A': trans_A,
                    'transpose_B': trans_B,
                    'mode': s_type,
                },
                port_map={sparse_tensor: select(sparse_tensor, target_order, 'ABC')},
                connectable=compressed,
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

    def build(self, config: Dict[str, Dict[str, Any]]):
        super().build(config)
        forward_kernel = self._kernels['forward:C'].active_kernel()
        if forward_kernel is not None:
            if self._biased:
                self.forward_C = lambda A, B, bias: forward_kernel(A, B, bias)
            else:
                self.forward_C = lambda A, B: forward_kernel(A, B)
            if self._mode == 'dds' and self._compressed:
                C_indexes = self._kernels['forward:C'].active_kernel().ports['C'].indexes
                self.backward_bias = lambda grad_C: C_indexes.sum(grad_C, axis=-2)
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

    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        A = sample_inputs[0]
        B = sample_inputs[1]
        if self._biased:
            bias = sample_inputs[2].detach()
            self._kernels['forward:C'].set_sample_inputs([A, B, bias])
        else:
            self._kernels['forward:C'].set_sample_inputs([A, B])
        if sample_grads is not None:
            grad_C = sample_grads[0]
            if self._transpose_A:
                self._kernels['backward:A'].set_sample_inputs([B, grad_C])
            else:
                self._kernels['backward:A'].set_sample_inputs([grad_C, B])
            if self._transpose_B:
                self._kernels['backward:B'].set_sample_inputs([grad_C, A])
            else:
                self._kernels['backward:B'].set_sample_inputs([A, grad_C])

    def get_connections(self, backward: bool = False):
        if self._compressed and backward:
            conditions = [{}, {}]
            for kernel_name, tesa_shapes in self._tesa_shapes.items():
                for k, dim in enumerate(tesa_shapes):
                    conditions[k][kernel_name] = f'BLOCK_SIZE_{dim}_VALUE'
            return conditions
        else:
            return []

    def dense_forward(self, *args):
        return self._kernels['forward:C'].dense_func(*args)


class SparseBatchMatMulFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sparta_ctx: SparseBatchMatMulCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
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
