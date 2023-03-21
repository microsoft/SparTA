# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Tuple, Optional

import torch

from sparta.kernels import SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.operators import Port, SparseOperator, SparseAutoGrad


class SparseBatchMatMulForward(SparseOperator):

    __batched__ = True

    def __init__(
        self,
        mode: str,
        transpose_A: bool,
        transpose_B: bool,
        biased: bool,
        compressed: bool,
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse matmul mode: {mode}')

        super().__init__(compressed)

        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._biased = biased
        self._compressed = compressed

        specs = {
            'mode': mode,
            'biased': biased,
            'transpose_A': transpose_A,
            'transpose_B': transpose_B,
            'compressed': compressed,
            'batched': self.__batched__,
        }
        self._set_kernel_group('forward', {
            'sparta': SparTASparseMatMulKernel(**specs),
            'openai': OpenAISparseMatMulKernel(**specs),
        })

        for p in ['A', 'B', 'C', 'bias'] if biased else ['A', 'B', 'C']:
            self.ports[p] = Port(self, p)
        self._sparse_port = self.ports['ABC'[mode.find('s')]]
        self._sparse_port.get_attr = lambda : self.kernel_groups['forward'].attr
        self._sparse_port.compressed = compressed

        self._set_forward()

    def _get_sample_inputs(self, kernel_name: str):
        return [
            self.ports[p].get_sample_data()
            for p in (['A', 'B', 'bias'] if self._biased else ['A', 'B'])
        ]

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        # TODO: check shape conflicts
        if self.__batched__:
            batch_size = sample_inputs[0].shape[0]
        else:
            batch_size = 1
        if self._transpose_A:
            K, M = sample_inputs[0].shape[-2:]
        else:
            M, K = sample_inputs[0].shape[-2:]
        if self._transpose_B:
            N, K = sample_inputs[1].shape[-2:]
        else:
            K, N = sample_inputs[1].shape[-2:]
        return (batch_size, M, K, N)

    def _set_sample_shape(self, sample_shape: Tuple):
        self.kernel_groups['forward'].set_sample_shape(sample_shape)

    def _set_forward(self):
        kernel_group = self.kernel_groups['forward']
        if kernel_group.ready:
            self.forward_func = kernel_group.active_kernel
        else:
            def forward_func(*inputs):
                self.ports['A'].sample_data = inputs[0]
                self.ports['B'].sample_data = inputs[1]
                if self._biased:
                    self.ports['bias'].sample_data = inputs[2]
                C = kernel_group.active_kernel(*inputs)
                self.ports['C'].sample_data = C
                return C
            self.forward_func = forward_func


class SparseMatMulForward(SparseBatchMatMulForward):

    __batched__ = False


class SparseBatchMatMulBackward(SparseOperator):

    __batched__ = True

    def __init__(
        self,
        mode: str,
        transpose_A: bool,
        transpose_B: bool,
        biased: bool,
        compressed: bool,
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse matmul mode: {mode}')

        super().__init__(compressed)

        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._biased = biased
        self._compressed = compressed

        A_spec = {
            'mode': ''.join(mode[i] for i in ([1, 2, 0] if transpose_A else [2, 1, 0])),
            'biased': False,
            'transpose_A': transpose_A and transpose_B,
            'transpose_B': transpose_A or not transpose_B,
            'compressed': compressed,
            'batched': self.__batched__,
        }
        B_spec = {
            'mode': ''.join(mode[i] for i in ([2, 0, 1] if transpose_B else [0, 2, 1])),
            'biased': False,
            'transpose_A': not transpose_A or transpose_B,
            'transpose_B': transpose_A and transpose_B,
            'compressed': compressed,
            'batched': self.__batched__,
        }

        self._set_kernel_group('backward:A', {
            'sparta': SparTASparseMatMulKernel(**A_spec),
            'openai': OpenAISparseMatMulKernel(**A_spec),
        })
        self._set_kernel_group('backward:B', {
            'sparta': SparTASparseMatMulKernel(**B_spec),
            'openai': OpenAISparseMatMulKernel(**B_spec),
        })

        # TODO: connect sparsity attrs for single use
        # TODO: set ports for single use
        # TODO: set forward function for single use

    def _get_sample_inputs(self, kernel_name: str):
        grad_C = self.ports['C'].get_sample_data(grad=True)
        if kernel_name == 'backward:A':
            B = self.ports['B'].get_sample_data()
            if self._transpose_A:
                return [B, grad_C]
            else:
                return [grad_C, B]
        elif kernel_name == 'backward:B':
            A = self.ports['A'].get_sample_data()
            if self._transpose_B:
                return [grad_C, A]
            else:
                return [A, grad_C]
        else:
            raise ValueError(f'unrecognized kernel name: {kernel_name}')

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        pass

    def _set_sample_shape(self, sample_shape: Tuple):
        batch_size, M, K, N = sample_shape
        if self._transpose_A:
            self.kernel_groups['backward:A'].set_sample_shape((batch_size, K, N, M))
        else:
            self.kernel_groups['backward:A'].set_sample_shape((batch_size, M, N, K))
        if self._transpose_B:
            self.kernel_groups['backward:B'].set_sample_shape((batch_size, N, M, K))
        else:
            self.kernel_groups['backward:B'].set_sample_shape((batch_size, K, M, N))

    def _set_forward(self):
        kg_A = self.kernel_groups['backward:A']
        kg_B = self.kernel_groups['backward:B']
        if self._transpose_A:
            backward_A = lambda grad_C, B: kg_A.active_kernel(B, grad_C)
        else:
            backward_A = lambda grad_C, B: kg_A.active_kernel(grad_C, B)
        if self._transpose_B:
            backward_B = lambda grad_C, A: kg_B.active_kernel(grad_C, A)
        else:
            backward_B = lambda grad_C, A: kg_B.active_kernel(A, grad_C)
        C_attr = self.ports['C'].get_attr()
        if C_attr is not None and self._compressed:
            backward_bias = lambda grad_C: C_attr.indexes.sum(grad_C, axis=-2)
        else:
            backward_bias = lambda grad_C: grad_C.sum(-2)

        def backward(grad, A, B, needs_grad):
            grad_A, grad_B, grad_bias = None, None, None
            if needs_grad[1]:
                grad_A = backward_A(grad, B)
            if needs_grad[2]:
                grad_B = backward_B(grad, A)
            if self._biased and needs_grad[3]:
                grad_bias = backward_bias(grad)
            return grad_A, grad_B, grad_bias

        self.forward_func = backward


class SparseMatMulBackward(SparseBatchMatMulBackward):

    __batched__ = False


class _SparseMatMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        func: SparseAutoGrad,
        *inputs,
    ):
        ctx.save_for_backward(inputs[0], inputs[1])
        ctx.backward = func.backward_op
        return func.forward_func(*inputs)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor):
        A, B = ctx.saved_tensors
        return None, *ctx.backward(grad, A, B, ctx.needs_input_grad)


class SparseBatchMatMul(SparseAutoGrad, SparseBatchMatMulForward):

    __static_func__ = _SparseMatMul

    def __init__(
        self,
        mode: str,
        transpose_A: bool,
        transpose_B: bool,
        biased: bool,
        compressed: bool = True,
    ):
        super().__init__(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
        )
        self._set_backward(SparseBatchMatMulBackward(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
        ))

    def _set_sample_shape(self, sample_shape: Tuple):
        super()._set_sample_shape(sample_shape)
        self.backward_op._set_sample_shape(sample_shape)


class SparseMatMul(SparseAutoGrad, SparseMatMulForward):

    __static_func__ = _SparseMatMul

    def __init__(
        self,
        mode: str,
        transpose_A: bool,
        transpose_B: bool,
        biased: bool,
        compressed: bool = True,
    ):
        super().__init__(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
        )
        self._set_backward(SparseMatMulBackward(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
        ))

    def _set_sample_shape(self, sample_shape: Tuple):
        super()._set_sample_shape(sample_shape)
        self.backward_op._set_sample_shape(sample_shape)


class SparseLinear(SparseMatMul):

    def __init__(self, raw_module: torch.nn.Linear, mode: str = 'dsd'):
        self._biased = raw_module.bias is not None
        self._compressed = mode == 'dsd'
        super().__init__(mode, False, True, self._biased, self._compressed)
        self.weight: torch.nn.Parameter = None
        self.bias = raw_module.bias
        self.ports['B'].sample_data = raw_module.weight
        if self._biased:
            self.ports['bias'].sample_data = raw_module.bias
            self.forward = self._forward_with_bias
        else:
            self.forward = self._forward_without_bias

    def _forward_with_bias(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self.weight, self.bias)

    def _forward_without_bias(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self.weight)

    def _update_weight(self):
        weight = self.ports['B'].get_sample_data()
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def set_mask(self, mask: torch.Tensor):
        super().set_mask(mask)
        self._update_weight()

    def _post_build(self):
        self._update_weight()
        return super()._post_build()

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        sample_inputs.append(self.ports['B'].sample_data)
        return super()._read_sample_inputs(sample_inputs)
