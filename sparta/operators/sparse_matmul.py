# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch
import numpy as np

from sparta.kernels import KernelBase, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.operators import Port, SparsityAttr, SparseOperator, SparseAutoGrad


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

        super().__init__()

        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._biased = biased
        self._compressed = compressed

        self._sparse_axis = {
            'sdd': ['K', 'M'] if transpose_A else ['M', 'K'],
            'dsd': ['N', 'K'] if transpose_B else ['K', 'N'],
            'dds': ['M', 'N'],
        }[mode]
        self._BCSR = {
            'sdd': not transpose_A,
            'dsd': transpose_B,
            'dds': True,
        }[mode]

        self._sparse_port = 'ABC'[mode.find('s')]
        self.ports['A'] = Port(self, 'A')
        self.ports['B'] = Port(self, 'B')
        self.ports['C'] = Port(self, 'C', fine_mask=False)  # DDS known issue
        if biased:
            self.ports['bias'] = Port(self, 'bias')
        self.ports[self._sparse_port].attr = SparsityAttr(self._BCSR, not self._BCSR)

        specs = {
            'mode': mode,
            'biased': biased,
            'transpose_A': transpose_A,
            'transpose_B': transpose_B,
            'compressed': compressed,
            'batched': self.__batched__,
        }
        self._kernels['forward'] = {
            'sparta': SparTASparseMatMulKernel(**specs),
            'openai': OpenAISparseMatMulKernel(**specs),
        }

        self.shape: Tuple[int, int, int, int] = None

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
        self.shape = (batch_size, M, K, N)
        self.ports['A'].set_data(sample_inputs[0])
        self.ports['B'].set_data(sample_inputs[1])
        if self._biased:
            self.ports['bias'].set_data(sample_inputs[2])

    def _compile_kernel(self, kernel_name: str, kernel: KernelBase, params: Dict[str, Any]):
        sparse_attr = self.get_sparse_attr()
        kernel.set_parameter('BCSR', self._BCSR or sparse_attr.BCSR)
        kernel.set_parameter('BCSC', not self._BCSR)
        block_size = [params[f'BLOCK_SIZE_{axis}_VALUE'] for axis in self._sparse_axis]
        sparse_attr.set_block_size(*block_size)
        kernel.compile(params, self.shape, sparse_attr)

    def _set_forward(self):
        if 'forward' in self._compiled_kernels:
            self.forward_func = self._compiled_kernels['forward']

    def _kernel_func_call(self, kernel_name: str):
        A = self.ports['A'].get_data(compressed=self._compressed)
        B = self.ports['B'].get_data(compressed=self._compressed)
        kernel = self._compiled_kernels[kernel_name]
        if self._biased:
            bias = self.ports['bias'].get_data()
            return lambda : kernel(A, B, bias)
        else:
            return lambda : kernel(A, B)

    def _kernel_reference(self, kernel_name: str):
        return self.ports['C'].get_data(compressed=self._compressed)

    def _calc_kernel_flops(self, kernel_name: str):
        indexes = self.get_sparse_attr().indexes
        sparse_rate = indexes.block_nnz / indexes.row_num / indexes.col_num
        return np.prod(self.shape) * sparse_rate

    def reference(self, *inputs):
        if len(inputs) > 0:
            self._read_sample_inputs(inputs)
        A = self.ports['A'].get_data(compressed=False)
        B = self.ports['B'].get_data(compressed=False)
        if self._transpose_A:
            A = A.swapaxes(self.__batched__ + 0, self.__batched__ + 1)
        if self._transpose_B:
            B = B.swapaxes(self.__batched__ + 0, self.__batched__ + 1)
        C = torch.bmm(A, B) if self.__batched__ else torch.mm(A, B)
        if self._biased:
            bias = self.ports['bias'].get_data()
            C += bias.unsqueeze(1) if self.__batched__ else bias
        self.ports['C'].set_data(C)
        return C


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
        ports: Dict[str, Port],
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise ValueError(f'invalid sparse matmul mode: {mode}')

        super().__init__()

        self._mode = mode
        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        self._biased = biased
        self._compressed = compressed

        self.ports = ports
        self._sparse_port = 'ABC'[mode.find('s')]

        self._BCSR = {
            'backward:A': {
                'sdd': True,
                'dsd': not transpose_B,
                'dds': True,
            }[mode],
            'backward:B': {
                'sdd': transpose_A,
                'dsd': True,
                'dds': False,
            }[mode],
        }
        self.get_sparse_attr().update_axis(True, True)

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

        self._kernels['backward:A'] = {
            'sparta': SparTASparseMatMulKernel(**A_spec),
            'openai': OpenAISparseMatMulKernel(**A_spec),
        }
        self._kernels['backward:B'] = {
            'sparta': SparTASparseMatMulKernel(**B_spec),
            'openai': OpenAISparseMatMulKernel(**B_spec),
        }

        self.shape: Tuple[int, int, int, int] = None

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        pass

    def _compile_kernel(self, kernel_name: str, kernel: KernelBase, params: Dict[str, Any]):
        batch, M, K, N = self.shape
        shape = {
            'backward:A': (batch, K, N, M) if self._transpose_A else (batch, M, N, K),
            'backward:B': (batch, N, M, K) if self._transpose_B else (batch, K, M, N),
        }[kernel_name]
        sparse_attr = self.get_sparse_attr()
        kernel.set_parameter('BCSR', self._BCSR[kernel_name] or sparse_attr.BCSR)
        kernel.set_parameter('BCSC', not self._BCSR[kernel_name])
        kernel.compile(params, shape, sparse_attr)

    def _set_forward(self):
        if 'backward:A' in self._compiled_kernels:
            kernel_A = self._compiled_kernels['backward:A']
        else:
            kernel_A = lambda *inputs: None
        if 'backward:B' in self._compiled_kernels:
            kernel_B = self._compiled_kernels['backward:B']
        else:
            kernel_B = lambda *inputs: None
        if self._transpose_A:
            backward_A = lambda grad_C, B: kernel_A(B, grad_C)
        else:
            backward_A = lambda grad_C, B: kernel_A(grad_C, B)
        if self._transpose_B:
            backward_B = lambda grad_C, A: kernel_B(grad_C, A)
        else:
            backward_B = lambda grad_C, A: kernel_B(A, grad_C)
        if self._mode == 'dds' and self._compressed:
            C_attr = self.ports['C'].attr
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

    def _kernel_func_call(self, kernel_name: str):
        grad_C = self.ports['C'].get_data(grad=True, compressed=self._compressed)
        kernel = self._compiled_kernels[kernel_name]
        if kernel_name == 'backward:A':
            B = self.ports['B'].get_data(compressed=self._compressed)
            if self._transpose_A:
                return lambda : kernel(B, grad_C)
            else:
                return lambda : kernel(grad_C, B)
        elif kernel_name == 'backward:B':
            A = self.ports['A'].get_data(compressed=self._compressed)
            if self._transpose_A:
                return lambda : kernel(grad_C, A)
            else:
                return lambda : kernel(A, grad_C)
        else:
            raise ValueError(f'kernel not found: {kernel_name}')

    def _kernel_reference(self, kernel_name: str):
        if kernel_name == 'backward:A':
            return self.ports['A'].get_data(grad=True, compressed=False)
        elif kernel_name == 'backward:B':
            return self.ports['B'].get_data(grad=True, compressed=False)
        else:
            raise ValueError(f'kernel not found: {kernel_name}')

    def _calc_kernel_flops(self, kernel_name: str):
        indexes = self.get_sparse_attr().indexes
        sparse_rate = indexes.block_nnz / indexes.row_num / indexes.col_num
        return np.prod(self.shape) * sparse_rate

    def reference(self, *inputs):
        pass


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
        super().__init__(mode, transpose_A, transpose_B, biased, compressed)
        self._set_backward(SparseBatchMatMulBackward(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
            ports=self.ports,
        ))

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        super()._read_sample_inputs(sample_inputs)
        self.backward_op.shape = self.shape


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
        super().__init__(mode, transpose_A, transpose_B, biased, compressed)
        self._set_backward(SparseMatMulBackward(
            mode=mode,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            biased=biased,
            compressed=compressed,
            ports=self.ports,
        ))

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        super()._read_sample_inputs(sample_inputs)
        self.backward_op.shape = self.shape


class SparseLinear(SparseMatMul):

    def __init__(self, raw_module: torch.nn.Linear, mode: str = 'dsd'):
        self._biased = raw_module.bias is not None
        self._compressed = mode == 'dsd'
        super().__init__(mode, False, True, self._biased, self._compressed)
        self.weight: torch.nn.Parameter = None
        self.bias = raw_module.bias
        self.ports['B'].set_data(raw_module.weight)
        if self._biased:
            self.ports['bias'].set_data(raw_module.bias)
            self.forward = self._forward_with_bias
        else:
            self.forward = self._forward_without_bias

    def _update_weight(self):
        if 'forward' in self._compiled_kernels:
            weight = self.ports['B'].get_data(compressed=self._compressed)
            self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def set_mask(self, mask: torch.Tensor):
        super().set_mask(mask)
        self._update_weight()

    def build(
        self,
        config: Dict[str, Dict[str, Any]],
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ):
        super().build(config, sample_inputs)
        self._update_weight()

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        sample_inputs.append(self.ports['B'].get_data())
        if self._biased:
            sample_inputs.append(self.ports['bias'].get_data())
        super()._read_sample_inputs(sample_inputs)

    def _forward_with_bias(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self.weight, self.bias)

    def _forward_without_bias(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self.weight)
