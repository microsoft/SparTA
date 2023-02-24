# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch
import numpy as np

from sparta.kernels import KernelBase, SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.operators.operator_base import Port, SparsityAttr, SparseOperator, SparseAutoGrad
from sparta.testing import sparse_softmax_forward_reference, sparse_softmax_backward_reference

class SparseBatchSoftmaxForward(SparseOperator):

    __batched__ = True
    __direction__ = 'forward'

    def __init__(self, compressed: bool = False, temperature: Optional[float] = 1):
        super().__init__()

        self._compressed = compressed
        self._T = None if temperature is None else np.float32(1 / temperature)

        self._sparse_port = 'y'
        sparse_attr = SparsityAttr(True, False)
        for port_name in ['x', 'y']:
            self.ports[port_name] = Port(self, port_name)
            self.ports[port_name].attr = sparse_attr

        self._kernels[self.__direction__] = {
            'sparta': {
                'forward': SparTASparseSoftmaxForwardKernel,
                'backward': SparTASparseSoftmaxBackwardKernel,
            }[self.__direction__](
                compressed=compressed,
                batched=self.__batched__,
            ),
        }

        self.shape: Tuple[int, int, int] = None

    def set_temperature(self, temperature: float):
        self._T = np.float32(1 / temperature)

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        x = sample_inputs[0]
        if self.__batched__:
            batch_size = x.shape[0]
        else:
            batch_size = 1
        H, W = x.shape[-2:]
        self.shape = (batch_size, H, W)
        self.ports['x'].set_data(x)
        if self._T is None:
            self.set_temperature(np.sqrt(W))

    def _compile_kernel(self, kernel_name: str, kernel: KernelBase, params: Dict[str, Any]):
        sparse_attr = self.get_sparse_attr()
        sparse_attr.set_block_size(
            block_H=params['BLOCK_SIZE_H_VALUE'],
            block_W=params['BLOCK_SIZE_W_VALUE'],
        )
        kernel.set_parameter('MAX_W_VALUE', self.shape[-1])
        kernel.compile(params, self.shape, sparse_attr)

    def _set_forward(self):
        if self.__direction__ in self._compiled_kernels:
            kernel = self._compiled_kernels[self.__direction__]
            sparse_attr = self.get_sparse_attr()
            self.forward_func = lambda *inputs: kernel(*inputs, sparse_attr.mask, self._T)

    def _kernel_func_call(self, kernel_name: str):
        x = self.ports['x'].get_data(compressed=self._compressed)
        sparse_attr = self.get_sparse_attr()
        kernel = self._compiled_kernels[kernel_name]
        return lambda : kernel(x, sparse_attr.mask, self._T)

    def _kernel_reference(self, kernel_name: str):
        return self.ports['y'].get_data(compressed=self._compressed)

    def _calc_kernel_flops(self, kernel_name: str):
        indexes = self.get_sparse_attr().indexes
        sparse_rate = indexes.block_nnz / indexes.row_num / indexes.col_num
        return np.prod(self.shape) * sparse_rate

    def reference(self, *inputs):
        if len(inputs) > 0:
            self._read_sample_inputs(inputs)
        x = self.ports['x'].get_data(compressed=False)
        mask = self.get_sparse_attr().mask
        y = sparse_softmax_forward_reference(x, mask, 1 / self._T)
        self.ports['y'].set_data(y)
        return y


class SparseSoftmaxForward(SparseBatchSoftmaxForward):

    __batched__ = False


class SparseBatchSoftmaxBackward(SparseBatchSoftmaxForward):

    __batched__ = True
    __direction__ = 'backward'

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        pass

    def _kernel_func_call(self, kernel_name: str):
        gy = self.ports['y'].get_data(grad=True, compressed=self._compressed)
        y = self.ports['y'].get_data(grad=False, compressed=self._compressed)
        sparse_attr = self.get_sparse_attr()
        kernel = self._compiled_kernels[kernel_name]
        return lambda : kernel(gy, y, sparse_attr.mask, self._T)

    def _kernel_reference(self, kernel_name: str):
        return self.ports['x'].get_data(grad=True, compressed=self._compressed)

    def reference(self, *inputs):
        pass


class SparseSoftmaxBackward(SparseBatchSoftmaxBackward):

    __batched__ = False


class _SparseSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        func: SparseAutoGrad,
        x: torch.Tensor,
    ):
        ctx.backward = func.backward_op
        y = func.forward_func(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor):
        y = ctx.saved_tensors[0]
        if ctx.needs_input_grad[1]:
            return None, ctx.backward(grad, y)
        else:
            return None, None


class SparseBatchSoftmax(SparseAutoGrad, SparseBatchSoftmaxForward):

    __static_func__ = _SparseSoftmax

    def __init__(self, compressed: bool = False, temperature: Optional[float] = 1):
        super().__init__(compressed, temperature)
        self._set_backward(SparseBatchSoftmaxBackward(compressed, temperature))
        self.backward_op.ports = self.ports

    def set_temperature(self, temperature: float):
        super().set_temperature(temperature)
        self.backward_op.set_temperature(temperature)

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        super()._read_sample_inputs(sample_inputs)
        self.backward_op.shape = self.shape


class SparseSoftmax(SparseAutoGrad, SparseSoftmaxForward):

    __static_func__ = _SparseSoftmax

    def __init__(self, compressed: bool = False, temperature: Optional[float] = 1):
        super().__init__(compressed, temperature)
        self._set_backward(SparseSoftmaxBackward(compressed, temperature))
        self.backward_op.ports = self.ports

    def set_temperature(self, temperature: float):
        super().set_temperature(temperature)
        self.backward_op.set_temperature(temperature)

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        super()._read_sample_inputs(sample_inputs)
        self.backward_op.shape = self.shape
