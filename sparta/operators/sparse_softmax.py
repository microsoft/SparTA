# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Tuple, Optional

import torch
import numpy as np

from sparta.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.operators.operator_base import Port, SparseOperator, SparseAutoGrad

class SparseBatchSoftmaxForward(SparseOperator):

    __batched__ = True
    __direction__ = 'forward'

    def __init__(self, compressed: bool = False, temperature: Optional[float] = 1):
        super().__init__()

        self._compressed = compressed
        self._T = None if temperature is None else np.float32(1 / temperature)

        self._set_kernel_group(self.__direction__, {
            'sparta': {
                'forward': SparTASparseSoftmaxForwardKernel,
                'backward': SparTASparseSoftmaxBackwardKernel,
            }[self.__direction__](
                compressed=compressed,
                batched=self.__batched__,
            )
        })

        self._sparse_port = 'y'
        sparse_attr = self.kernel_groups[self.__direction__].attr
        for port_name in ['x', 'y']:
            self.ports[port_name] = Port(self, port_name)
            self.ports[port_name].attr = sparse_attr
        if compressed:
            self._attr = sparse_attr

    def set_temperature(self, temperature: float):
        self._T = np.float32(1 / temperature)

    def _get_sample_inputs(self, kernel_name: str):
        return [
            self.ports['x'].get_sample_data(),
            self.ports['x'].attr.mask,
            self._T,
        ]

    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        x = sample_inputs[0]
        batch_size = x.shape[0] if self.__batched__ else 1
        H, W = x.shape[-2:]
        return (batch_size, H, W)

    def _set_sample_shape(self, sample_shape: Tuple):
        self.kernel_groups[self.__direction__].set_sample_shape(sample_shape)
        if self._T is None:
            self.set_temperature(np.sqrt(sample_shape[-1]))

    def _set_forward(self):
        kernel_group = self.kernel_groups[self.__direction__]
        attr = kernel_group.attr
        if kernel_group.ready:
            def forward_func(x):
                return kernel_group.active_kernel(x, attr.mask, self._T)
        else:
            def forward_func(x):
                self.ports['x'].sample_data = x
                y = kernel_group.active_kernel(x, attr.mask, self._T)
                self.ports['y'].sample_data = y
                return y
        self.forward_func = forward_func


class SparseSoftmaxForward(SparseBatchSoftmaxForward):

    __batched__ = False


class SparseBatchSoftmaxBackward(SparseBatchSoftmaxForward):

    __batched__ = True
    __direction__ = 'backward'

    def _get_sample_inputs(self, kernel_name: str):
        return [
            self.ports['y'].get_sample_data(grad=True),
            self.ports['y'].get_sample_data(),
            self.ports['y'].attr.mask,
            self._T,
        ]

    def _set_forward(self):
        kernel_group = self.kernel_groups[self.__direction__]
        attr = kernel_group.attr
        def backward_func(gy, y):
            return kernel_group.active_kernel(gy, y, attr.mask, self._T)
        self.forward_func = backward_func


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

    def set_temperature(self, temperature: float):
        super().set_temperature(temperature)
        self.backward_op.set_temperature(temperature)

    def _set_sample_shape(self, sample_shape: Tuple):
        super()._set_sample_shape(sample_shape)
        self.backward_op._set_sample_shape(sample_shape)


class SparseSoftmax(SparseAutoGrad, SparseSoftmaxForward):

    __static_func__ = _SparseSoftmax

    def __init__(self, compressed: bool = False, temperature: Optional[float] = 1):
        super().__init__(compressed, temperature)
        self._set_backward(SparseSoftmaxBackward(compressed, temperature))

    def set_temperature(self, temperature: float):
        super().set_temperature(temperature)
        self.backward_op.set_temperature(temperature)

    def _set_sample_shape(self, sample_shape: Tuple):
        super()._set_sample_shape(sample_shape)
        self.backward_op._set_sample_shape(sample_shape)
