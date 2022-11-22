# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Optional

import torch
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseBatchSoftmaxCtx(SparseCtxBase):

    def __init__(self, compressed: bool, temperature: float = 1):
        super().__init__()

        self._compressed = compressed
        self._T = np.float32(1 / temperature)
        self._batch_size: int = None

        for kernel_name, kernel_class, first_tensor in zip(
            ['forward:y', 'backward:x'],
            [SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel],
            ['x', 'grad_y'],
        ):
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={'sparta': kernel_class},
                args={'compressed': compressed},
                mask_map={'x': first_tensor},
            )

    def set_temperature(self, temperature: float):
        self._T = np.float32(1 / temperature)

    def set_shape(self, batch_size: int, H: int, W: int):
        self._kernels['forward:y'].set_shape(batch_size, H, W)
        self._kernels['backward:x'].set_shape(batch_size, H, W)
        self._batch_size = batch_size

    def get_conditions(self, impls: Dict[str, str]):
        if self._compressed and len(impls) > 1:
            return [
                ['forward:y;BLOCK_SIZE_H_VALUE', 'backward:x;BLOCK_SIZE_H_VALUE'],
                ['forward:y;BLOCK_SIZE_W_VALUE', 'backward:x;BLOCK_SIZE_W_VALUE'],
            ]
        else:
            return []

    def build(self, config: Dict[str, Dict[str, Any]]):
        super().build(config)
        forward_kernel = self._kernels['forward:y'].active_kernel()
        if forward_kernel is not None:
            self.forward = lambda x: forward_kernel(x, self._T)
        backward_kernel = self._kernels['backward:x'].active_kernel()
        if backward_kernel is not None:
            self.backward = lambda grad, output: backward_kernel(grad, output, self._T)

    def set_sample_inputs(
        self, sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None
    ):
        x = sample_inputs[0]
        self._kernels['forward:y'].set_sample_inputs([x, self._T])
        if sample_grads is not None:
            grad_y = sample_grads[0]
            y = self._kernels['forward:y'].dense_func(x, self._T)
            self._kernels['backward:x'].set_sample_inputs([grad_y, y, self._T])

    def get_connections(self, backward: bool = False):
        if self._compressed and backward:
            return [
                {'forward:y': 'BLOCK_SIZE_H_VALUE', 'backward:x': 'BLOCK_SIZE_H_VALUE'},
                {'forward:y': 'BLOCK_SIZE_W_VALUE', 'backward:x': 'BLOCK_SIZE_W_VALUE'},
            ]
        else:
            return []

    def dense_forward(self, *args):
        return self._kernels['forward:y'].dense_func(*args, self._T)


class SparseBatchSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sparta_ctx: SparseBatchSoftmaxCtx, x: torch.Tensor
    ):
        ctx.sparta_ctx = sparta_ctx
        output = sparta_ctx.forward(x.detach())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor):
        output, = ctx.saved_tensors
        return None, ctx.sparta_ctx.backward(grad.detach(), output.detach())
