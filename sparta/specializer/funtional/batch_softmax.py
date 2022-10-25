# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Optional

import torch
import numpy as np

from sparta.specializer.kernels import SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseBatchSoftmaxCtx(SparseCtxBase):

    def __init__(self, compressed: bool, temperature: Optional[float] = None):
        super().__init__()

        self._compressed = compressed
        self._T = np.float32(1. if temperature is None else 1 / temperature)
        self._mask: torch.Tensor = None
        self._batch_size: int = None

        for kernel_name, kernel_class, first_tensor in zip(
            ['forward:y', 'backward:x'],
            [SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel],
            ['x', 'grad_y'],
        ):
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                impls={'sparta': kernel_class},
                args={'compressed': compressed},
                mask_map={'x': first_tensor},
            )

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

    def build(self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        super().build(config, mask)
        self._mask = mask['x'].to(torch.int32)
        if self._compressed:
            self._mask = self.get_converter('forward:y', 'x').convert(self._mask)

    def _split_graph(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None
    ):
        funcs, inputs = [], []
        if 'forward:y' in kernels:
            funcs.append(self.forward)
            inputs.append([sample_inputs['x'], self._mask, self._T])
        if 'backward:x' in kernels:
            funcs.append(self.backward)
            output = self.forward(sample_inputs['x'], self._mask, self._T)
            inputs.append([sample_grad, output, self._mask, self._T])
        return funcs, inputs

    def forward(self, x: torch.Tensor):
        return self._kernels['forward:y'].active_kernel()(x, self._mask, self._T)

    def backward(self, grad: torch.Tensor, output: torch.Tensor):
        return self._kernels['backward:x'].active_kernel()(grad, output, self._mask, self._T)


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
