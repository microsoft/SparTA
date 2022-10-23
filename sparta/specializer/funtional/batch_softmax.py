# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, Dict, Optional

import torch

from sparta.specializer.kernels import SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel
from sparta.specializer.funtional import SparseCtxBase, KernelPlaceholder


class SparseBatchSoftmaxCtx(SparseCtxBase):

    def __init__(self, compressed: bool, temperature: Optional[float] = None):
        super().__init__()

        self._compressed = compressed
        self._T = None if temperature is None else 1 / temperature
        self._mask: torch.Tensor = None

        for kernel_name, kernel_class in zip([
            ['forward:output', 'backward:input'],
            [SparTASparseSoftmaxKernel, SparTASparseSoftmaxBackwardKernel]
        ]):
            self._kernels[kernel_name] = KernelPlaceholder(
                name=kernel_name,
                cat=kernel_name,
                impls={'sparta': kernel_class},
                args={'compressed': compressed},
                mask_map={p: 'input' for p in ['input', 'mask', 'output']},
            )

    def set_shape(self, batch_size: int, H: int, W: int):
        self._kernels['forward:output'].set_shape(batch_size, H, W)
        self._kernels['backward:input'].set_shape(batch_size, H, W)
        if self._T is None:
            self._T = 1 / torch.sqrt(W)

    def get_conditions(self, impls: Dict[str, str]):
        if self._compressed and len(impls) > 1:
            return [
                ['forward:output;BLOCK_SIZE_H_VALUE', 'backward:input;BLOCK_SIZE_H_VALUE'],
                ['forward:output;BLOCK_SIZE_W_VALUE', 'backward:input;BLOCK_SIZE_W_VALUE'],
            ]
        else:
            return []

    def build(self, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        self._mask = mask['input']
        super().build(config, mask)

    def _split_graph(
        self, kernels: List[str], sample_inputs: Dict[str, torch.Tensor],
        sample_grad: Optional[torch.Tensor] = None
    ):
        funcs, inputs = [], []
        for kernel_name in kernels:
            if kernel_name == 'forward':
                funcs.append(self.forward)
                inputs.append([sample_inputs['input'], self._mask, self._T])
            elif kernel_name == 'backward':
                funcs.append(self.backward)
                inputs.append([sample_grad, self._mask, self._T])
        return funcs, inputs

    def forward(self, x: torch.Tensor):
        return self._kernels['forward:output'].active_kernel()(x, self._mask, self._T)

    def backward(self, grad: torch.Tensor):
        return self._kernels['backward:input'].active_kernel()(grad, self._mask, self._T)


class SparseBatchSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sparta_ctx: SparseBatchSoftmaxCtx, x: torch.Tensor
    ):
        # ctx.save_for_backward(input, weight, bias)
        ctx.sparta_ctx = sparta_ctx
        return sparta_ctx.forward(x.detach())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor):
        # input, weight, bias = ctx.saved_tensors
        return ctx.sparta_ctx.backward(grad.detach())
