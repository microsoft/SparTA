# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Optional

import torch

from sparta.specializer import kernels
from sparta.specializer.operators.operator_base import OperatorBase


class SparseLinear(OperatorBase):
    """this is the docstring """

    def __init__(
        self, raw_module: torch.nn.Linear,
        input_mask: Optional[torch.Tensor] = None, weight_mask: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None
    ):
        super().__init__(raw_module, torch.nn.Linear)
        N, K = raw_module.weight.shape
        M = None
        if sum(map(lambda x: x is not None, [input_mask, weight_mask, output_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')
        if input_mask is not None:
            self._stype = 'sdd'
            self._compressed = False
            input_mask = input_mask.cpu().detach().numpy()
            if input_mask.shape[0] == K:
                M = input_mask.shape[1]
                self._mask = {'A': input_mask.T}
            elif input_mask.shape[1] == K:
                M = input_mask.shape[0]
                self._mask = {'A': input_mask}
            else:
                raise ValueError(f'invalid input mask shape {input_mask.shape}')
        elif weight_mask is not None:
            self._stype = 'dsd'
            self._compressed = True
            weight_mask = weight_mask.cpu().detach().numpy()
            if weight_mask.shape == (N, K):
                self._mask = {'B': weight_mask}
            elif weight_mask.shape == (K, N):
                self._mask = {'B': weight_mask.T}
            else:
                raise ValueError(f'invalid weight mask shape: {weight_mask.shape}')
        elif output_mask is not None:
            self._stype = 'dds'
            self._compressed = False
            output_mask = output_mask.cpu().detach().numpy()
            if output_mask.shape[0] == N:
                M = output_mask.shape[1]
                self._mask = {'A': output_mask.T}
            elif output_mask.shape[1] == N:
                M = output_mask.shape[0]
                self._mask = {'A': output_mask}
            else:
                raise ValueError(f'invalid output mask shape {output_mask.shape}')
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')
        self._shape = {'GLOBAL_N_VALUE': N, 'GLOBAL_K_VALUE': K}
        if M is not None:
            self._shape |= {'GLOBAL_M_VALUE': M}
        self._biased = raw_module.bias is not None
        self._transpose = True
        self._dtype = 'int' if 'int' in str(raw_module.weight.dtype) else 'float'

    def _create_forward_kernel(self, kernel_class: type[kernels.MatMulKernelBase]) -> kernels.KernelBase:
        return kernel_class(self._stype, self._dtype, self._biased, self._transpose, self._compressed)

    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        device = self._raw_module.weight.device
        if self._biased:
            bias = self._raw_module.bias.cpu().detach().numpy().astype(f'{self._dtype}32')
            self.bias = torch.nn.Parameter(torch.from_numpy(bias)).to(device)
        else:
            self.bias = None
        weight = self._raw_module.weight.cpu().detach().numpy().astype(f'{self._dtype}32')
        if self._stype == 'dsd':
            B_tensor = forward_kernel.get_input('B')
            B_tensor.set_data(weight)
            weight = B_tensor.sparse()['val']
        self.weight = torch.nn.Parameter(torch.from_numpy(weight)).to(device)

    def _possible_implementations(self):
        return {
            'sparta': kernels.OurTemplateSparseMatMulKernel,
            'openai': kernels.OpenAITemplateSparseMatMulKernel,
        }

    def _sparse_forward(self, A: torch.Tensor):
        if self._biased:
            return self._forward_function(A.to(self.weight.dtype), self.weight, self.bias)
        else:
            return self._forward_function(A.to(self.weight.dtype), self.weight)

    def _read_sample_inputs(self, A: torch.Tensor):
        M, K = A.shape
        shape = copy.deepcopy(self._shape)
        shape.update({
            'GLOBAL_M_VALUE': M,
            'GLOBAL_K_VALUE': K
        })
        inputs = {
            'A': A.cpu().detach().numpy().astype(f'{self._dtype}32'),
            'B': self._raw_module.weight.cpu().detach().numpy().astype(f'{self._dtype}32'),
        }
        if self._biased:
            inputs['bias'] = self._raw_module.bias.cpu().detach().numpy().astype(f'{self._dtype}32')
        return shape, inputs
