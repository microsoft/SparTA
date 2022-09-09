# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Optional, Type

import torch

from sparta.specializer import kernels
from sparta.specializer.operators.operator_base import OperatorBase


class SparseLinear(OperatorBase):
    '''Sparse linear operator.

    Examples:

        .. code-block:: python

            # Create a dense linear layer
            dense_linear = torch.nn.Linear(1024, 2048)

            # Create a mask
            weight_mask = torch.rand((2048, 1024)) > 0.99

            # Create a sparse linear layer using the dense layer and the mask
            sparse_linear = sparta.nn.SparseLinear(dense_linear, weight_mask=weight_mask)

            # Tune the sparse linear layer
            sparta.tune(sparse_linear, sample_inputs=[torch.rand((512, 1024))])

    Args:
        raw_module (torch.nn.Linear): The corresponding dense linear operator.
        input_mask (torch.Tensor): The input mask tensor with shape (\*, in_features).
            The kernel mode will be "sparse x dense => dense" if the input mask is set.
        weight_mask (torch.Tensor): The weight mask tensor with shape (out_features, in_features).
            The kernel mode will be "dense x sparse => dense" if the input mask is set.
        output_mask (torch.Tensor): The output mask tensor with shape (\*, out_features).
            The kernel mode will be "dense x dense => sparse" if the input mask is set.
    '''

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
            if input_mask.shape[1] == K:
                M = input_mask.shape[0]
                self._mask = {'A': input_mask}
            else:
                raise ValueError(f'expected input mask shape (?, {K}), got {input_mask.shape}')
        elif weight_mask is not None:
            self._stype = 'dsd'
            self._compressed = True
            weight_mask = weight_mask.cpu().detach().numpy()
            if weight_mask.shape == (N, K):
                self._mask = {'B': weight_mask}
            else:
                raise ValueError(f'expected weight mask shape ({N}, {K}), got {weight_mask.shape}')
        elif output_mask is not None:
            self._stype = 'dds'
            self._compressed = False
            output_mask = output_mask.cpu().detach().numpy()
            if output_mask.shape[1] == N:
                M = output_mask.shape[0]
                self._mask = {'C': output_mask}
            else:
                raise ValueError(f'expected output mask shape (?, {N}), got {output_mask.shape}')
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')
        self._shape = {'GLOBAL_M_VALUE': M, 'GLOBAL_N_VALUE': N, 'GLOBAL_K_VALUE': K}
        self._biased = raw_module.bias is not None
        self._transpose = True
        self._dtype = 'int' if 'int' in str(raw_module.weight.dtype) else 'float'
        if self._stype == 'dds':
            self._possible_implementations = {
                'openai': kernels.OpenAITemplateSparseMatMulKernel(self._stype, self._dtype, self._biased, self._transpose, self._compressed),
            }
        else:
            self._possible_implementations = {
                'sparta': kernels.SparTATemplateSparseMatMulKernel(self._stype, self._dtype, self._biased, self._transpose, self._compressed),
                'openai': kernels.OpenAITemplateSparseMatMulKernel(self._stype, self._dtype, self._biased, self._transpose, self._compressed),
            }

    def _load_compile_kernel(self, forward_kernel: kernels.MatMulKernelBase):
        '''Set PyTorch module parameters: weight and bias (if exists).

        Args:
            forward_kernel (kernels.MatMulKernelBase): A matmul kernel object which provides the
                function to sparsify the weight tensor in "dense x sparse => dense" mode.
        '''
        device = self._raw_module.weight.device
        if self._biased:
            bias = self._raw_module.bias.cpu().detach().numpy().astype(f'{self._dtype}32')
            self.bias = torch.nn.Parameter(torch.from_numpy(bias), requires_grad=False).to(device)
        else:
            self.bias = None
        weight = self._raw_module.weight.cpu().detach().numpy().astype(f'{self._dtype}32')
        if self._stype == 'dsd':
            B_tensor = forward_kernel.get_input('B')
            B_tensor.set_data(weight)
            weight = B_tensor.sparse()['val']
        self.weight = torch.nn.Parameter(torch.from_numpy(weight), requires_grad=False).to(device)

    def _sparse_forward(self, A: torch.Tensor):
        '''Calls the sparse forward kernel.

        Args:
            A (torch.Tensor): The input tensor.
        '''
        if self._biased:
            return self._forward_function(A, self.weight, self.bias)
        else:
            return self._forward_function(A, self.weight)

    def _read_sample_inputs(self, A: torch.Tensor):
        '''Read shape config and convert sample inputs to test inputs.
        The captured shape config will be passed to implements (kernels).

        Args:
            A (torch.Tensor): The sample input tensor.

        Returns:
            Tuple: The first value is the shape dict, the second value is the test input dict.
        '''
        M, K = A.shape
        assert self._shape['GLOBAL_K_VALUE'] == K
        self._shape['GLOBAL_M_VALUE'] = M
        for kern in self._possible_implementations.values():
            kern.set_parameters(self._shape)
        inputs = {
            'A': A.cpu().detach().numpy().astype(f'{self._dtype}32'),
            'B': self._raw_module.weight.cpu().detach().numpy().astype(f'{self._dtype}32'),
        }
        if self._biased:
            inputs['bias'] = self._raw_module.bias.cpu().detach().numpy().astype(f'{self._dtype}32')
        return self._shape, inputs


if __name__ == '__main__':
    pass