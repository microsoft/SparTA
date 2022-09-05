# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from sparta.specializer import kernels
from sparta.specializer.operators.operator_base import OperatorBase


class SparseSoftmax(OperatorBase):
    '''Sparse softmax operator.

    Examples:

        .. code-block:: python

            # Create a dense softmax layer
            dense_softmax = torch.nn.Softmax

            # Create a mask
            mask = torch.rand((2048, 1024)) > 0.99

            # Create a sparse softmax layer using the dense layer and the mask
            sparse_softmax = sparta.nn.SparseSoftmax(dense_softmax, mask=mask)

            # Tune the sparse softmax layer
            sparta.tune(sparse_softmax, sample_inputs=[torch.rand((2048, 1024))])

    Args:
        raw_module (torch.nn.Softmax): The corresponding dense softmax operator.
        mask (torch.Tensor): The mask with the same shape as the input tensor.
    '''

    def __init__(self, raw_module: torch.nn.Softmax, mask: Optional[torch.Tensor] = None):
        super().__init__(raw_module, torch.nn.Softmax)
        self._raw_module = raw_module
        self._mask = {'C_in': mask.cpu().detach().numpy()}
        self._compressed = False
        self._dtype = 'float'

    def _create_forward_kernel(self, kernel_class: type[kernels.SoftmaxKernelBase]) -> kernels.KernelBase:
        '''Instantiate a forward kernel object using the specified softmax kernel class.

        Args:
            kernel_class (type[kernels.SoftmaxKernelBase]): A softmax kernel class which belongs to
                possible implementations.
        '''
        return kernel_class(self._dtype, self._compressed)

    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        '''No parameters need to be set here.'''
        pass

    def _possible_implementations(self):
        '''Get possible implementations.

        Returns:
            dict: Only SparTA's softmax kernel is supported.
        '''
        return {
            'sparta': kernels.SparTATemplateSparseSoftmaxKernel,
        }

    def _sparse_forward(self, C_in: torch.Tensor):
        '''Calls the sparse forward kernel.

        Args:
            C_in (torch.Tensor): The input tensor.
        '''
        return self._forward_function(C_in.to(torch.float32))

    def _read_sample_inputs(self, C_in: torch.Tensor):
        '''Read shape config and convert sample inputs to test inputs.

        Args:
            C_in (torch.Tensor): The sample input tensor.

        Returns:
            tuple: The first value is the shape dict, the second value is the test input dict.
        '''
        H, W = C_in.shape
        shape = {
            'GLOBAL_H_VALUE': H,
            'GLOBAL_W_VALUE': W
        }
        inputs = {
            'C_in': C_in.cpu().detach().numpy().astype('float32')
        }
        return shape, inputs
