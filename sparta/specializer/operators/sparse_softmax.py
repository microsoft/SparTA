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
        numpy_mask = mask.cpu().detach().numpy()
        self._mask = {'C_in': numpy_mask, 'C_mask': numpy_mask, 'C_out': numpy_mask}
        self._compressed = False
        self._dtype = 'float'
        self._shape = None
        self._possible_implementations = {
            'sparta': kernels.SparTATemplateSparseSoftmaxKernel(self._dtype, self._compressed),
        }

    def _load_compile_kernel(self, forward_kernel: kernels.KernelBase):
        '''No parameters need to be set here.'''
        mask = torch.from_numpy(self._mask['C_mask'].astype('int32'))
        self.C_mask = torch.nn.Parameter(mask, requires_grad=False).cuda()

    def _sparse_forward(self, C_in: torch.Tensor):
        '''Calls the sparse forward kernel.

        Args:
            C_in (torch.Tensor): The input tensor.
        '''
        return self._forward_function(
            C_in.unsqueeze(0),
            self.C_mask.unsqueeze(0)
        ).squeeze(0)

    def _read_sample_inputs(self, C_in: torch.Tensor):
        '''Read shape config and convert sample inputs to test inputs.

        Args:
            C_in (torch.Tensor): The sample input tensor.

        Returns:
            Tuple: The first value is the shape dict, the second value is the test input dict.
        '''
        H, W = C_in.shape
        self._shape = {
            'GLOBAL_H_VALUE': H,
            'GLOBAL_W_VALUE': W
        }
        for kern in self._possible_implementations.values():
            kern.set_parameters(self._shape)
        inputs = {
            'C_in': C_in.cpu().detach().unsqueeze(0).numpy().astype('float32')
        }
        return self._shape, inputs
