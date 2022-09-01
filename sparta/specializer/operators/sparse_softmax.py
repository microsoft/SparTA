# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from sparta.specializer import kernels
from sparta.specializer.operators.operator_base import OperatorBase


class SparseSoftmax(OperatorBase):

    def __init__(self, raw_module: torch.nn.Softmax, mask: Optional[torch.Tensor] = None):
        super().__init__(raw_module, torch.nn.Softmax)
        self._raw_module = raw_module
        self._mask = {'C_in': mask.cpu().detach().numpy()}
        self._compressed = False
        self._dtype = 'float'

    def _create_forward_kernel(self, kernel_class: type[kernels.SoftmaxKernelBase]) -> kernels.KernelBase:
        return kernel_class(self._dtype, self._compressed)

    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        pass

    def _possible_implementations(self):
        return {
            'sparta': kernels.OurTemplateSparseSoftmaxKernel,
        }

    def _sparse_forward(self, C_in: torch.Tensor):
        return self._forward_function(C_in.to(torch.float32))

    def _read_sample_inputs(self, C_in: torch.Tensor):
        H, W = C_in.shape
        shape = {
            'GLOBAL_H_VALUE': H,
            'GLOBAL_W_VALUE': W
        }
        inputs = {
            'C_in': C_in.cpu().detach().numpy().astype('float32')
        }
        return shape, inputs
