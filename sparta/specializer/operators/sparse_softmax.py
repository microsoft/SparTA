# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import numpy as np

from sparta.specializer import kernels
from sparta.specializer.operators.operator_base import OperatorBase


class SparseSoftmax(OperatorBase):

    def __init__(
        self, raw_module: torch.nn.Linear, dtype: str = 'float', implementation: str = 'our',
        mask: Optional[np.ndarray] = None
    ):
        super().__init__(implementation)
        if type(raw_module) is not torch.nn.modules.activation.Softmax:
            raise ValueError(f'expected a torch.nn.Softmax module')
        self._raw_module = raw_module
        self._mask = {'C_in': mask}
        self._compressed = False
        self._dtype = dtype

    def _create_forward_kernel(self, kernel_class: type[kernels.SoftmaxKernelBase]) -> kernels.KernelBase:
        return kernel_class(self._dtype, self._compressed)

    def _set_parameters(self, forward_kernel: kernels.KernelBase):
        pass

    def _possible_implementations(self):
        return {
            'our': kernels.OurTemplateSparseSoftmaxKernel,
        }

    def _sparse_forward(self, C_in: torch.Tensor):
        return self._forward_function(C_in.to(torch.int32 if self._dtype == 'int' else torch.float32))

    def _read_sample_inputs(self, C_in: torch.Tensor):
        H, W = C_in.shape
        shape = {
            'GLOBAL_H_VALUE': H,
            'GLOBAL_W_VALUE': W
        }
        inputs = {
            'C_in': C_in.cpu().detach().numpy().astype(f'{self._dtype}32')
        }
        return shape, inputs
