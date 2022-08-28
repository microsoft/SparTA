# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Optional, List, Dict

import torch
import numpy as np

from sparta import specializer

class SparseModuleBase(torch.nn.Module):

    @abc.abstractclassmethod
    def _get_specializer(self, mask: np.ndarray) -> specializer.Specializer:
        '''
        Get the specializer corresponding to the module
        '''

    def find_best_config(
        self, mask: Optional[np.ndarray], search_space: Optional[Dict[str, List[int]]] = None
    ) -> Optional[Dict[str, int]]:
        op_specializer = load_module(module, mask=mask)
        return op_specializer.find_best_config(search_space=search_space)


class SparseLinear(SparseModuleBase):

    def __init__(
        self, raw_module: torch.nn.Linear, input_mask: Optional[np.ndarray] = None,
        weight_mask: Optional[np.ndarray] = None, output_mask: Optional[np.ndarray] = None,
        config: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        if type(raw_module) is not torch.nn.modules.linear.Linear:
            raise ValueError(f'expected a torch.nn.Linear module')
        N, K = raw_module.weight.shape
        M = None
        if sum(map(lambda x: x is None, [input_mask, weight_mask, output_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')
        if input_mask is not None:
            self._op_type = 'sdd'
            if input_mask.shape[0] == K:
                M = input_mask.shape[1]
                mask = {'A': input_mask.T}
            elif input_mask.shape[1] == K:
                M = input_mask.shape[0]
                mask = {'A': input_mask}
            else:
                raise ValueError(f'invalid input mask shape {input_mask.shape}')
        elif weight_mask is not None:
            self._op_type = 'dsd'
            if mask.shape == (N, K):
                mask = {'B': weight_mask}
            elif mask.shape == (K, N):
                mask = {'B': weight_mask.T}
            else:
                raise ValueError(f'invalid weight mask shape: {weight_mask.shape}')
        elif output_mask is not None:
            self._op_type = 'dds'
            if output_mask.shape[0] == N:
                M = output_mask.shape[1]
                mask = {'A': output_mask.T}
            elif output_mask.shape[1] == N:
                M = output_mask.shape[0]
                mask = {'A': output_mask}
            else:
                raise ValueError(f'invalid output mask shape {output_mask.shape}')
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')
        shape = {'GLOBAL_N_VALUE': N, 'GLOBAL_K_VALUE': K}
        if M is not None:
            shape |= {'GLOBAL_M_VALUE': M}
        if raw_module.bias is None:
            op_name = f'sparse_linear_{self._op_type}_t'
        else:
            op_name = f'sparse_linear_{self._op_type}_b_t'
        self._specializer = specializer.Specializer(op_name, shape=shape, mask=mask)
        self._raw_module = raw_module
        self.ready = False
        if config is not None:
            self.build(config)

    def build(self, config: Dict[str, int]):
        print('Building PyTorch Module, it will take about one minute...')
        module_interface = self._specializer.get_module_interface(config)
        self._forward_function = module_interface.get_module().forward
        if self._raw_module.bias is None:
            self.bias = None
        else:
            self.bias = torch.clone(self._raw_module.bias).to(torch.float32)
        weight = self._raw_module.weight.cpu().detach().numpy().astype(np.float32)
        if self._op_type == 'dsd':
            weight = module_interface.convert_dense_input('B', weight)
        self.weight = torch.from_numpy(weight).to(torch.float32)
        self.ready = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.ready:
            if self.bias is None:
                return self._forward_function(input, self.weight)
            else:
                return self._forward_function(input, self.weight, self.bias)
