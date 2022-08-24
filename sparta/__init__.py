# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Optional, List, Dict

import torch
import numpy as np

from sparta import specializer


def load_module(
    module: torch.nn.Module,
    shape: Optional[Dict[str, int]] = None,
    mask: Optional[np.ndarray] = None
) -> specializer.Specializer:
    op_shape = {} if shape is None else copy.deepcopy(shape)
    if type(module) is torch.nn.modules.linear.Linear:
        N, K = module.weight.shape
        op_shape |= {'GLOBAL_K_VALUE': K, 'GLOBAL_N_VALUE': N}
        op_name = 'sparse_linear_dsd_t' if module.bias is None else 'sparse_linear_dsd_b_t'
        if mask is None:
            op_mask = None
        else:
            if mask.shape == (N, K):
                op_mask = {'B': mask}
            elif mask.shape == (K, N):
                op_mask = {'B': mask.T}
            else:
                raise ValueError(f'invalid mask shape: {mask.shape}')
        return specializer.Specializer(op_name, shape=op_shape, mask=op_mask)
    else:
        raise ValueError(f'unsupported module type: {type(module)}')

def find_best_config(
    module: torch.nn.Module, shape: Optional[Dict[str, int]] = None,
    mask: Optional[np.ndarray] = None, search_space: Optional[Dict[str, List[int]]] = None
) -> Optional[Dict[str, int]]:
    op_specializer = load_module(module, shape=shape, mask=mask)
    return op_specializer.find_best_config(search_space=search_space)

class SparseLinear(torch.nn.Module):

    def __init__(self, raw_module: torch.nn.Linear, mask: np.ndarray, config: Dict[str, int]):
        super().__init__()
        raw_module.forward
        if type(raw_module) is not torch.nn.modules.linear.Linear:
            raise ValueError(f'expected a torch.nn.Linear module as input')
        linear_specializer = load_module(raw_module, mask=mask)
        print('Building PyTorch Module, it will take about a minute...')
        module_interface = linear_specializer.get_module_interface(config)
        self._forward_function = module_interface.get_module().forward
        self.bias = copy.deepcopy(raw_module.bias).to(torch.float32)
        raw_weight = raw_module.weight.cpu().detach().numpy().astype(np.float32)
        weight = module_interface.convert_dense_input('B', raw_weight)
        self.weight = torch.from_numpy(weight).to(torch.float32).to(raw_module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return self._forward_function(input, self.weight)
        else:
            return self._forward_function(input, self.weight, self.bias)
