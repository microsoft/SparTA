# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMul


class SparseLinear(OperatorBase):

    __base_class__ = torch.nn.Linear
    __sparse_func__ = SparseBatchMatMul

    def __init__(
        self, raw_module: torch.nn.Linear, input_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None, output_mask: Optional[torch.Tensor] = None
    ):
        super().__init__(raw_module)

        M = None
        N, K = raw_module.weight.shape
        biased = raw_module.bias is not None

        if sum(map(lambda x: x is not None, [input_mask, weight_mask, output_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')

        if input_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('sdd', False, True, biased, False)
            assert input_mask.shape[1] == K, f'expected input mask shape (?, {K}), got {input_mask.shape}'
            self._set_masks({'A': input_mask})
            M = input_mask.shape[0]
        elif weight_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dsd', False, True, biased, True)
            assert weight_mask.shape == (N, K), f'expected weight mask shape ({N}, {K}), got {weight_mask.shape}'
            self._set_masks({'B': weight_mask})
        elif output_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dds', False, True, biased, False)
            assert output_mask.shape[1] == N, f'expected output mask shape (?, {N}), got {output_mask.shape}'
            self._set_masks({'C': output_mask})
            M = output_mask.shape[0]
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')

        self._shape = {'batch_size': 1, 'M': M, 'K': K, 'N': N}
        self._raw_weight = torch.clone(raw_module.weight)
        self.weight = None
        self.bias = raw_module.bias

    def _read_sample_inputs(self, A: torch.Tensor):
        M, K = self._shape['M'], self._shape['K']
        if M is None:
            assert K == A.shape[1], f'expect input shape (?, {K}), got {A.shape}'
            self._shape['M'] = A.shape[0]
        else:
            assert (M, K) == A.shape, f'expect input shape ({M}, {K}), got {A.shape}'

    def build(self, params: Dict[str, Any], sample_inputs: List[Any]):
        super().build(params, sample_inputs)
        weight = self._raw_weight
        weight_converter = self._sparse_ctx.get_converter('forward:C', 'B')
        if weight_converter is not None:
            weight = weight_converter.convert(weight.detach())
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def _sparse_forward(self, input_tensor: torch.Tensor):
        inputs = [self._sparse_ctx, input_tensor, self.weight]
        if self.bias is not None:
            inputs.append(self.bias)
        return self.__sparse_func__.apply(*inputs).squeeze(0)
