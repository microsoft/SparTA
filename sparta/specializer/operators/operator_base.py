# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import warnings
from typing import Any, List, Dict, Optional, Callable, Type

import torch

from sparta.specializer.funtional import SparseCtxBase, SparseBatchMatMulCtx, SparseBatchMatMul

class OperatorBase(torch.nn.Module):

    __base_class__: Type[torch.nn.Module] = None
    __sparse_func__: Type[torch.autograd.Function] = None

    def __init__(self, raw_module: torch.nn.Module):
        if type(raw_module) is not self.__base_class__:
            raise ValueError(f'expected a {self.__base_class__} module')
        super().__init__()
        self._raw_module = raw_module
        self._sparse_ctx: SparseCtxBase = None
        self._shape: Dict[str, int] = None
        self._mask: Dict[str, torch.Tensor] = None
        self.ready: bool = False

    @abc.abstractmethod
    def _read_sample_inputs(self, *args):
        '''Read missing shape value from sample inputs.'''

    def build(self, params: Dict[str, Any], sample_inputs: List[Any]):
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        self._sparse_ctx.build()

    def forward(self, *args):
        '''Forward function. Calls the corresponding dense operator if not built.'''
        if self.ready:
            return self.__sparse_func__.apply(self._sparse_ctx, *args)
        else:
            warnings.warn('the sparse module is not compiled, using the dense module to forward')
            return self._raw_module.forward(*args)


class SparseLinear(OperatorBase):

    __base_class__ = torch.nn.Linear
    __sparse_func__ = SparseBatchMatMul

    def __init__(
        self, raw_module: torch.nn.Linear, input_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None, output_mask: Optional[torch.Tensor] = None
    ):
        super().__init__()

        M = None
        N, K = raw_module.weight.shape
        biased = raw_module.bias is not None

        if sum(map(lambda x: x is not None, [input_mask, weight_mask, output_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')

        if input_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('sdd', False, True, biased, False)
            assert input_mask.shape[1] == K, f'expected input mask shape (?, {K}), got {input_mask.shape}'
            self._mask = {'A': input_mask}
            M = input_mask.shape[1]
        elif weight_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dsd', False, True, biased, True)
            assert weight_mask.shape == (N, K), f'expected weight mask shape ({N}, {K}), got {weight_mask.shape}'
            self._mask = {'B': weight_mask}
        elif output_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx('dds', False, True, biased, False)
            assert output_mask.shape[1] == N, f'expected output mask shape (?, {N}), got {output_mask.shape}'
            self._mask = {'C': output_mask}
            M = output_mask.shape[0]
        else:
            raise ValueError(f'expected a sparse mask on input / weight / output')

        self._shape = {'batch_size': 1, 'M': M, 'K': K, 'N': N}

    def _read_sample_inputs(self, A: torch.Tensor):
        if self._shape['M'] is None:
            assert self._shape['K'] == A.shape[1], f'expect input shape (?, {K}), got {A.shape}'
            self._shape['M'] = A.shape[0]
        else:
            expected_shape = (self._shape['M'], self._shape['K'])
            assert expected_shape == A.shape, f'expect input shape {expected_shape}, got {A.shape}'
