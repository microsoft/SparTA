# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import warnings
from typing import Any, List, Dict, Optional, Type

import torch

from sparta.specializer.funtional import SparseCtxBase


class OperatorBase(torch.nn.Module):

    __base_class__: Type[torch.nn.Module] = None
    __sparse_func__: Type[torch.autograd.Function] = None

    def __init__(self, raw_module: Optional[torch.nn.Module] = None):
        if self.__base_class__ is not None and type(raw_module) is not self.__base_class__:
            raise ValueError(f'expected a {self.__base_class__} module')
        super().__init__()
        self._raw_module = raw_module
        self._sparse_ctx: SparseCtxBase = None
        self._shape: Dict[str, int] = None
        self.ready: bool = False

    def _set_masks(self, masks: Dict[str, torch.Tensor]):
        self._sparse_ctx.set_masks(masks)

    @abc.abstractmethod
    def _read_sample_inputs(self, *args):
        '''Read missing shape value from sample inputs.'''

    def build(self, params: Dict[str, Any], sample_inputs: List[Any]):
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        self._sparse_ctx.build(params)
        self.forward = self._sparse_forward
        self.ready = True

    def _sparse_forward(self, *args):
        return self.__sparse_func__.apply(self._sparse_ctx, *args)

    def _dense_forward(self, *args):
        if self._raw_module is None:
            return self._sparse_ctx.dense_forward(*args)
        else:
            return self._raw_module.forward(*args)

    def forward(self, *args) -> torch.Tensor:
        '''Forward function. Calls the corresponding dense operator if not built.'''
        warnings.warn('the sparse module is not compiled, using the dense module to forward')
        return self._dense_forward(*args)

    def set_sample_inputs(
        self, sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None
    ):
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        self._sparse_ctx.set_sample_inputs(sample_inputs, sample_grads)

    def get_search_space(self, backward: bool = False):
        return self._sparse_ctx.get_search_space(backward)

    def get_connections(self, backward: bool = False):
        return self._sparse_ctx.get_connections(backward)

    def get_kernel_placeholders(self, backward: bool = False):
        return self._sparse_ctx.get_kernel_placeholders(backward)
