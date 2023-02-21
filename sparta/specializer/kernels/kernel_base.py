# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import warnings
import dataclasses
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch

from sparta import __env_ready__
if __env_ready__:
    # we may need to dry run without GPU (e.g., for document generation)
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule

from sparta.tuning import TunableItemCfg


@dataclasses.dataclass
class _Parameter:
    name: str
    value: Optional[Any] = None
    is_tunable: bool = False
    is_dynamic: bool = False
    search_space: Optional[TunableItemCfg] = None

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tunable


class KernelBase(Callable):

    def __init__(self):
        self._parameters: Dict[str, _Parameter] = {}
        self._kernel: Callable = None
        self._func: Callable = None
        self.ready = False
        self._add_parameters()
        self.estimated_latency_per_flop = float('inf')

    @abc.abstractmethod
    def _add_parameters(self):
        """Add kernel-specialized parameters."""
    
    def _add_parameter(
        self,
        name: str,
        value: Any = None,
        is_tunable: bool = False,
        is_dynamic: bool = False,
        search_space: Optional[List[Any]] = None,
    ):
        self._parameters[name] = _Parameter(name, value, is_tunable, is_dynamic, search_space)

    def set_search_space(self, search_space: Dict[str, List[Any]]):
        for name, space in search_space.items():
            self._parameters[name].search_space = space

    def get_search_space(self, fixed_params: Optional[Dict[str, Any]] = None):
        search_space = {
            name: param.search_space
            for name, param in self._parameters.items()
            if param.is_tunable
        }
        if fixed_params is not None:
            for name, value in fixed_params.items():
                if name not in self._parameters.keys():
                    return None
                param = self._parameters[name]
                if param.is_tunable:
                    if param.search_space.includes(value):
                        search_space[name] = TunableItemCfg('choice', [value])
                    else:
                        return None
                else:
                    if param.value is not None and param.value != value:
                        return None
        return search_space

    def set_parameter(self, name: str, value: Any):
        if name in self._parameters:
            self._parameters[name].value = value

    def set_parameters(self, params: Dict[str, Any]):
        for name, value in params.items():
            self.set_parameter(name, value)

    def get_parameter(self, name: str):
        return self._parameters[name].value

    def get_parameters(self, names: Optional[List[str]] = None):
        if names is None:
            names = self._parameters.keys()
        return {name: self.get_parameter(name) for name in names}

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        """Get CUDA code of the kernel."""

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
        """Get launch config: number of threads per block."""

    @abc.abstractmethod
    def _check_parameters(self, params: Dict[str, Any]):
        """Raise an error if the input paramater dict is invalid."""

    @abc.abstractmethod
    def set_kernel_call(self, shape: Tuple, sparse_attr: Any):
        """Convert pycuda kernel (self._kernel) to python function call (self._func)."""

    def compile(self, params: Dict[str, Any], shape: Any, sparse_attr: Any):
        self._check_parameters(params)
        self.set_parameters(params)
        kernel_code = self.get_kernel_code()
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(kernel_code, options=['-O3'])
        self._kernel = source_module.get_function(kernel_name)
        self.set_kernel_call(shape, sparse_attr)
        self.ready = True

    def __call__(self, *args) -> torch.Tensor:
        if self.ready:
            return self._func(*args)
        else:
            raise ValueError('The kernel is not compiled.')
