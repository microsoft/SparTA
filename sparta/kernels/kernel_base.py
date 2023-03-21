# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import warnings
import dataclasses
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch
import numpy as np

from sparta import __env_ready__
if __env_ready__:
    # we may need to dry run without GPU (e.g., for document generation)
    import pycuda.autoprimaryctx
    from pycuda.compiler import SourceModule

from sparta.tesa import get_bcs_function, BCSIndexes
from sparta.tuning import TunableItemCfg
from sparta.testing import profile


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


@dataclasses.dataclass
class _KernelAttrEdge:
    attr: SparsityAttr
    kernel: KernelBase
    block_H: str
    block_W: str

    def get_block_size(self):
        BH = self.kernel.get_parameter(self.block_H)
        BW = self.kernel.get_parameter(self.block_W)
        return BH, BW


class SparsityAttr(object):

    def __init__(
        self,
        kernel: KernelBase,
        block_H: str,
        block_W: str,
        BCSR: bool,
        BCSC: bool,
    ):
        self.edges = {kernel.id: _KernelAttrEdge(self, kernel, block_H, block_W)}
        self.groups: List[KernelGroup] = []
        self.BCSR = BCSR
        self.BCSC = BCSC
        self.mask: torch.Tensor = None
        self._block_H: int = 0
        self._block_W: int = 0
        self.indexes: BCSIndexes = None
        self.ready = False

    def update_block_size(self, kernel_id: int):
        block_H, block_W = self.edges[kernel_id].get_block_size()
        if block_H != self._block_H or block_W != self._block_W:
            self._block_H = block_H
            self._block_W = block_W
            self._update_indexes()

    def set_mask(self, mask: torch.Tensor):
        self.mask = mask
        self._update_indexes()

    def _update_indexes(self):
        if self._block_H > 0 and self._block_W > 0 and self.mask is not None:
            self.indexes = get_bcs_function(
                self._block_H, self._block_W,
                self.BCSR, self.BCSC,
            ).build_indexes(self.mask)
            self.ready = True

    def connect(self, other: SparsityAttr):
        self.BCSR |= other.BCSR
        self.BCSC |= other.BCSC
        for kernel_id, kernel_edge in other.edges.items():
            self.edges[kernel_id] = kernel_edge
            kernel_edge.kernel.attr = self
        for kernel_group in other.groups:
            self.groups.append(kernel_group)
            kernel_group.attr = self

    def get_search_space(self, backward: bool = False):
        pass


class KernelGroup(object):

    def __init__(
        self,
        kernel_name: str,
        kernels: Dict[str, KernelBase],
        input_getter: Callable[[], List[torch.Tensor]],
    ):
        self.for_backward = kernel_name.startswith('backward')
        self._kernels = kernels
        self._get_inputs = input_getter
        kernel_list = list(kernels.values())
        self.attr = kernel_list[0].attr
        self.attr.groups.append(self)
        for kernel in kernel_list[1:]:
            self.attr.connect(kernel.attr)
        self.active_kernel: Callable = kernel_list[0].reference
        self.ready = False

    def set_sample_shape(self, shape: Tuple):
        self._shape = shape

    def build(self, params: Dict[str, Any]):
        self.active_kernel = self._kernels[params['_impl']]
        self.active_kernel.compile(params, self._shape)
        self.ready = True

    def get_search_space(self):
        return {
            impl: kernel.get_search_space()
            for impl, kernel in self._kernels.items()
        }


_kernel_num = 0
def _next_kernel_id():
    global _kernel_num
    _kernel_num += 1
    return _kernel_num


class KernelBase(Callable):

    __lut_shape__: Tuple = ()

    def __init__(self):
        self.id = _next_kernel_id()
        self.attr: SparsityAttr = None
        self._parameters: Dict[str, _Parameter] = {}
        self._kernel: Callable = None
        self._func: Callable = self.reference
        self.ready = False
        self._add_parameters()
        self._lut_latency = float('inf')
        self.estimate_latency = float('inf')

    @abc.abstractmethod
    def _add_parameters(self):
        """Add kernel-specialized parameters and set sparsity attribute."""
    
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
    def set_kernel_call(self, shape: Tuple):
        """Convert pycuda kernel (self._kernel) to python function call (self._func)."""

    def compile(self, params: Dict[str, Any], shape: Tuple):
        self._check_parameters(params)
        self.set_parameters(params)
        self.attr.update_block_size(self.id)
        kernel_code = self.get_kernel_code()
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(kernel_code, options=['-O3'])
        self._kernel = source_module.get_function(kernel_name)
        self.set_kernel_call(shape)
        self.ready = True
        # Calc estimated latency
        indexes = self.attr.indexes
        sparse_rate = indexes.block_nnz / indexes.row_num / indexes.col_num
        shape_rate = np.prod(shape) / np.prod(self.__lut_shape__)
        self.estimate_latency = self._lut_latency * shape_rate * sparse_rate

    @abc.abstractmethod
    def reference(self, *inputs, sparse: bool = False):
        """Reference forward function."""

    def profile(
        self,
        inputs: List[torch.Tensor],
        num_warmups: int = 20,
        num_iters: int = 100,
        cuda: bool = False
    ):
        target_output = self.reference(*inputs, sparse=True)
        return profile(self, inputs, [target_output], num_warmups, num_iters, cuda)

    def __call__(self, *args) -> torch.Tensor:
        return self._func(*args)
