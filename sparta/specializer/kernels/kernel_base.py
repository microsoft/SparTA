# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import warnings
import dataclasses
from typing import Any, Dict, List, Tuple, Callable, Optional, Type

import torch

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
class PortConfig(object):
    name: str
    is_input: bool
    is_sparse: bool = False
    BCSR: bool = False
    BCSC: bool = False

    def __post_init__(self):
        self.mask: torch.Tensor = None
        self._block_H: int = 0
        self._block_W: int = 0
        self.indexes: BCSIndexes = None

    def set_sparse(self, BCSR: bool, BCSC: bool):
        self.is_sparse = True
        self.BCSR = BCSR
        self.BCSC = BCSC

    def set_block_size(self, block_H: int, block_W: int):
        if self.is_sparse:
            if block_H != self._block_H or block_W != self._block_W:
                self._block_H = block_H
                self._block_W = block_W
                self._update_indexes()

    def set_mask(self, mask: torch.Tensor):
        if self.is_sparse:
            self.mask = mask
            self._update_indexes()

    def _update_indexes(self):
        if self._block_H > 0 and self._block_W > 0 and self.mask is not None:
            self.indexes = get_bcs_function(
                self._block_H, self._block_W,
                self.BCSR, self.BCSC,
            ).build_indexes(self.mask)

    def connect(self, kernel: KernelBase, port_name: str):
        other_port = kernel.ports[port_name]
        assert self.is_sparse == other_port.is_sparse
        self.BCSR |= other_port.BCSR
        self.BCSC |= other_port.BCSC
        kernel.ports[port_name] = self


class KernelBase(Callable):

    def __init__(self):
        self._parameters: Dict[str, _Parameter] = {}
        self._kernel: Callable = None
        self._func: Callable = None
        self.ports: Dict[str, PortConfig] = {}
        self.ready = False
        self._add_parameters()
        self._set_ports()

    @abc.abstractmethod
    def _set_ports(self):
        """Set input and output ports."""

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
        if name not in self._parameters and name in ['_name', '_impl']:
            return  # ignore some special key words
        self._parameters[name].value = value

    def set_parameters(self, params: Dict[str, Any]):
        for name, value in params.items():
            self.set_parameter(name, value)

    def get_parameter(self, name: str):
        return self._parameters[name].value

    def get_parameters(self, names: Optional[List[str]] = None):
        if names is None:
            return {k: v.value for k, v in self._parameters.items()}
        else:
            return {k: self._parameters[k].value for k in names}

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        """Set shape parameters."""

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        """Get CUDA code of the kernel."""

    @abc.abstractmethod
    def blocks_per_grid(self: int) -> Tuple[int]:
        """Get launch config: number of blocks per grid."""

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
        """Get launch config: number of threads per block."""

    @abc.abstractmethod
    def _check_parameters(self, params: Dict[str, Any]):
        """Raise an error if the input paramater dict is invalid."""

    @abc.abstractmethod
    def update_func(self):
        """Convert pycuda kernel (self._kernel) to python function call (self._func)."""

    def compile(self, params: Dict[str, Any]):
        self._check_parameters(params)
        self.set_parameters(params)
        kernel_code = self.get_kernel_code()
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(kernel_code, options=['-O3'])
        self._kernel = source_module.get_function(kernel_name)
        self.update_func()
        self.ready = True

    @abc.abstractmethod
    def reference(self, *args) -> Any:
        """Dense reference. Note that all inputs and outputs are dense tensors here."""

    @abc.abstractmethod
    def _convert_data(self, inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        """Convert sample inputs and target outputs to sparse tenors in place if necessary."""

    def test(
        self,
        inputs: List[torch.Tensor],
        num_warmups: int = 20,
        num_iters: int = 100,
        cuda: bool = False,
    ):
        """Note that all inputs and outputs are dense tensors here."""
        sparse_inputs = [x for x in inputs]
        sparse_outputs = self.reference(*sparse_inputs)
        if type(sparse_outputs) is not tuple:
            sparse_outputs = (sparse_outputs, )
        sparse_outputs = [y for y in sparse_outputs]
        self._convert_data(sparse_inputs, sparse_outputs)
        return profile(self, sparse_inputs, sparse_outputs, num_warmups, num_iters, cuda)

    def __call__(self, *args) -> torch.Tensor:
        if self.ready:
            return self._func(*args)
        else:
            raise ValueError('The kernel is not compiled.')
