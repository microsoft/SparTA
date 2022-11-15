# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

from sparta.common.tesa import TeSAConverter, BCSR
from sparta.common.tuning import TunableItemCfg
from sparta.testing import profile


@dataclasses.dataclass
class _Parameter:
    name: str
    value: Any
    is_tunable: Optional[bool] = False
    is_dynamic: Optional[bool] = False
    search_space: Optional[TunableItemCfg] = None

    def __post_init__(self):
        if self.search_space is not None:
            assert self.is_tunable


@dataclasses.dataclass
class PortConfig(object):
    name: str
    is_input: bool
    tesa_type: Optional[Type[TeSAConverter]] = None

    def __post_init__(self):
        self._tesa_config: Optional[List[Any]] = None
        self._tesa_params: Optional[List[str]] = None
        self.mask: Optional[torch.Tensor] = None
        self.parent: Optional[PortConfig] = None
        self.children: List[PortConfig] = []
        self.converter: Optional[TeSAConverter] = None

    def set_tesa(self, tesa_type: Type[TeSAConverter], tesa_params: List[str]):
        self.tesa_type = tesa_type
        self._tesa_params = tesa_params

    def set_mask(self, mask: torch.Tensor):
        if self.parent is not None:
            self.parent.set_mask()
        else:
            self.mask = mask
            for child in self.children:
                child.mask = mask
            self._update_converter()

    def set_params(self, params: Dict[str, Any]):
        if self._tesa_params is not None:
            tesa_config = [params[p] for p in self._tesa_params]
            if self._tesa_config is not None:
                if all([a != b for a, b in zip(tesa_config, self._tesa_config)]):
                    return
            self._tesa_config = tesa_config
            self._update_converter()

    def _update_converter(self):
        if self._tesa_config is not None and self.mask is not None:
            if self.tesa_type is not None:
                if issubclass(self.tesa_type, BCSR):
                    H, W, BH, BW = self._tesa_config
                    converter = self.tesa_type(
                        mask=self.mask,
                        size=(H, W),
                        block_size=(BH, BW)
                    )
                else:
                    raise ValueError(f'unsupported TeSA type {self.tesa_type}')
                self.set_converter(converter)

    def set_converter(self, converter: TeSAConverter):
        if self.parent is not None:
            self.parent.set_converter(converter)
        else:
            self.converter = converter
            for child in self.children:
                child.converter = converter

    def connect(self, port: 'PortConfig'):
        if self.parent is not None:
            self.parent.connect(port)
        elif port.parent is not None:
            self.connect(port.parent)
        else:
            self.children.append(port)
            if len(port.children) > 0:
                self.children = list(self.children, *port.children)
                port.children = []


class KernelBase(Callable):

    def __init__(self):
        self._parameters: Dict[str, _Parameter] = {}
        self._func: Callable = None
        self.ports: Dict[str, PortConfig] = {}
        self.ready = False
        self._add_parameters()
        self._set_ports()

    @abc.abstractmethod
    def _set_ports(self):
        '''Set input and output ports.'''

    @abc.abstractmethod
    def _add_parameters(self):
        '''Add kernel-specialized parameters.'''
    
    def _add_parameter(
        self, name: str, value: Any = None, is_tunable: bool = False, is_dynamic: bool = False,
        search_space: Optional[List[Any]] = None
    ):
        self._parameters[name] = _Parameter(name, value, is_tunable, is_dynamic, search_space)

    def set_search_space(self, search_space: Dict[str, List[Any]]):
        for name, space in search_space.items():
            self._parameters[name].search_space = space

    def get_search_space(self):
        return {p.name: p.search_space for p in self._parameters.values() if p.is_tunable}

    def set_parameter(self, name: str, value: Any):
        if name not in self._parameters and name in ['_name']:
            return  # ignore some special key words
        self._parameters[name].value = value

    def set_parameters(self, params: Dict[str, Any]):
        for name, value in params.items():
            self.set_parameter(name, value)

    def set_port_params(self):
        for port in self.ports.values():
            port.set_params(self.get_parameters())

    def get_parameter(self, name: str):
        return self._parameters[name].value

    def get_parameters(self, names: Optional[List[str]] = None):
        if names is None:
            return {k: v.value for k, v in self._parameters.items()}
        else:
            return {k: self._parameters[k].value for k in names}

    def set_mask(self, name: str, value: torch.Tensor):
        self.ports[name].set_mask(value)

    def set_masks(self, mask_dict: Dict[str, torch.Tensor]):
        for name, value in mask_dict.items():
            self.set_mask(name, value)

    def get_mask(self, name: str):
        return self.ports[name].mask

    def get_converter(self, name: str) -> TeSAConverter:
        return self.ports[name].converter

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        '''Set shape parameters.'''

    @abc.abstractmethod
    def get_kernel_code(self) -> str:
        '''Get CUDA code of the kernel.'''

    @abc.abstractmethod
    def blocks_per_grid(self: int) -> Tuple[int]:
        '''Get launch config: number of blocks per grid.'''

    @abc.abstractmethod
    def threads_per_block(self) -> Tuple[int]:
        '''Get launch config: number of threads per block.'''

    @abc.abstractmethod
    def _check_parameters(self, params: Dict[str, Any]):
        '''Raise an error if the input paramater dict is invalid.'''

    @abc.abstractmethod
    def _set_func_call(self, kernel_func_call: Callable) -> Callable:
        '''Convert python function call to pycuda kernel function call.'''

    def compile(self, params: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        self._check_parameters(params)
        self.set_masks(mask)
        self.set_parameters(params)
        self.set_port_params()
        kernel_code = self.get_kernel_code()
        kernel_name = kernel_code[kernel_code.find('__global__ void') + 15:]
        kernel_name = kernel_name[:kernel_name.find('(')].strip()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_module = SourceModule(kernel_code, options=['-O3'])
        kernel_func_call = source_module.get_function(kernel_name)
        self._func = self._set_func_call(kernel_func_call)
        self.ready = True

    @abc.abstractmethod
    def reference(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        '''Calc target output by dense method.'''

    def test(self, inputs: List[torch.Tensor]):
        return profile(self, inputs, self.reference(inputs), cuda=True)

    def __call__(self, *args) -> torch.Tensor:
        if self.ready:
            return self._func(*args)
        else:
            raise ValueError('The kernel is not compiled.')
