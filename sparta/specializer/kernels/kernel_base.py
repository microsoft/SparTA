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

from sparta.common.tesa import TeSAConverter, BCS
from sparta.common.tuning import TunableItemCfg
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

    def __post_init__(self):
        self.tesa_type: Optional[Type[TeSAConverter]] = None
        self.real_tesa_type: Optional[Type[TeSAConverter]] = None
        self._tesa_config: Optional[List[Any]] = None
        self._tesa_params: Optional[List[str]] = None
        self.mask: Optional[torch.Tensor] = None
        self.parent: Optional[PortConfig] = None
        self.children: List[PortConfig] = []
        self.converter: Optional[TeSAConverter] = None

    def set_tesa(
        self, tesa_type: Type[TeSAConverter], tesa_params: List[str],
        real_tesa_type: Type[TeSAConverter] = None
    ):
        self.tesa_type = tesa_type
        self._tesa_params = tesa_params
        self.real_tesa_type = tesa_type if real_tesa_type is None else real_tesa_type 

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
            if any([val is None for val in tesa_config]):
                return
            if self._tesa_config is not None:
                if all([a == b for a, b in zip(tesa_config, self._tesa_config)]):
                    return
            self._tesa_config = tesa_config
            self._update_converter()

    def _update_converter(self):
        if self._tesa_config is not None and self.mask is not None:
            if self.real_tesa_type is not None:
                if issubclass(self.real_tesa_type, BCS):
                    BH, BW = self._tesa_config
                    converter = self.real_tesa_type(
                        mask=self.mask,
                        block_size=(BH, BW)
                    )
                else:
                    raise ValueError(f'unsupported TeSA type {self.real_tesa_type}')
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

    def compile(self, params: Dict[str, Any]):
        self._check_parameters(params)
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
    def reference(self, *args) -> Any:
        '''Dense reference. Note that all inputs and outputs are dense tensors here.'''

    @abc.abstractmethod
    def _convert_data(self, inputs: List[torch.Tensor], outputs: List[torch.Tensor]):
        '''Convert sample inputs and target outputs to sparse tenors in place if necessary.'''

    def test(
        self, inputs: List[torch.Tensor],
        num_warmups: int = 10, num_iters: int = 10, cuda: bool = True
    ):
        '''Note that all inputs and outputs are dense tensors here.'''
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
