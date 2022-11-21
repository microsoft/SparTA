# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Dict, List, Type, Optional

import torch

from sparta.specializer.kernels import KernelBase


class KernelPlaceholder(object):

    def __init__(
        self, name: str, impls: Dict[str, Type[KernelBase]],
        args: Dict[str, Any], mask_map: Dict[str, str]
    ):
        self.name = name
        self._possible_kernels = {key: impl(**args) for key, impl in impls.items()}
        self._active_impl: str = None
        self._mask_map = mask_map
        self.sample_inputs: List[torch.Tensor] = []
        self.target_outputs: List[torch.Tensor] = []
        self.dense_func = list(self._possible_kernels.values())[0].reference
        self.ready = False

    def set_shape(self, *args, **kwargs):
        for kernel in self._possible_kernels.values():
            kernel.set_shape(*args, **kwargs)

    def set_masks(self, masks: Dict[str, torch.Tensor]):
        mapped_masks = {y: masks[x] for x, y in self._mask_map.items()}
        for kernel in self._possible_kernels.values():
            kernel.set_masks(mapped_masks)

    def build(self, impl: str, config: Dict[str, Any], ):
        self._active_impl = impl
        self._possible_kernels[impl].compile(config)
        self.ready = True

    def get_converter(self, name: str):
        if name in self._mask_map:
            return self.active_kernel().get_converter(self._mask_map[name])
        else:
            return None

    def active_kernel(self):
        if self._active_impl is None:
            return None
        else:
            return self._possible_kernels[self._active_impl]

    def get_search_space(self, fixed_params: Optional[Dict[str, Any]] = None):
        search_space = {}
        for impl, kernel in self._possible_kernels.items():
            kernel_search_space = kernel.get_search_space(fixed_params)
            if kernel_search_space is not None:
                search_space[impl] = kernel_search_space
        return search_space

    def set_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        self.sample_inputs = [x.detach() for x in sample_inputs]
        self.target_outputs = self.dense_func(sample_inputs)
        if type(self.target_outputs) is not tuple:
            self.target_outputs = (self.target_outputs)

    def test(self, num_warmups: int = 10, num_iters: int = 10, cuda: bool = True):
        return self.active_kernel().test(
            inputs=self.sample_inputs,
            target_outputs=self.target_outputs,
            num_warmups=num_warmups,
            num_iters=num_iters,
            cuda=cuda,
        )


class SparseCtxBase(object):

    def __init__(self):
        self._kernels: Dict[str, KernelPlaceholder] = {}

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        '''Set shape parameters.'''

    def set_masks(self, masks: Dict[str, torch.Tensor]):
        for kernel in self._kernels.values():
            kernel.set_masks(masks)

    def build(self, config: Dict[str, Dict[str, Any]]):
        for kernel_name, kernel_config in config.items():
            impl = kernel_config['_impl']
            self._kernels[kernel_name].build(impl, kernel_config)

    def get_converter(self, kernel_name: str, tensor_name: str):
        return self._kernels[kernel_name].get_converter(tensor_name)

    def get_kernel_placeholders(self, backward: bool = False):
        return {
            kernel_name: kernel
            for kernel_name, kernel in self._kernels.items()
            if backward or not kernel_name.startswith('backward')
        }

    @abc.abstractmethod
    def set_sample_inputs(
        self, sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None
    ):
        '''Set sample inputs and gradients for tuning.'''

    @abc.abstractmethod
    def get_connections(self, backward: bool = False) -> List[Dict[str, str]]:
        '''Get connected params among different kernels.'''

    def get_search_space(self, backward: bool = False):
        return {
            kernel_name: kernel.get_search_space()
            for kernel_name, kernel in self._kernels.items()
            if backward or not kernel_name.startswith('backward')
        }

    @abc.abstractmethod
    def dense_forward(self, *args) -> Any:
        '''Use dense method to forward (requires gradient).'''
