# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Dict

import torch

from sparta.specializer.kernels import KernelBase


class KernelPlaceholder(object):

    def __init__(
        self, name: str, cat: str, impls: Dict[str, type[KernelBase]],
        args: Dict[str, Any], mask_map: Dict[str, str]
    ):
        self.name = name
        self.category = cat
        self._possible_kernels = {key: impl(**args) for key, impl in impls.items()}
        self._active_impl: str = None
        self._mask_map = mask_map
        self.ready = False

    def set_shape(self, *args, **kwargs):
        for kernel in self._possible_kernels.values():
            kernel.set_shape(*args, **kwargs)

    def build(self, impl: str, config: Dict[str, Any], mask: Dict[str, torch.Tensor]):
        self._active_impl = impl
        mapped_mask = {y: mask[x] for x, y in self._mask_map.items()}
        self._possible_kernels[impl].compile(config, mapped_mask)
        self.ready = True

    def get_converter(self, name: str):
        return self.active_kernel().get_converter(self._mask_map[name])

    def active_kernel(self):
        return self._possible_kernels[self._active_impl]


class SparseCtxBase(object):

    def __init__(self):
        self._kernels: Dict[str, KernelPlaceholder] = {}

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        '''Set shape parameters.'''

    def build(
        self, impl: Dict[str, str], config: Dict[str, Dict[str, Any]],
        mask: Dict[str, torch.Tensor]
    ):
        for kernel_name, impl_name in impl.items():
            self._kernels[kernel_name].build(impl_name, config[kernel_name], mask)

    def get_converter(self, kernel_name: str, tensor_name: str):
        return self._kernels[kernel_name].get_converter(tensor_name)
