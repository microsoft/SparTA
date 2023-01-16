# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Dict, List, Type, Optional

import torch

from sparta.specializer.kernels import KernelBase, PortConfig


class KernelPlaceholder(object):

    def __init__(
        self,
        name: str,
        impls: Dict[str, Type[KernelBase]],
        args: Dict[str, Any],
        port_map: Dict[str, str],
        connectable: bool,
    ):
        self.name = name
        self._possible_kernels = {key: impl(**args) for key, impl in impls.items()}
        self.port_map = port_map
        self.connectable = connectable
        self._active_impl: str = None
        self.sample_inputs: List[torch.Tensor] = []
        self.dense_func = list(self._possible_kernels.values())[0].reference
        self.ready = False

    def set_shape(self, *args, **kwargs):
        for kernel in self._possible_kernels.values():
            kernel.set_shape(*args, **kwargs)

    def select_impl(self, impl: str):
        self._active_impl = impl

    def active_kernel(self):
        if self._active_impl is None:
            return None
        else:
            return self._possible_kernels[self._active_impl]

    def build(self, config: Dict[str, Any]):
        self.active_kernel().compile(config)
        self.ready = True

    def get_search_space(self, fixed_params: Optional[Dict[str, Any]] = None):
        search_space = {}
        for impl, kernel in self._possible_kernels.items():
            kernel_search_space = kernel.get_search_space(fixed_params)
            if kernel_search_space is not None:
                search_space[impl] = kernel_search_space
        return search_space

    def set_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        self.sample_inputs = [
            x.detach() if type(x) is torch.Tensor else x
            for x in sample_inputs
        ]

    def test(self, num_warmups: int = 10, num_iters: int = 10, cuda: bool = True):
        return self.active_kernel().test(
            inputs=self.sample_inputs,
            num_warmups=num_warmups,
            num_iters=num_iters,
            cuda=cuda,
        )


class SparseCtxBase(object):

    def __init__(self):
        self._kernels: Dict[str, KernelPlaceholder] = {}
        self.sparse_ports: Dict[str, List[PortConfig]] = {}

    @abc.abstractmethod
    def set_shape(self, *args, **kwargs):
        """Set shape parameters."""

    def select_impls(self, impls: Dict[str, str]):
        connected_ports: Dict[str, PortConfig] = {}
        for kernel_name, kernel_impl in impls.items():
            kernel_placeholder = self._kernels[kernel_name]
            kernel_placeholder.select_impl(kernel_impl)
            for global_port_name, kernel_port_name in kernel_placeholder.port_map.items():
                kernel = kernel_placeholder.active_kernel()
                if kernel_placeholder.connectable:
                    if global_port_name in connected_ports:
                        connected_ports[global_port_name].connect(kernel, kernel_port_name)
                    else:
                        connected_ports[global_port_name] = kernel.ports[kernel_port_name]
                else:
                    self.sparse_ports[global_port_name].append(kernel.ports[kernel_port_name])
        for global_port_name, port in connected_ports.items():
            self.sparse_ports[global_port_name].append(port)

    def build(self, config: Dict[str, Dict[str, Any]]):
        for kernel_name, kernel_config in config.items():
            self._kernels[kernel_name].build(kernel_config)

    def get_kernel_placeholders(self, backward: bool = False):
        return {
            kernel_name: kernel
            for kernel_name, kernel in self._kernels.items()
            if backward or kernel_name.startswith('forward')
        }

    @abc.abstractmethod
    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        """Set sample inputs and gradients for tuning."""

    @abc.abstractmethod
    def get_connections(self, backward: bool = False) -> List[Dict[str, str]]:
        """Get connected params among different kernels."""

    def get_search_space(self, backward: bool = False):
        return {
            kernel_name: kernel.get_search_space()
            for kernel_name, kernel in self._kernels.items()
            if backward or kernel_name.startswith('forward')
        }

    @abc.abstractmethod
    def dense_forward(self, *args) -> Any:
        """Use dense method to forward (requires gradient)."""
