# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from typing import Any, Dict, List, Callable, Optional

import torch

from sparta.tesa import get_bcs_function, BCSIndexes
from sparta.specializer.kernels import KernelBase
from sparta.testing import profile


class SparsityAttr(object):

    def __init__(self, BCSR: bool, BCSC: bool):
        self.BCSR = BCSR
        self.BCSC = BCSC
        self.mask: torch.Tensor = None
        self._block_H: int = 0
        self._block_W: int = 0
        self.indexes: BCSIndexes = None

    def update_axis(self, BCSR: bool, BCSC: bool):
        self.BCSR |= BCSR
        self.BCSC |= BCSC

    def set_block_size(self, block_H: int, block_W: int):
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


class Port(object):

    def __init__(self, func: SparseFunctionBase, name: str, fine_mask: bool = True):
        self.name = name
        self.funcs: List[SparseFunctionBase] = [func]
        self.attr: SparsityAttr = None
        self._sample_data: torch.Tensor = None  # Always dense
        self._fine_mask = fine_mask

    def set_data(self, data: torch.Tensor, grad: bool = False):
        if grad:
            self._sample_data.grad = data
        else:
            self._sample_data = data

    def get_data(self, grad: bool = False, compressed: bool = False):
        data: torch.Tensor = self._sample_data.grad if grad else self._sample_data
        if self.attr is not None and data is not None:
            if self._fine_mask:
                data = data * self.attr.mask
            if compressed:
                data = self.attr.indexes.convert(data.detach())
            elif not self._fine_mask:
                data = data * self.attr.indexes.get_mask()
        return data

    def clear_data(self):
        self._sample_data = None

    def connect(self, other: Port):
        for func in other.funcs:
            func.ports[other.name] = self
            self.funcs.append(func)
        if self.attr is not None and other.attr is not None:
            self.attr.update_axis(other.attr.BCSR, other.attr.BCSC)


class SparseFunctionBase(Callable):

    def __init__(self):
        self.kernels: Dict[str, Dict[str, KernelBase]] = {}
        self._compiled_kernels: Dict[str, KernelBase] = {}
        self.ports: Dict[str, Port] = {}
        self._sparse_port: str = ''
        self.forward: Callable = None
        self.backward: SparseFunctionBase = None

    def get_sparse_attr(self):
        return self.ports[self._sparse_port].attr

    def __call__(self, *inputs):
        return self.forward(*inputs)

    @abc.abstractmethod
    def _set_forward(self):
        """Build forward function with compiled kernels."""

    @abc.abstractmethod
    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]):
        """Get shape parameters from sample inputs."""

    @abc.abstractmethod
    def _compile_kernel(self, kernel_name: str, kernel: KernelBase, params: Dict[str, Any]):
        """Compile kernel with params, shapes and sparse indexes."""

    def build(
        self,
        config: Dict[str, Dict[str, Any]],
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ):
        if sample_inputs is not None:
            self._read_sample_inputs(sample_inputs)
        self._compiled_kernels: Dict[str, KernelBase] = {}
        for kernel_name, params in config.items():
            if kernel_name in self.kernels:
                kernel = self.kernels[kernel_name][params['_impl']]
                self._compile_kernel(kernel_name, kernel, params)
                self._compiled_kernels[kernel_name] = kernel
        self._set_forward()

    def clear_sample_data(self):
        for port in self.ports.values():
            port.clear_data()

    @abc.abstractmethod
    def _kernel_reference(self, kernel_name: str) -> torch.Tensor:
        """Get kernel reference output from related port(s)."""

    @abc.abstractmethod
    def _kernel_func_call(self, kernel_name: str) -> Callable[[], torch.Tensor]:
        """Callable kernel function based on sample data of ports."""

    def profile_kernel(
        self,
        kernel_name: str,
        num_warmups: int = 20,
        num_iters: int = 100,
        cuda: bool = False,
    ):
        """Profile kernel latency. Note that all inputs and outputs are dense tensors here."""
        kernel_func = self._kernel_func_call(kernel_name)
        target_output = self._kernel_reference(kernel_name)
        return profile(kernel_func, [], [target_output], num_warmups, num_iters, cuda)

    @abc.abstractmethod
    def _calc_kernel_flops(self, kernel_name: str):
        """Calculate kernel flops using sparse rate and shape."""

    def estimate_kernel(self, kernel_name: str):
        kernel = self._compiled_kernels[kernel_name]
        flops = self._calc_kernel_flops(kernel_name)
        return kernel.estimated_latency_per_flop * flops

    @abc.abstractmethod
    def reference_forward(self, sample_inputs: Optional[List[torch.Tensor]] = None):
        """Read input data from input port(s) and set output data to output port(s)."""


class SparseAutoGradFunction(SparseFunctionBase):

    __static_func__: torch.autograd.Function = None

    def __call__(self, *inputs):
        return self.__static_func__.apply(self, *inputs)

    def build(
        self,
        config: Dict[str, Dict[str, Any]],
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ):
        super().build(config, sample_inputs)
        self.backward.build(config, sample_inputs)

    def profile_kernel(
        self,
        kernel_name: str,
        num_warmups: int = 20,
        num_iters: int = 100,
        cuda: bool = False
    ):
        if kernel_name in self.kernels:
            return super().profile_kernel(kernel_name, num_warmups, num_iters, cuda)
        elif self.backward is not None:
            return self.backward.profile_kernel(kernel_name, num_warmups, num_iters, cuda)

    def estimate_kernel(self, kernel_name: str):
        if kernel_name in self.kernels:
            return super().estimate_kernel(kernel_name)
        elif self.backward is not None:
            return self.backward.estimate_kernel(kernel_name)

    @abc.abstractmethod
    def reference_backward(self, sample_grads: Optional[List[torch.Tensor]] = None):
        """Read grad data from output port(s) and backward by auto-grad."""
