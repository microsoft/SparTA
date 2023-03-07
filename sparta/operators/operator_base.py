# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from typing import Any, Dict, List, Tuple, Callable, Optional

import torch

from sparta.kernels import KernelBase, SparsityAttr, KernelGroup


class Port(object):

    def __init__(self, operator: SparseOperator, name: str):
        self.name = name
        self.ops: List[SparseOperator] = [operator]
        self.sample_data: torch.Tensor = None  # Dense, not masked
        self.attr: Optional[SparsityAttr] = None
        self.compressed: bool = False

    def get_sample_data(self, grad: bool = False):
        data = self.sample_data.grad if grad else self.sample_data
        data = data.detach()
        if self.attr is not None:
            data = data * self.attr.mask
            if self.compressed and self.attr.ready:
                data = self.attr.indexes.convert(data)
        return data

    def clear_data(self):
        self.sample_data = None

    def connect(self, other: Port):
        for operator in other.ops:
            operator.ports[other.name] = self
            self.ops.append(operator)
        if self.attr is None:
            self.attr = other.attr
        elif other.attr is not None:
            self.attr.connect(other.attr)
        # TODO: compressed


class SparseOperator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.kernel_groups: Dict[str, KernelGroup] = {}
        self.ports: Dict[str, Port] = {}
        self._attr: Optional[SparsityAttr] = None
        self.forward_func: Callable = None

    def _set_kernel_group(self, kernel_name: str, kernels: Dict[str, KernelBase]):
        input_getter = lambda : self._get_sample_inputs(kernel_name)
        self.kernel_groups[kernel_name] = KernelGroup(kernels, input_getter)

    @abc.abstractmethod
    def _get_sample_inputs(self, kernel_name: str) -> List[torch.Tensor]:
        """Get sample inputs from ports for specified kernel."""

    def get_sparse_indexes(self):
        assert self._attr is not None
        return self._attr.indexes

    def set_mask(self, mask: torch.Tensor):
        if self._attr is None:
            for kernel_group in self.kernel_groups.values():
                kernel_group.attr.set_mask(mask)
        else:
            self._attr.set_mask(mask)

    def forward(self, *inputs):
        return self.forward_func(*inputs)

    @abc.abstractmethod
    def _set_forward(self):
        """Build forward function with compiled kernels. Set sample data if reference."""

    @abc.abstractmethod
    def _read_sample_inputs(self, sample_inputs: List[torch.Tensor]) -> Tuple:
        """Get shape parameters from sample inputs."""

    @abc.abstractmethod
    def _set_sample_shape(self, sample_shape: Tuple):
        """Set sample shape to kernels."""

    def build(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        sample_inputs: Optional[List[torch.Tensor]] = None,
    ):
        if sample_inputs is not None:
            self._set_sample_shape(self._read_sample_inputs(sample_inputs))
        if config is not None:
            for kernel_name, params in config.items():
                self.kernel_groups[kernel_name].build(params)
        for kernel_name, kernel_group in self.kernel_groups.items():
            assert kernel_group.ready, f'{kernel_name} kernel is not built'
        self._post_build()

    def _post_build(self):
        self._set_forward()
        for port in self.ports.values():
            port.clear_data()


class SparseAutoGrad(SparseOperator):

    __static_func__: torch.autograd.Function = None

    def _set_backward(self, backward_op: SparseOperator):
        self.backward_op = backward_op
        backward_op.ports = self.ports
        for kernel_name, kernel_group in backward_op.kernel_groups.items():
            self.kernel_groups[kernel_name] = kernel_group
            if self._attr is not None:
                self._attr.connect(kernel_group.attr)
        backward_op._set_forward()

    def forward(self, *inputs):
        return self.__static_func__.apply(self, *inputs)

    def _post_build(self):
        self.backward_op._set_forward()
        super()._post_build()
