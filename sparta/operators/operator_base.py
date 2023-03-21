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
        self.get_attr: Callable[[], SparsityAttr] = lambda : None
        self.compressed: bool = False

    def get_sample_data(self, grad: bool = False):
        data = self.sample_data.grad if grad else self.sample_data
        data = data.detach()
        attr = self.get_attr()
        if attr is not None:
            data = data * attr.mask
            if self.compressed and attr.ready:
                data = attr.indexes.convert(data)
        return data

    def clear_data(self):
        self.sample_data = None

    def connect(self, other: Port):
        for operator in other.ops:
            operator.ports[other.name] = self
            self.ops.append(operator)
        self_attr = self.get_attr()
        other_attr = other.get_attr()
        if self_attr is None:
            self.get_attr = other.get_attr
        elif other_attr is None:
            other.get_attr = self.get_attr
        else:
            self_attr.connect(other_attr)


class SparseOperator(torch.nn.Module):

    def __init__(self, compressed: bool):
        super().__init__()
        self._compressed = compressed
        self.kernel_groups: Dict[str, KernelGroup] = {}
        self.ports: Dict[str, Port] = {}
        self._sparse_port: Port = None
        self.forward_func: Callable = None

    def _set_kernel_group(self, kernel_name: str, kernels: Dict[str, KernelBase]):
        input_getter = lambda : self._get_sample_inputs(kernel_name)
        self.kernel_groups[kernel_name] = KernelGroup(kernel_name, kernels, input_getter)

    @abc.abstractmethod
    def _get_sample_inputs(self, kernel_name: str) -> List[torch.Tensor]:
        """Get sample inputs from ports for specified kernel."""

    def get_sparse_indexes(self):
        assert self._compressed, 'only compressed sparse operators can export sparse indexes'
        return self._sparse_port.get_attr().indexes

    def set_mask(self, mask: torch.Tensor):
        if self._compressed:
            self._sparse_port.get_attr().set_mask(mask)
        else:
            for kernel_group in self.kernel_groups.values():
                kernel_group.attr.set_mask(mask)

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
            if self._compressed:
                self_attr = self._sparse_port.get_attr()
                self_attr.connect(kernel_group.attr)
        backward_op._set_forward()

    def forward(self, *inputs):
        return self.__static_func__.apply(self, *inputs)

    def _post_build(self):
        self.backward_op._set_forward()
        super()._post_build()
