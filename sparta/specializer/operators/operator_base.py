# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import warnings
from typing import Any, List, Dict, Optional, Type

import torch

from sparta.specializer.funtional import SparseCtxBase


class OperatorBase(torch.nn.Module):
    """Base class of sparse operators.

    Each sparse operator contains a sparse function context and applies a PyTorch
    autograd function to do forward and backward calculation.

    SparTA does not handle parameter initialization. Instead, a PyTorch dense operator
    is required to provide parameter(s) for parametric operators.

    Args:
        raw_module (Optional[torch.nn.Module]): The corresponding dense operator.

    """

    __base_class__: Type[torch.nn.Module] = None
    __sparse_func__: Type[torch.autograd.Function] = None

    def __init__(self, raw_module: Optional[torch.nn.Module] = None):
        if self.__base_class__ is not None and type(raw_module) is not self.__base_class__:
            raise ValueError(f'expected a {self.__base_class__} module')
        super().__init__()
        self._raw_module = raw_module
        self._sparse_ctx: SparseCtxBase = None
        self._shape: Dict[str, int] = None
        self.ready: bool = False

    def _set_masks(self, masks: Dict[str, torch.Tensor]):
        """Set masks for the sparse context."""
        self._sparse_ctx.set_masks(masks)

    @abc.abstractmethod
    def _read_sample_inputs(self, *args):
        """Read missing shape value from sample inputs."""

    def build(self, config: Dict[str, Any], sample_inputs: List[Any]):
        """The build function includes following steps:
        1. Read and confirm the operator shape from sample inputs.
        2. Set shape for the sparse context.
        3. Build the sparse context with input config.
        4. Replace the forward function from dense to sparse version.
        5. Disconnect the raw module which contains dense parameter(s).
        6. Mark the sparse operator as ready.

        Args:
            config (Dict[str, Any]): A dictionary gives value of each required
                hyper parameter of the sparse context.
            sample_inputs (List[Any]): List of sample inputs.
        """
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        self._sparse_ctx.build(config)
        self.forward = self._sparse_forward
        self._raw_module = None
        self.ready = True

    def _sparse_forward(self, *args):
        """Apply the sparse autograd function."""
        return self.__sparse_func__.apply(self._sparse_ctx, *args)

    def _dense_forward(self, *args):
        """The dense forward function for reference."""
        if self._raw_module is None:
            return self._sparse_ctx.dense_forward(*args)
        else:
            return self._raw_module.forward(*args)

    def forward(self, *args) -> torch.Tensor:
        """Forward function. Calls the corresponding dense operator if not built."""
        warnings.warn('the sparse module is not compiled, using the dense module to forward')
        return self._dense_forward(*args)

    def set_sample_inputs(
        self,
        sample_inputs: List[torch.Tensor],
        sample_grads: Optional[List[torch.Tensor]] = None,
    ):
        """Set sample inputs and gradients for tuning."""
        self._read_sample_inputs(*sample_inputs)
        self._sparse_ctx.set_shape(**self._shape)
        self._sparse_ctx.set_sample_inputs(sample_inputs, sample_grads)

    def get_search_space(self, backward: bool = False):
        """Get search space of the sparse context."""
        return self._sparse_ctx.get_search_space(backward)

    def get_connections(self, backward: bool = False):
        """Get cross-kernel connected hyper parameters of the sparse context."""
        return self._sparse_ctx.get_connections(backward)

    def get_converter(self, kernel_name: str, tensor_name: str):
        """Get a sparse tensor's TeSA converter of a kernel."""
        return self._sparse_ctx.get_converter(kernel_name, tensor_name)

    def get_kernel_placeholders(self, backward: bool = False):
        """Get kernel placeholders.
        Returns only forward kernel placeholders(s) if backward is not required.
        """
        return self._sparse_ctx.get_kernel_placeholders(backward)
