# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from typing import Any

import torch


class TeSAIndexes(object):

    def __init__(self, function_context: TeSAFunctionContext, mask: torch.Tensor):
        self.raw_mask = mask
        self.device = mask.device
        self.function_context = function_context

    @abc.abstractmethod
    def get_mask(self) -> torch.Tensor:
        """Get the mask actually used."""

    def to(self, device: Any):
        self.device = device
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, torch.Tensor):
                attr_val.to(device)


class TeSAFunctionContext(object):

    @abc.abstractmethod
    def build_indexes(self, mask: torch.Tensor, *args) -> TeSAIndexes:
        """Build TeSA index tensors."""

    @abc.abstractmethod
    def convert(self, dense: torch.Tensor, *args) -> torch.Tensor:
        """Convert dense tensor to compressed sparse value."""

    @abc.abstractmethod
    def inverse(self, sparse_val: torch.Tensor, *args) -> torch.Tensor:
        """Inversely convert compressed sparse value to dense tensor."""
