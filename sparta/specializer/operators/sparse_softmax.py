
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmax


class SparseSoftmax(OperatorBase):

    __sparse_func__ = SparseBatchSoftmax

    def __init__(self, mask: torch.Tensor, temperature: float = 1):
        super().__init__()
        H, W = mask.shape
        self._sparse_ctx = SparseBatchSoftmaxCtx(compressed=False, temperature=temperature)
        self._mask = {'x': mask}
        self._shape = {'batch_size': 1, 'H': H, 'W': W}

    def _read_sample_inputs(self, x: torch.Tensor):
        H, W = self._shape['H'], self._shape['W']
        assert (H, W) == x.shape, f'expect input shape ({H}, {W}), got {x.shape}'

    def _sparse_forward(self, x: torch.Tensor):
        return self.__sparse_func__.apply(self._sparse_ctx, x).squeeze(0)

    def _construct_inputs(self, raw_inputs: List[torch.Tensor]):
        return {'x': raw_inputs[0]}
