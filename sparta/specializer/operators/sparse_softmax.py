
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmax


class SparseSoftmax(OperatorBase):

    __sparse_func__ = SparseBatchSoftmax

    def __init__(
        self, mask: torch.Tensor, temperature: float = 1,
        compressed: bool = False, batch_size: Optional[int] = None
    ):
        super().__init__()
        H, W = mask.shape
        self._sparse_ctx = SparseBatchSoftmaxCtx(compressed, temperature)
        self._set_masks({'x': mask})
        if batch_size is None:
            self._sparse_forward = self._sparse_forward_squeeze
            self._dense_forward = self._dense_forward_squeeze
            batch_size = 1
        self._shape = {'batch_size': batch_size, 'H': H, 'W': W}

    def _read_sample_inputs(self, x: torch.Tensor):
        B, H, W = self._shape['batch_size'], self._shape['H'], self._shape['W']
        if len(x.shape) == 2:
            assert (H, W) == x.shape, f'expect input shape ({H}, {W}), got {x.shape}'
        else:
            assert (B, H, W) == x.shape, f'expect input shape ({B}, {H}, {W}), got {x.shape}'

    def _sparse_forward_squeeze(self, x: torch.Tensor):
        return self.__sparse_func__.apply(self._sparse_ctx, x).squeeze(0)

    def _dense_forward_squeeze(self, *args):
        return self._sparse_ctx.dense_forward(*args).squeeze(0)
