
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmax


class SparseSoftmax(OperatorBase):

    __sparse_func__ = SparseBatchSoftmax

    def __init__(self, mask: torch.Tensor, temperature: float = 1, compressed: bool = False):
        super().__init__()
        H, W = mask.shape
        self._sparse_ctx = SparseBatchSoftmaxCtx(compressed, temperature)
        self._set_masks({'x': mask})
        self._shape = {'H': H, 'W': W}

    def set_temperature(self, temperature):
        self._sparse_ctx.set_temperature(temperature)

    def _read_sample_inputs(self, x: torch.Tensor):
        H, W = self._shape['H'], self._shape['W']
        if len(x.shape) == 2:
            assert (H, W) == x.shape, f'expect input shape ({H}, {W}), got {x.shape}'
            self._shape['batch_size'] = 1
            self._sparse_forward = self._sparse_forward_squeeze
            self._dense_forward = self._dense_forward_squeeze
        else:
            assert (H, W) == x.shape[-2:], f'expect input shape (?, {H}, {W}), got {x.shape}'
            self._shape['batch_size'] = math.prod(x.shape[:-2])

    def _sparse_forward_squeeze(self, x: torch.Tensor):
        return self.__sparse_func__.apply(self._sparse_ctx, x).squeeze(0)

    def _dense_forward_squeeze(self, *args):
        return self._sparse_ctx.dense_forward(*args).squeeze(0)
