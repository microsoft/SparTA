# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMulFunc


class SparseBatchMatMul(OperatorBase):

    __sparse_func__ = SparseBatchMatMulFunc

    def __init__(
        self,
        A_mask: Optional[torch.Tensor] = None,
        B_mask: Optional[torch.Tensor] = None,
        C_mask: Optional[torch.Tensor] = None,
        transpose_A: bool = False,
        transpose_B: bool = False,
        compressed: bool = True,
    ):
        super().__init__()

        self._transpose_A = transpose_A
        self._transpose_B = transpose_B
        ctx_args = {
            'transpose_A': transpose_A,
            'transpose_B': transpose_B,
            'biased': False,
            'compressed': compressed,
        }
        batch_size, M, K, N = None, None, None, None

        if sum(map(lambda x: x is not None, [A_mask, B_mask, C_mask])) > 1:
            raise ValueError(f'linear operators with multiple sparse masks are not supported')

        if A_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx(mode='sdd', **ctx_args)
            self._set_masks({'A': A_mask})
            if transpose_A:
                K, M = A_mask.shape
            else:
                M, K = A_mask.shape
        elif B_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx(mode='dsd', **ctx_args)
            self._set_masks({'B': B_mask})
            if transpose_B:
                N, K = B_mask.shape
            else:
                K, N = B_mask.shape
        elif C_mask is not None:
            self._sparse_ctx = SparseBatchMatMulCtx(mode='dds', **ctx_args)
            self._set_masks({'C': C_mask})
            M, N = C_mask.shape
        else:
            raise ValueError(f'expected a sparse mask on A / B / C')

        self._shape = {'batch_size': batch_size, 'M': M, 'K': K, 'N': N}

    def _read_sample_inputs(self, A: torch.Tensor, B: torch.Tensor):
        # TODO: check shape conflicts
        if self._transpose_A:
            batch_size, K, M = A.shape
        else:
            batch_size, M, K = A.shape
        if self._transpose_B:
            batch_size, N, K = B.shape
        else:
            batch_size, K, N = B.shape
        self._shape = {'batch_size': batch_size, 'M': M, 'K': K, 'N': N}
