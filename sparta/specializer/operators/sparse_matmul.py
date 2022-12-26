# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMulFunc


class SparseBatchMatMul(OperatorBase):
    r"""The sparse batch matrix multiplication operator: :math:`C = AB`

    Args:
        A_mask (Optional[torch.Tensor]): The mask of the first input tensor.
            If `A_mask` is set, the other two masks should be `None`
            and the internal MatMul kernel will choose SD=>D mode.
        B_mask (Optional[torch.Tensor]): The mask of the second input tensor.
            If `B_mask` is set, the other two masks should be `None`
            and the internal MatMul kernel will choose DS=>D mode.
        C_mask (Optional[torch.Tensor]): The mask of the output tensor.
            If `C_mask` is set, the other two masks should be `None` 
            and the internal MatMul kernel will choose DD=>S mode.
        transpose_A (bool): Determines whether the first input tensor is transposed.
        transpose_B (bool): Determines whether the second input tensor is transposed.
        compressed (bool): Determines whether the sparse tensor is compressed to
            BCSR / BCSC format.

    Shape:
        - Input1: :math:`(B, K, M)` (if `transpose_A == True`)
            or :math:`(B, M, K)` (if `transpose_A == False`).
            If `A_mask` is set and `compressed == True`, the first input will be
            compressed to BCSR / BCSC format and the shape will be :math:`(B, *)`.
        - Input2: :math:`(B, N, K)` (if `transpose_B == True`)
            or :math:`(B, K, N)` (if `transpose_B == False`).
            If `B_mask` is set and `compressed == True`, the second input will be
            compressed to BCSR / BCSC format and the shape will be :math:`(B, *)`.
        - Output: :math:`(B, M, N)`.
            If `C_mask` is set and `compressed == True`, the output will be
            compressed to BCSR format and the shape will be :math:`(B, *)`.

    Examples:

        .. code-block:: python
    
            B, M, K, N = 4, 1024, 1024, 1024

            # Create a output mask
            mask = sparta.testing.block_mask((M, N), sparsity=0.99)

            # Create a sparse batch matmul operator using the mask
            sparse_matmul = sparta.nn.SparseBatchMatMul(C_mask=mask)

            # Tune the sparse batch matmul operator
            sparta.nn.tune(sparse_matmul, sample_inputs=[
                torch.rand((B, M, K), device='cuda'),
                torch.rand((B, K, N), device='cuda'),
            ])

    """

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
