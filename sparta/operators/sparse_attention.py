# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional
import warnings

import torch
import numpy as np

from sparta.operators import SparseBatchMatMul, SparseBatchSoftmax


class SparseAttention(torch.nn.Module):
    r"""The sparse attention operator.

    .. math::
        \text{Attention}(Q, K, V) = \text{Softmax}(Q K) V

    Args:
        mask (torch.Tensor): The mask tensor of shape :math:`(N_{target}, N_{sourse})`,
            where :math:`N_{target}` is the target sequence length
            and :math:`N_{sourse}` is the sourse sequence length.

    Shape:
        - Input1: :math:`(B \times H, N_{target}, E)` where :math:`B` is the batch size,
            :math:`H` is the number of heads and :math:`E` is the embed dimension.
        - Input2: :math:`(B \times H, N_{sourse}, E)`.
        - Input3: :math:`(B \times H, N_{sourse}, E)`, same shape as the second input.
        - Output: :math:`(B \times H, N_{target}, E)`, same shape as the first input.

    Examples:

        .. code-block:: python
    
            B, H, Ns, Nt, E = 4, 4, 1024, 1024, 1024

            # Create a mask
            mask = sparta.testing.block_mask((Nt, Ns), sparsity=0.99)

            # Create a sparse attention operator using the mask
            sparse_attention = sparta.nn.SparseAttention(mask=mask)

            # Tune the sparse attention operator
            sparta.nn.tune(sparse_attention, sample_inputs=[
                torch.rand((B * H, Nt, E), device='cuda'),
                torch.rand((B * H, Ns, E), device='cuda'),
                torch.rand((B * H, Ns, E), device='cuda'),
            ])

    """

    def __init__(self, mask: Optional[torch.Tensor] = None):
        super().__init__()
        self._matmul_qk = SparseBatchMatMul('dds', False, True, False, True)
        self._softmax = SparseBatchSoftmax(True, temperature=None)
        self._matmul_out = SparseBatchMatMul('sdd', False, False, False, True)
        self._matmul_qk.ports['C'].connect(self._softmax.ports['x'])
        self._softmax.ports['y'].connect(self._matmul_out.ports['A'])
        if mask is not None:
            self.set_mask(mask)

    def set_mask(self, mask: torch.Tensor):
        self._softmax.set_mask(mask)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        QK = self._matmul_qk(Q, K)
        SM = self._softmax(QK)
        return self._matmul_out(SM, V)
