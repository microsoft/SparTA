
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

from sparta.specializer.operators import OperatorBase
from sparta.specializer.funtional import SparseBatchSoftmaxCtx, SparseBatchSoftmaxFunc


class SparseSoftmax(OperatorBase):
    r"""The sparse softmax operator.

    .. math::
        \text{Softmax}(x_{i}, T) = \frac{\exp(\frac{x_i}{T})}{\sum_j \exp(\frac{x_j}{T})}

    Args:
        mask (torch.Tensor): The mask tensor marking all positions to be calculated.
        temperature (float): The hyper parameter :math:`T` to control the smoothness of the results.
        compressed (bool): Determines whether input and output tensors are compressed to BCSR format.

    Shape:
        - Input: :math:`(*, H, W)` where `*` means any number of additional dimensions.
        - Output: :math:`(*, H, W)`, same shape as the input.

    Examples:

        .. code-block:: python
    
            B, H, W = 4, 1024, 1024

            # Create a mask
            mask = sparta.testing.block_mask((H, W), sparsity=0.99)

            # Create a sparse softmax operator using the mask
            sparse_softmax = sparta.nn.SparseSoftmax(mask=mask)

            # Tune the sparse softmax operator
            sparta.nn.tune(sparse_softmax, sample_inputs=[
                torch.rand((B, H, W), device='cuda'),
            ])

    """

    __sparse_func__ = SparseBatchSoftmaxFunc

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
            self._shape['batch_size'] = int(np.prod(x.shape[:-2]))

    def _sparse_forward_squeeze(self, x: torch.Tensor):
        return self.__sparse_func__.apply(self._sparse_ctx, x).squeeze(0)

    def _dense_forward_squeeze(self, *args):
        return self._sparse_ctx.dense_forward(*args).squeeze(0)
