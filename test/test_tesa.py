# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import torch
import pytest
from scipy.sparse import bsr_matrix

from sparta.common.tesa import BCSRH, BCSRV
from sparta.testing import block_mask


def test_bcsr(
    shape: Tuple[int, int] = (128, 256),
    batch_size: int = 4,
    data_block_size: Tuple[int, int] = (4, 8),
    data_sparsity: float = 0.99,
    tesa_block_size: Tuple[int, int] = (32, 16),
    random_seed: int = 2022,
):
    torch.manual_seed(random_seed)
    mask = block_mask(shape, data_block_size, data_sparsity, device='cpu')
    data = torch.rand((batch_size, ) + shape, device='cpu') * mask
    bcsrh = BCSRH(mask, tesa_block_size, device='cpu')
    bcsrv = BCSRV(mask, tesa_block_size, device='cpu')

    # Test BCSRH convert function
    row_ptr = bcsrh.get_attr('row_ptr').numpy()
    col_idx = bcsrh.get_attr('col_idx').numpy()
    mask_val = bcsrh.convert(mask).reshape((-1, ) + tesa_block_size).numpy()
    mask_bsr = bsr_matrix((mask_val, col_idx, row_ptr), shape=shape).toarray()
    torch.testing.assert_close(mask, torch.tensor(mask_bsr))

    # Test BCSRV convert function
    col_ptr = bcsrv.get_attr('col_ptr').numpy()
    row_idx = bcsrv.get_attr('row_idx').numpy()
    mask_val = bcsrv.convert(mask).reshape((-1, ) + tesa_block_size).swapaxes(1, 2).numpy()
    mask_bsr = bsr_matrix((mask_val, row_idx, col_ptr), shape=shape[::-1]).toarray()
    torch.testing.assert_close(mask, torch.tensor(mask_bsr).swapaxes(0, 1))

    # Test inverse functions
    sparse_val_h = bcsrh.convert(data)
    torch.testing.assert_close(data, bcsrh.inverse(sparse_val_h))
    sparse_val_v = bcsrv.convert(data)
    torch.testing.assert_close(data, bcsrv.inverse(sparse_val_v))

    # Test reorder functions
    torch.testing.assert_close(sparse_val_v, bcsrh.reorder_H_to_V(sparse_val_h))
    torch.testing.assert_close(sparse_val_h, bcsrv.reorder_V_to_H(sparse_val_v))

    # Test sum function
    torch.testing.assert_close(data.sum(-1), bcsrh.sum(sparse_val_h, axis=-1))
    torch.testing.assert_close(data.sum(-2), bcsrh.sum(sparse_val_h, axis=-2))
    torch.testing.assert_close(data.sum(-1), bcsrv.sum(sparse_val_v, axis=-1))
    torch.testing.assert_close(data.sum(-2), bcsrv.sum(sparse_val_v, axis=-2))
