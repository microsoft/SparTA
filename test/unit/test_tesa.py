# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Tuple

import torch
import pytest
from scipy.sparse import bsr_matrix

from sparta.common.tesa import BCSR, BCSC
from sparta.testing import block_mask


def test_bcsr(
    shape: Tuple[int, int] = (128, 256),
    batch_size: int = 4,
    data_block_size: Tuple[int, int] = (4, 8),
    data_sparsity: float = 0.99,
    tesa_block_size: Tuple[int, int] = (32, 16),
    random_seed: int = 2022,
    device: Any = 'cpu',
):
    torch.manual_seed(random_seed)
    mask = block_mask(shape, data_block_size, data_sparsity, device=device)
    data = torch.rand((batch_size, ) + shape, device=device) * mask
    bcsr = BCSR(mask, tesa_block_size, device=device)
    bcsc = BCSC(mask, tesa_block_size, device=device)

    # Test BCSRH convert function
    row_ptr = bcsr.get_attr('row_ptr').numpy()
    col_idx = bcsr.get_attr('col_idx').numpy()
    mask_val = bcsr.convert(mask).reshape((-1, ) + tesa_block_size).numpy()
    mask_bsr = bsr_matrix((mask_val, col_idx, row_ptr), shape=shape).toarray()
    torch.testing.assert_close(mask, torch.tensor(mask_bsr))

    # Test BCSRV convert function
    col_ptr = bcsc.get_attr('col_ptr').numpy()
    row_idx = bcsc.get_attr('row_idx').numpy()
    mask_val = bcsc.convert(mask).reshape((-1, ) + tesa_block_size).swapaxes(1, 2).numpy()
    mask_bsr = bsr_matrix((mask_val, row_idx, col_ptr), shape=shape[::-1]).toarray()
    torch.testing.assert_close(mask, torch.tensor(mask_bsr).swapaxes(0, 1))

    # Test inverse functions
    sparse_val_bcsr = bcsr.convert(data)
    torch.testing.assert_close(data, bcsr.inverse(sparse_val_bcsr))
    sparse_val_bcsc = bcsc.convert(data)
    torch.testing.assert_close(data, bcsc.inverse(sparse_val_bcsc))

    # Test reorder functions
    torch.testing.assert_close(sparse_val_bcsc, bcsr.reorder_BCSR_to_BCSC(sparse_val_bcsr))
    torch.testing.assert_close(sparse_val_bcsr, bcsc.reorder_BCSC_to_BCSR(sparse_val_bcsc))

    # Test sum function
    torch.testing.assert_close(data.sum(-1), bcsr.sum(sparse_val_bcsr, axis=-1))
    torch.testing.assert_close(data.sum(-2), bcsr.sum(sparse_val_bcsr, axis=-2))
    torch.testing.assert_close(data.sum(-1), bcsc.sum(sparse_val_bcsc, axis=-1))
    torch.testing.assert_close(data.sum(-2), bcsc.sum(sparse_val_bcsc, axis=-2))
