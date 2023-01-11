# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.tesa import get_bcs_function
from sparta.testing import block_mask


def reduce_mask_ref(mask: torch.Tensor, BH: int, BW: int):
    H, W = mask.shape
    row_num = H // BH
    col_num = W // BW
    reduced = mask.reshape((row_num, BH, col_num, BW))
    reduced = reduced.swapaxes(1, 2).any(-1).any(-1).contiguous()
    return reduced


@pytest.mark.parametrize("BH", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("BW", [4, 8, 16, 32, 64, 128])
def test_bcsr(
    BH: int,
    BW: int,
    H: int = 1024,
    W: int = 768,
    batch_size: int = 2,
    sparsity: float = 0.999,
    random_seed: int = 2023,
):
    torch.manual_seed(random_seed)
    mask = block_mask(shape=(H, W), sparsity=sparsity)
    BCSR_function = get_bcs_function(BH, BW, True, False)

    # Test BCSR indexes
    reduced_mask = reduce_mask_ref(mask, BH, BW)
    BCSR_indexes = BCSR_function.build_indexes(mask)
    torch.testing.assert_close(BCSR_indexes.get_block_mask(), reduced_mask)

    # Test convert & inverse functions
    dense = torch.rand((batch_size, H, W), dtype=torch.float32, device='cuda') * mask
    sparse_val = BCSR_indexes.convert(dense)
    torch.testing.assert_close(BCSR_indexes.inverse(sparse_val), dense)

    # Test sum function
    torch.testing.assert_close(BCSR_indexes.sum(sparse_val, axis=-1), dense.sum(-1))
    torch.testing.assert_close(BCSR_indexes.sum(sparse_val, axis=-2), dense.sum(-2))


@pytest.mark.parametrize("BH", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("BW", [4, 8, 16, 32, 64, 128])
def test_bcsc(
    BH: int,
    BW: int,
    H: int = 1024,
    W: int = 768,
    batch_size: int = 2,
    sparsity: float = 0.999,
    random_seed: int = 2023,
):
    torch.manual_seed(random_seed)
    mask = block_mask(shape=(H, W), sparsity=sparsity)
    BCSC_function = get_bcs_function(BH, BW, False, True)

    # Test BCSC indexes
    reduced_mask = reduce_mask_ref(mask, BH, BW)
    BCSC_indexes = BCSC_function.build_indexes(mask)
    torch.testing.assert_close(BCSC_indexes.get_block_mask(), reduced_mask)

    # Test convert & inverse functions
    dense = torch.rand((batch_size, H, W), dtype=torch.float32, device='cuda') * mask
    sparse_val = BCSC_indexes.convert(dense)
    torch.testing.assert_close(BCSC_indexes.inverse(sparse_val), dense)

    # Test sum function
    torch.testing.assert_close(BCSC_indexes.sum(sparse_val, axis=-1), dense.sum(-1))
    torch.testing.assert_close(BCSC_indexes.sum(sparse_val, axis=-2), dense.sum(-2))


@pytest.mark.parametrize("BH", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("BW", [4, 8, 16, 32, 64, 128])
def test_bcsrc(
    BH: int,
    BW: int,
    H: int = 1024,
    W: int = 768,
    batch_size: int = 2,
    sparsity: float = 0.999,
    random_seed: int = 2023,
):
    torch.manual_seed(random_seed)
    mask = block_mask(shape=(H, W), sparsity=sparsity)
    BCSRC_function = get_bcs_function(BH, BW, True, True)

    # Test BCSR indexes
    reduced_mask = reduce_mask_ref(mask, BH, BW)
    BCSRC_indexes = BCSRC_function.build_indexes(mask)
    torch.testing.assert_close(BCSRC_indexes.get_block_mask(), reduced_mask)

    # Test BCSC indexes and block index
    BCSR_row_idx = BCSRC_indexes.BCSR_idx.bitwise_right_shift(16)
    BCSR_col_idx = BCSRC_indexes.BCSR_idx.bitwise_and(0xffff)
    BCSC_row_idx = BCSRC_indexes.BCSC_idx.to(torch.int32).bitwise_and(0xffff)
    BCSC_col_idx = BCSRC_indexes.BCSC_idx.to(torch.int32).bitwise_right_shift(16)
    BCSC_block_index = BCSRC_indexes.BCSC_idx.bitwise_right_shift(32)
    torch.testing.assert_close(BCSR_row_idx[BCSC_block_index], BCSC_row_idx)
    torch.testing.assert_close(BCSR_col_idx[BCSC_block_index], BCSC_col_idx)

    # Test convert & inverse functions
    dense = torch.rand((batch_size, H, W), dtype=torch.float32, device='cuda') * mask
    sparse_val = BCSRC_indexes.convert(dense)
    torch.testing.assert_close(BCSRC_indexes.inverse(sparse_val), dense)

    # Test sum function
    torch.testing.assert_close(BCSRC_indexes.sum(sparse_val, axis=-1), dense.sum(-1))
    torch.testing.assert_close(BCSRC_indexes.sum(sparse_val, axis=-2), dense.sum(-2))
