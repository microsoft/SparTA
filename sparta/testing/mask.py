# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Any

import torch


def block_mask(
    shape: Tuple[int],
    granularity: Tuple[int] = (1, 1),
    sparsity: float = 0.99,
    algo: str = 'rand',
    device: Any = 'cuda',
):
    """Generate a 2D uint8 tensor as block mask.

    Args:
        shape (Tuple[int]): Mask shape.
        granularity (Tuple[int]): block shape. (1, 1) means finegrained mask.
        sparsity (float): The ratio of empty blocks.
        algo (str): Algorithm to generate mask. Only random generator is supported now.
    """
    assert len(shape) == 2, 'only 2D mask is supported'
    assert len(granularity) == 2, 'only 2D mask is supported'
    assert shape[0] % granularity[0] == 0, f'invalid granularity shape {granularity}'
    assert shape[1] % granularity[1] == 0, f'invalid granularity shape {granularity}'
    if algo == 'rand':
        return random_block_mask(shape, granularity, sparsity, device)
    else:
        raise ValueError(f'unsupported mask generator: {algo}')


def random_block_mask(
    shape: Tuple[int],
    granularity: Tuple[int],
    sparsity: float = 0.99,
    device: Any = 'cuda',
):
    """Randomly generate a 2D uint8 tensor as block mask.

    Args:
        shape (Tuple[int]): Mask shape.
        granularity (Tuple[int]): block shape.
        sparsity (float): The ratio of empty blocks.
    """
    compressed_shape = (shape[0] // granularity[0], shape[1] // granularity[1])
    mask = random_mask(compressed_shape, sparsity, device)
    mask = mask.reshape(compressed_shape + (1, 1)).tile((1, 1) + granularity)
    return mask.swapaxes(1, 2).reshape(shape).contiguous()


def random_mask(shape: Tuple[int], sparsity: float = 0.99, device: Any = 'cuda'):
    """Randomly generate a 2D uint8 tensor as finegrained mask.

    Args:
        shape (Tuple[int]): Mask shape.
        sparsity (float): The ratio of empty items.
    """
    return (torch.rand(shape, device=device) > sparsity).to(torch.uint8)
