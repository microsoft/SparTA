# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import torch


def block_mask(
    shape: Tuple[int], block: Tuple[int] = (1, 1),
    sparsity: float = 0.99, algo: str = 'rand'
):
    '''Generate a 2D bool tensor as block mask.

    Args:
        shape (tuple[int]): Mask shape.
        block (tuple[int]): Block shape. (1, 1) means finegrained mask.
        sparsity (float): The ratio of empty block number to total block number.
        algo (str): Algorithm to generate mask. Only random generator is supported now.
    '''
    assert len(shape) == 2, 'only 2D mask is supported'
    assert len(block) == 2, 'only 2D mask is supported'
    assert shape[0] % block[0] == 0, f'invalid block shape {block}'
    assert shape[1] % block[1] == 0, f'invalid block shape {block}'
    if algo == 'rand':
        return random_block_mask(shape, block, sparsity)
    else:
        raise ValueError(f'unsupported mask generator: {algo}')


def random_block_mask(shape: Tuple[int], block: Tuple[int], sparsity: float = 0.99):
    '''Randomly generate a 2D bool tensor as block mask.

    Args:
        shape (tuple[int]): Mask shape.
        block (tuple[int]): Block shape.
        sparsity (float): The ratio of empty block number to total block number.
    '''
    compressed_shape = (shape[0] // block[0], shape[1] // block[1])
    mask = random_mask(compressed_shape, sparsity)
    mask = mask.reshape(compressed_shape + (1, 1)).tile((1, 1) + block)
    return mask.swapaxes(1, 2).reshape(shape)


def random_mask(shape: Tuple[int], sparsity: float = 0.99):
    '''Randomly generate a 2D bool tensor as finegrained mask.

    Args:
        shape (tuple[int]): Mask shape.
        sparsity (float): The ratio of empty block number to total block number.
    '''
    return torch.rand(shape) > sparsity
