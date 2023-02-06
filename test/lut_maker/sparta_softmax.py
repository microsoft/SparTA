# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import itertools
from typing import Dict

import torch
import numpy as np
import pandas as pd

from sparta.specializer.kernels import  SparTASparseSoftmaxForwardKernel, SparTASparseSoftmaxBackwardKernel
from sparta.testing import block_mask


SIZE = 4096
RANDOM_SEED = 2022
SEARCH_SPACE = {
    'BH': [8, 16, 32, 64, 128],
    'BW': [8, 16, 32, 64, 128],
    'RT': [1, 2, 4, 8, 16],
}
HYPER_PARAMS = ['BH', 'BW']


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def test_sparta_softmax_kernel(
    data: Dict,
    mask: torch.Tensor,
    direction: str,
    BH: int,
    BW: int,
    RT: int,
):
    if direction == 'forward':
        kernel = SparTASparseSoftmaxForwardKernel(compressed=True)
    elif direction == 'backward':
        kernel = SparTASparseSoftmaxBackwardKernel(compressed=True)
    else:
        raise ValueError(f'unrecognized direction: {direction}')

    try:
        kernel.ports['y'].set_mask(mask)
        kernel.set_shape(1, SIZE, SIZE)
        kernel.compile({
            'BLOCK_SIZE_H_VALUE': BH,
            'BLOCK_SIZE_W_VALUE': BW,
            'ROW_TILE_VALUE': RT,
        })
        if direction == 'forward':
            inputs = [data['x'], data['T']]
        else:
            inputs = [data['grad_y'], data['y'], data['T']]
        latency = kernel.test(inputs, num_warmups=10, num_iters=10, cuda=False)
    except:
        latency = float('inf')

    return latency


def make_sparta_softmax_lut(direction: str):
    major, minor = torch.cuda.get_device_capability()
    lut_file = os.path.join(
        'sparta',
        'specializer',
        'kernels',
        'look_up_tables',
        f'softmax.{direction}.sparta.{major}{minor}.csv'
    )
    log_file = os.path.join(
        'test',
        'lut_maker',
        f'softmax.{direction}.sparta.{major}{minor}.log.csv'
    )
    _logger.info(f'========== Making LUT: {lut_file} ==========')

    num = 1
    keys, values = [], []
    for k, v in SEARCH_SPACE.items():
        keys.append(k)
        values.append(v)
        num *= len(v)

    with open(log_file, 'w') as f:
        f.write(','.join(keys) + ',latency\n')

    torch.manual_seed(RANDOM_SEED)
    data = {}
    data['x'] = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    data['T'] = np.float32(1 / np.sqrt(SIZE))
    data['y'] = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    data['grad_y'] = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    mask = block_mask((SIZE, SIZE), sparsity=0, device='cuda')

    for i, params in enumerate(itertools.product(*values)):
        latency = test_sparta_softmax_kernel(
            data,
            mask,
            direction,
            **{k: v for k, v in zip(keys, params)}
        )
        with open(log_file, 'a') as f:
            f.write(','.join([str(x) for x in params]) + f',{latency}\n')
        _logger.info(f'[{i} / {num}] {params} => {latency} ms')

    df = pd.read_csv(log_file)
    df = df.loc[df.groupby(HYPER_PARAMS).aggregate({'latency': 'idxmin'})['latency']]
    with open(lut_file, 'w') as f:
        f.write(df.reset_index(drop=True).to_csv(index=False))

    _logger.info(f'========== Finished. Output: {lut_file} ==========')


if __name__ == '__main__':
    _logger.setLevel(logging.DEBUG)
    make_sparta_softmax_lut('forward')
    make_sparta_softmax_lut('backward')
