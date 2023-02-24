# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import itertools
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd

from sparta.operators.sparse_softmax import SparseBatchSoftmax
from sparta.testing import block_mask


SIZE = 4096
RANDOM_SEED = 2022
SEARCH_SPACE = {
    'BLOCK_SIZE_H_VALUE': [8, 16, 32, 64, 128],
    'BLOCK_SIZE_W_VALUE': [8, 16, 32, 64, 128],
    'ROW_TILE_VALUE': [1, 2, 4, 8, 16],
}
HYPER_PARAMS = ['BH', 'BW']


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def test_softmax_kernel(
    impl: str,
    operator: SparseBatchSoftmax,
    direction: str,
    params: Dict[str, Any],
):
    try:
        operator.build(config={direction: {'_impl': impl, **params}})
        latency = operator.profile_kernel(direction, num_warmups=10, num_iters=10, cuda=False)
    except:
        latency = float('inf')

    return latency


def make_softmax_lut(impl: str, direction: str):
    major, minor = torch.cuda.get_device_capability()
    lut_file = os.path.join(
        impl,
        'kernels',
        'look_up_tables',
        f'softmax.{direction}.{impl}.{major}{minor}.csv'
    )
    log_file = os.path.join(
        'test',
        'lut_maker',
        f'softmax.{direction}.{impl}.{major}{minor}.log.csv'
    )
    _logger.info(f'========== Making LUT: {lut_file} ==========')

    num = 1
    keys, alts, values = [], [], []
    for k, v in SEARCH_SPACE.items():
        keys.append(k)
        alt = [s[0] for s in k.split('_')]
        alts.append(f'{alt[0]}{alt[-2]}')
        values.append(v)
        num *= len(v)
    bits = len(str(num))

    with open(log_file, 'w') as f:
        f.write(','.join(alts) + ',latency\n')

    torch.manual_seed(RANDOM_SEED)
    x = torch.rand(size=(1, SIZE, SIZE), device='cuda', requires_grad=True)
    grad_y = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    mask = block_mask((SIZE, SIZE), sparsity=0, device='cuda')

    operator = SparseBatchSoftmax(compressed=True, temperature=np.float32(1 / np.sqrt(SIZE)))
    operator.get_sparse_attr().set_mask(mask)
    operator.reference(x).backward(grad_y)

    iters = 0
    for params in itertools.product(*values):
        param_dict = {k: v for k, v in zip(keys, params)}
        latency = test_softmax_kernel(impl, operator, direction, param_dict)
        with open(log_file, 'a') as f:
            f.write(','.join([str(x) for x in params]) + f',{latency}\n')
        iters += 1
        _logger.info(f'[{str(iters).rjust(bits)} / {num}] {params} => {latency} ms')

    df = pd.read_csv(log_file)
    df = df.loc[df.groupby(HYPER_PARAMS).aggregate({'latency': 'idxmin'})['latency']]
    with open(lut_file, 'w') as f:
        f.write(df.reset_index(drop=True).to_csv(index=False))

    _logger.info(f'========== Finished. Output: {lut_file} ==========')


if __name__ == '__main__':
    _logger.setLevel(logging.DEBUG)
    make_softmax_lut('sparta', 'forward')
    make_softmax_lut('sparta', 'backward')
