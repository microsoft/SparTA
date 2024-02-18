# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import itertools
from typing import Dict, Any

import torch
import pandas as pd

from sparta.operators.sparse_matmul import SparseBatchMatMulForward
from sparta.testing import block_mask


SIZE = 4096
RANDOM_SEED = 2022
SPEC_SEARCH_SPACE = {
    'mode': ['sdd', 'dsd', 'dds'],
    'trans_A': [False, True],
    'trans_B': [False, True],
}
PARAM_SEARCH_SPACE = {
    'sparta': {
        'BLOCK_SIZE_M_VALUE': [8, 16, 32, 64, 128],
        'BLOCK_SIZE_K_VALUE': [8, 16, 32, 64, 128],
        'BLOCK_SIZE_N_VALUE': [8, 16, 32, 64, 128],
        'THREAD_SIZE_M_VALUE': [2, 4, 8, 16],
        'THREAD_SIZE_K_VALUE': [2, 4, 8, 16],
        'THREAD_SIZE_N_VALUE': [2, 4, 8, 16],
    },
    'openai': {
        'BLOCK_SIZE_M_VALUE': [32],
        'BLOCK_SIZE_K_VALUE': [64],
        'BLOCK_SIZE_N_VALUE': [32],
    },
}
HYPER_PARAMS = ['mode', 'trans_A', 'trans_B', 'BM', 'BK', 'BN']


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def test_matmul_kernel(
    impl: str,
    operator: SparseBatchMatMulForward,
    params: Dict[str, Any],
):
    try:
        operator.build(config={'forward': {'_impl': impl, **params}})
        latency = operator.profile_kernel('forward', num_warmups=10, num_iters=10, cuda=False)
    except:
        latency = float('inf')

    return latency


def make_matmul_lut(impl: str):
    major, minor = torch.cuda.get_device_capability()
    lut_file = os.path.join(
        'sparta',
        'kernels',
        'look_up_tables',
        f'matmul.{impl}.{major}{minor}.csv'
    )
    log_file = os.path.join(
        'test',
        'lut_maker',
        f'matmul.{impl}.{major}{minor}.log.csv'
    )
    _logger.info(f'========== Making LUT: {lut_file} ==========')

    num = 1
    spec_keys, spec_values = [], []
    for k, v in SPEC_SEARCH_SPACE.items():
        spec_keys.append(k)
        spec_values.append(v)
        num *= len(v)
    param_keys, param_alts, param_values = [], [], []
    for k, v in PARAM_SEARCH_SPACE[impl].items():
        param_keys.append(k)
        param_alts.append(f'{k[0]}{k[-7]}')
        param_values.append(v)
        num *= len(v)
    bits = len(str(num))

    with open(log_file, 'w') as f:
        header = ','.join(spec_keys) + ',' + ','.join(param_alts) + ',latency\n'
        header = header.replace('BLOCK', 'B').replace('THREAD', 'T')
        header = header.replace('_SIZE_', '').replace('_VALUE', '')
        f.write(header)

    torch.manual_seed(RANDOM_SEED)
    A = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    B = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    mask = block_mask((SIZE, SIZE), sparsity=0, device='cuda')

    iters = 0
    for specs in itertools.product(*spec_values):
        mode, trans_A, trans_B = specs
        operator = SparseBatchMatMulForward(mode, trans_A, trans_B, False, True)
        operator.set_mask(mask)
        operator.reference(A, B)
        for params in itertools.product(*param_values):
            param_dict = {k: v for k, v in zip(param_keys, params)}
            latency = test_matmul_kernel(impl, operator, param_dict)
            with open(log_file, 'a') as f:
                items = [mode, trans_A, trans_B, *params, latency]
                f.write(','.join([str(x) for x in items]) + '\n')
            iters += 1
            _logger.info(f'[{str(iters).rjust(bits)} / {num}] {params} => {latency} ms')

    df = pd.read_csv(log_file)
    df = df.loc[df.groupby(HYPER_PARAMS).aggregate({'latency': 'idxmin'})['latency']]
    with open(lut_file, 'w') as f:
        f.write(df.reset_index(drop=True).to_csv(index=False))

    _logger.info(f'========== Finished. Output: {lut_file} ==========')


if __name__ == '__main__':
    _logger.setLevel(logging.DEBUG)
    make_matmul_lut('sparta')
    make_matmul_lut('openai')
