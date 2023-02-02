# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import itertools

import torch
import pandas as pd

from sparta.specializer.kernels import SparTASparseMatMulKernel
from sparta.testing import block_mask


SIZE = 4096
RANDOM_SEED = 2022
SEARCH_SPACE = {
    'mode': ['sdd', 'dsd', 'dds'],
    'trans_A': [False, True],
    'trans_B': [False, True],
    'BM': [8, 16, 32, 64, 128],
    'BK': [8, 16, 32, 64, 128],
    'BN': [8, 16, 32, 64, 128],
    'TM': [2, 4, 8, 16],
    'TK': [2, 4, 8, 16],
    'TN': [2, 4, 8, 16],
}


_logger = logging.Logger(__name__)
_handler = logging.StreamHandler()
_logger.addHandler(_handler)


def test_sparta_matmul_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    trans_A: bool,
    trans_B: bool,
    BM: int,
    BK: int,
    BN: int,
    TM: int,
    TK: int,
    TN: int,
):
    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    kernel = SparTASparseMatMulKernel(
        mode=mode,
        biased=False,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=True,
    )

    try:
        kernel.ports[sparse_port].set_mask(mask)
        kernel.set_shape(1, SIZE, SIZE, SIZE)
        kernel.compile({
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        })
        latency = kernel.test([A, B], num_warmups=10, num_iters=10, cuda=False)
    except:
        latency = float('inf')

    return latency


if __name__ == '__main__':
    _logger.setLevel(logging.DEBUG)
    major, minor = torch.cuda.get_device_capability()
    lut_file = os.path.join(
        'sparta',
        'specializer',
        'kernels',
        'look_up_tables',
        f'matmul.sparta.{major}{minor}.csv'
    )
    log_file = os.path.join(
        'test',
        'lut_maker',
        f'matmul.sparta.{major}{minor}.log.csv'
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

    torch.manual_seed(2022)
    A = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    B = torch.rand(size=(1, SIZE, SIZE), device='cuda')
    mask = block_mask((SIZE, SIZE), sparsity=0, device='cuda')

    for i, params in enumerate(itertools.product(*values)):
        latency = test_sparta_matmul_kernel(A, B, mask, **{k: v for k, v in zip(keys, params)})
        with open(log_file, 'a') as f:
            f.write(','.join([str(x) for x in params]) + f',{latency}\n')
        _logger.info(f'[{i} / {num}] {params} => {latency} ms')

    df = pd.read_csv(log_file)
    df = df.groupby(['mode', 'trans_A', 'trans_B', 'BM', 'BK', 'BN']).min('latency')
    with open(lut_file, 'w') as f:
        f.write(df.reset_index().to_csv(index=False))

    _logger.info(f'========== Finished. Output: {lut_file} ==========')
