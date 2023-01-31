# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
import itertools
from typing import Dict, Tuple

import torch
import pandas as pd

from sparta.specializer.kernels import SparTASparseMatMulKernel
from sparta.tesa import BCSIndexes
from sparta.testing import block_mask, profile


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


def prepare_data(
    batch: int = 4,
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    mode: str = 'dds',
    trans_A: bool = False,
    trans_B: bool = False,
    biased: bool = False,
    requires_grad: bool = False,
    random_seed: int = 2022,
):
    inputs = ['A', 'B']
    outputs = ['C']
    shapes = {
        'A': (K, M) if trans_A else (M, K),
        'B': (N, K) if trans_B else (K, N),
        'C': (M, N),
    }
    if biased:
        inputs.append('bias')
        shapes['bias'] = (N, )

    torch.manual_seed(random_seed)
    data: Dict[str, torch.Tensor] = {}
    for x in inputs:
        data[f'input_{x}'] = torch.rand(size=(batch, *shapes[x]), device='cuda')
    if requires_grad:
        for y in outputs:
            data[f'input_grad_{y}'] = torch.rand(size=(batch, *shapes[y]), device='cuda')

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    mask = block_mask(shapes[sparse_port], block=granularity, sparsity=sparsity, device='cuda')
    add_mask(data, {sparse_port: mask}, sparse_port, 'input')

    if requires_grad:
        for x in inputs:
            data[f'input_{x}'].requires_grad = True

    input_A = data['input_A'].swapaxes(1, 2) if trans_A else data['input_A']
    input_B = data['input_B'].swapaxes(1, 2) if trans_B else data['input_B']
    data['target_C'] = torch.bmm(input_A, input_B)
    if biased:
        data['target_C'] += data['input_bias'].unsqueeze(1)

    if requires_grad:
        data['target_C'].backward(data['input_grad_C'])
        data['target_grad_A'] = data['input_A'].grad
        data['input_A'].grad = None
        data['target_grad_B'] = data['input_B'].grad
        data['input_B'].grad = None
        if biased:
            data['target_grad_bias'] = data['input_bias'].grad
            data['input_bias'].grad = None

    add_mask(data, {sparse_port: mask}, sparse_port, 'target')

    return data, {sparse_port: mask}


def add_mask(
    data: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor], 
    sparse_port: str,
    stage: str,
):
    for name, val in data.items():
        if name.startswith(stage) and name.endswith(sparse_port):
            val *= masks[sparse_port]


def compress_data(
    indexes: BCSIndexes,
    sparse_port: str,
    data: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
):
    for name in data:
        if name.endswith(sparse_port):
            data[name] = indexes.convert(data[name].detach())
    masks[sparse_port] = indexes.convert(masks[sparse_port].to(torch.float32)).to(torch.uint8)
    if sparse_port in ['A', 'B']:
        data[f'input_{sparse_port}'].requires_grad = True


def check_results(data: Dict[str, torch.Tensor]):
    for name, val in data.items():
        if name.startswith('target_'):
            torch.testing.assert_close(val, data[name.replace('target', 'output')], msg=name)


def test_sparse_matmul_kernel(
    mode: str,
    trans_A: bool,
    trans_B: bool,
    BM: int,
    BK: int,
    BN: int,
    TM: int,
    TK: int,
    TN: int,
    biased: bool = False,
    compressed: bool = True,
    batch: int = 1,
    M: int = 4096,
    K: int = 4096,
    N: int = 4096,
    granularity: Tuple[int, int] = (1, 1),
    sparsity: float = 0,
):
    data, masks = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, biased, False)

    try:
        kernel = SparTASparseMatMulKernel(
            mode=mode,
            biased=biased,
            transpose_A=trans_A,
            transpose_B=trans_B,
            compressed=compressed,
        )

        for sparse_port, mask in masks.items():
            kernel.ports[sparse_port].set_mask(mask)
        kernel.set_shape(batch, M, K, N)
        kernel.compile({
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        })

        sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
        if compressed:
            compress_data(kernel.ports[sparse_port].indexes, sparse_port, data, masks)

        inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
        input_data = [data[f'input_{x}'].detach() for x in inputs]

        latency = profile(kernel, input_data, num_warmups=10, num_iters=10, cuda=False)
        torch.cuda.synchronize()
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

    for i, params in enumerate(itertools.product(*values)):
        latency = test_sparse_matmul_kernel(**{k: v for k, v in zip(keys, params)})
        with open(log_file, 'a') as f:
            f.write(','.join([str(x) for x in params]) + f',{latency}\n')
        _logger.info(f'[{i} / {num}] {params} => {latency} ms')

    df = pd.read_csv(log_file)
    df = df.groupby(['mode', 'trans_A', 'trans_B', 'BM', 'BK', 'BN']).min('latency')
    with open(lut_file, 'w') as f:
        f.write(df.reset_index().to_csv(index=False))

    _logger.info(f'========== Finished. Output: {lut_file} ==========')
