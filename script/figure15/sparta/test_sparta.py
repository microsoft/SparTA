# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import argparse
from typing import Dict, Tuple, Optional

import torch

from sparta.kernels import SparTASparseMatMulKernel
from sparta.tesa import BCSIndexes
from sparta.tuning import GridSearchTuner, TunableItemCfg
from sparta.testing import block_mask, profile


def prepare_data(
    batch: Optional[int] = 4,
    shape: Tuple[int, int, int] = (4096, 4096, 4096),
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    mode: str = 'dds',
    trans_A: bool = False,
    trans_B: bool = False,
    biased: bool = False,
    requires_grad: bool = False,
    mask: Optional[torch.Tensor] = None,
    random_seed: int = 2022,
):
    M, K, N = shape
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
        shape = shapes[x] if batch is None else (batch, *shapes[x])
        data[f'input_{x}'] = torch.rand(size=shape, device='cuda')
    if requires_grad:
        for y in outputs:
            shape = shapes[y] if batch is None else (batch, *shapes[y])
            data[f'input_grad_{y}'] = torch.rand(size=shape, device='cuda')

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    if mask is None:
        mask = block_mask(
            shape=shapes[sparse_port],
            granularity=granularity,
            sparsity=sparsity,
            device='cuda',
        )
    add_mask(data, mask, sparse_port, 'input')

    calc_target_data(data, requires_grad, trans_A, trans_B)
    add_mask(data, mask, sparse_port, 'target')

    return data, mask


def calc_target_data(
    data: Dict[str, torch.Tensor],
    requires_grad: bool,
    trans_A: bool,
    trans_B: bool,
):
    if requires_grad:
        for k, v in data.items():
            if k.startswith('input'):
                v.requires_grad = True

    if len(data['input_A'].shape) == 3:
        input_A = data['input_A'].swapaxes(1, 2) if trans_A else data['input_A']
        input_B = data['input_B'].swapaxes(1, 2) if trans_B else data['input_B']
        data['target_C'] = torch.bmm(input_A, input_B)
        if 'input_bias' in data:
            data['target_C'] += data['input_bias'].unsqueeze(1)
    else:
        input_A = data['input_A'].T if trans_A else data['input_A']
        input_B = data['input_B'].T if trans_B else data['input_B']
        data['target_C'] = torch.mm(input_A, input_B)
        if 'input_bias' in data:
            data['target_C'] += data['input_bias']

    if requires_grad:
        data['target_C'].backward(data['input_grad_C'])
        data['target_grad_A'] = data['input_A'].grad
        data['input_A'].grad = None
        data['target_grad_B'] = data['input_B'].grad
        data['input_B'].grad = None
        if 'input_bias' in data:
            data['target_grad_bias'] = data['input_bias'].grad
            data['input_bias'].grad = None


def add_mask(
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    sparse_port: str,
    stage: str,
):
    for name, val in data.items():
        if name.startswith(stage) and name.endswith(sparse_port):
            val *= mask


def compress_data(
    indexes: BCSIndexes,
    sparse_port: str,
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    requires_grad: bool,
):
    for name in data:
        if name.endswith(sparse_port):
            data[name] = indexes.convert(data[name].detach())
    mask = indexes.convert(mask.to(torch.float32)).to(torch.uint8)
    if sparse_port in ['A', 'B'] and requires_grad:
        data[f'input_{sparse_port}'].requires_grad = True
    return data, mask


def check_results(data: Dict[str, torch.Tensor]):
    for name, val in data.items():
        if name.startswith('target_'):
            out = data[name.replace('target', 'output')]
            torch.testing.assert_close(out, val, atol=1e-4, rtol=1e-4, msg=name)


def get_sparse_matmul_kernel(
    mode: str = 'sdd',
    biased: bool = False,
    compressed: bool = False,
    trans_A: bool = False,
    trans_B: bool = False,
    batch: Optional[int] = None,
    shape: Tuple[int, int, int] = (4096, 4096, 4096),
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    params: Dict = None,
):
    data, mask = prepare_data(
        batch, shape,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        False,
    )

    batched = batch is not None
    kernel = SparTASparseMatMulKernel(
        mode=mode,
        biased=biased,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
        batched=batched,
    )
    kernel.attr.set_mask(mask)
    batch = 1 if batch is None else batch

    kernel.compile(params, (batch, *shape))

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]

    if compressed:
        data, mask = compress_data(kernel.attr.indexes, sparse_port, data, mask, False)

    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    input_data = [data[f'input_{x}'] for x in inputs]

    data['output_C'] = kernel(*input_data)
    add_mask(data, mask, sparse_port, 'output')
    check_results(data)

    return kernel, input_data


def search_params(sparsity: float, block_M: int, block_K: int, output_path: str):
    static_params = {
        '_impl': 'sparta',
    }
    search_space = {
        'BLOCK_SIZE_M_VALUE': TunableItemCfg('choice', [8, 16, 32]),
        'BLOCK_SIZE_K_VALUE': TunableItemCfg('choice', [8, 16, 32]),
        'BLOCK_SIZE_N_VALUE': TunableItemCfg('choice', [8, 16, 32, 64]),
    }
    if block_M in [8, 16, 32, 64]:
        del search_space['BLOCK_SIZE_M_VALUE']
        static_params['BLOCK_SIZE_M_VALUE'] = block_M
    if block_K in [8, 16, 32, 64]:
        del search_space['BLOCK_SIZE_K_VALUE']
        static_params['BLOCK_SIZE_K_VALUE'] = block_K

    def eval_kernel(idx, cfg):
        print(f'#{idx} ({list(cfg.values())}):', end='')
        try:
            kernel, input_data = get_sparse_matmul_kernel(
                granularity=(block_M, block_K),
                sparsity=sparsity,
                params=dict(**static_params, **cfg),
            )
            result = profile(kernel, input_data)
            print(result)
        except KeyboardInterrupt as err:
            raise err
        except BaseException:
            result = float('inf')
            print('Failed')
        return result

    print(f'=============== Searching ===============')
    print(f'sparsity = {sparsity}, granularity = ({block_M}, {block_K}) ')
    tuner = GridSearchTuner(search_space, eval_kernel)
    tuner.tune()
    params = dict(**static_params, **tuner.best_config)
    print(f'best params: {params}')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(json.dumps(params, indent='\t'))
    return params


def get_params(sparsity: float, block_M: int, block_K: int):
    params_path = os.path.join(
        os.path.dirname(__file__),
        'sparta_params',
        f'{sparsity}_{block_M}x{block_K}.json',
    )
    try:
        with open(params_path) as f:
            params = json.loads(f.read())
        print(f'Read SparTA params from {params_path}')
    except FileNotFoundError:
        print(f'SparTA params file not found.')
        params = search_params(sparsity, block_M, block_K, params_path)
    except json.JSONDecodeError:
        print(f'Invalid SparTA params file.')
        params = search_params(sparsity, block_M, block_K, params_path)
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SparTA kernels')
    parser.add_argument('sparsity', type=float)
    parser.add_argument('M', type=int)
    parser.add_argument('K', type=int)
    parser.add_argument('N', type=int)
    parser.add_argument('block_M', type=int)
    parser.add_argument('block_K', type=int)
    args = parser.parse_args()
    params = get_params(args.sparsity, args.block_M, args.block_K)
    print(f'=============== Profiling ===============')
    kernel, input_data = get_sparse_matmul_kernel(
        shape=(args.M, args.K, args.N),
        granularity=(args.block_M, args.block_K),
        sparsity=args.sparsity,
        params=params,
    )
    latency = profile(kernel, input_data, num_warmups=200, num_iters=1000, cuda=False)
    print(f'SparTA latency(ms): {latency}\n')
    with open('sparta_results.csv', 'a') as f:
        f.write(f'{args.block_M}x{args.block_K},{args.sparsity},{latency}\n')
