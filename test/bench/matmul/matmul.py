# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Any, Dict, Tuple, Callable

import torch
import triton
import tabulate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sparta.nn import SparseBatchMatMul
from sparta.testing import block_mask, profile


M, K, N = 3072, 768, 4096
GRANULARITY_LIST = [1, 8, 32]
SPARSITY_LIST = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
WORK_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_WARMUPS = 100
NUM_ITERS = 100


def profile_matmul(
    func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: Dict[str, torch.Tensor],
):
    forward_latency = profile(
        func=func,
        inputs=[data['A'], data['B']],
        target_outputs=[data['C']],
        num_warmups=NUM_WARMUPS,
        num_iters=NUM_ITERS,
    )

    def matmul_forward_backward(A: torch.Tensor, B: torch.Tensor, grad_C: torch.Tensor):
        C = func(A, B)
        C.backward(grad_C)
        return C, A.grad, B.grad

    data['A'].requires_grad = True
    data['B'].requires_grad = True
    total_latency = profile(
        func=matmul_forward_backward,
        inputs=[data['A'], data['B'], data['grad_C']],
        target_outputs=[data['C'], data['grad_A'], data['grad_B']],
        num_warmups=NUM_WARMUPS,
        num_iters=NUM_ITERS,
    )

    return forward_latency, total_latency - forward_latency


def prepare_data(
    M: int,
    K: int,
    N: int,
    granularity: Tuple[int, int],
    sparsity: float,
    ndim: int = 3,
    seed: int = 2022,
    device: Any = 'cuda',
):
    torch.manual_seed(seed)
    data: Dict[str, torch.Tensor] = {}
    data['A'] = torch.rand((M, K), dtype=torch.float32, device=device)
    data['B'] = torch.rand((N, K), dtype=torch.float32, device=device)
    mask = block_mask((N, K), granularity, sparsity, device=device)
    data['B'] *= mask
    data['C'] = torch.matmul(data['A'], data['B'].T)
    data['grad_C'] = torch.rand((M, N), dtype=torch.float32, device=device)
    data['grad_A'] = torch.matmul(data['grad_C'], data['B'])
    data['grad_B'] = torch.matmul(data['grad_C'].T, data['A'])
    # No grad_B *= mask because the DD=>S known issue
    for _ in range(ndim - 2):
        data = {k: v.unsqueeze(0) for k, v in data.items()}
    return data, mask


def profile_triton_matmul(
    block_size: int,
    M: int,
    K: int,
    N: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(M, K, N, granularity, sparsity, ndim=4, device=device)
    if mask.sum() == 0:
        return 0., 0.

    layout = mask.reshape(N // block_size, block_size, K // block_size, block_size)
    layout = layout.swapaxes(1, 2).any(-1).any(-1).unsqueeze(0).to(torch.int32).cpu()

    data['B'] = triton.testing.sparsify_tensor(data['B'], layout, block_size)
    data['grad_B'] = triton.testing.sparsify_tensor(data['grad_B'], layout, block_size)

    triton_matmul = triton.ops.blocksparse.matmul(
        layout=layout,
        block=block_size,
        mode='dds',
        device=device,
        trans_a=False,
        trans_b=True,
    )

    return profile_matmul(triton_matmul, data)


def profile_sparta_matmul(
    config: Dict[str, Any],
    M: int,
    K: int,
    N: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(M, K, N, granularity, sparsity, ndim=3, device=device)
    if mask.sum() == 0:
        return 0., 0.

    sparta_matmul = SparseBatchMatMul(
        mode='dsd',
        transpose_A=False,
        transpose_B=True,
        biased=False,
        compressed=True,
    )
    sparta_matmul.set_mask(mask)
    sparta_matmul.build(config, sample_inputs=[data['A'], data['B']])

    indexes = sparta_matmul.get_sparse_indexes()
    data['B'] = indexes.convert(data['B'])
    data['grad_B'] = indexes.convert(data['grad_B'])

    return profile_matmul(sparta_matmul, data)


def profile_dense_matmul(
    M: int,
    K: int,
    N: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(M, K, N, granularity, sparsity, ndim=3, device=device)
    if mask.sum() == 0:
        return 0., 0.

    dense_matmul = lambda A, B: torch.einsum('bik, bjk -> bij', A, B)

    return profile_matmul(dense_matmul, data)


def get_sparta_config(configs: pd.DataFrame, granularity: int, sparsity: float):
    condition = (configs['granularity'] == granularity) & (configs['sparsity'] == sparsity)
    config: Dict[str, Any] = {}
    for key, val in configs[condition].iloc[0, :].to_dict().items():
        if ';' in key:
            kernel, param = key.split(';')
            if kernel not in config:
                config[kernel] = {}
            if param == '_impl':
                config[kernel][param] = val
            elif val > 0:
                config[kernel][param] = int(val)
    return config


def profile_all(log_path: str, device: Any = 'cuda'):
    cols = ['method', 'M', 'K', 'N', 'granularity', 'sparsity', 'forward', 'backward']
    with open(log_path, 'w') as f:
        f.write(','.join(cols) + '\n')
    sparta_configs = pd.read_csv(os.path.join(WORK_DIR, 'sparta_params.csv'))
    for g in GRANULARITY_LIST:
        for s in SPARSITY_LIST:
            print(f'========== Granularuty: {g} Sparsity: {s} ==========')
            latency: Dict[str, Tuple[int, int]] = {}
            latency['dense'] = profile_dense_matmul(M, K, N, (g, g), s, device)
            config = get_sparta_config(sparta_configs, g, s)
            latency['sparta'] = profile_sparta_matmul(config, M, K, N, (g, g), s, device)
            # for block in [16, 32, 64]:
            #     latency[f'triton-{block}'] = profile_triton_matmul(block, M, K, N, (g, g), s, device)
            with open(log_path, 'a') as f:
                for method, (lat_f, lat_b) in latency.items():
                    f.write(f'{method},{M},{K},{N},{g},{s},{lat_f},{lat_b}\n')
            print(tabulate.tabulate(
                tabular_data=[[method, lat_f, lat_b] for method, (lat_f, lat_b) in latency.items()],
                headers=['Method', 'Forward latency / ms', 'Backward latency / ms'],
            ))
    print(f'==================== Finished ====================')


def plot_latency(log_path: str):
    df = pd.read_csv(log_path)
    methods = sorted(set(df['method']))

    markers = {'dense': 'x', 'triton-16': '^', 'triton-32': 's', 'triton-64': 'o', 'sparta': '*'}
    colors = {'dense': 'C0', 'triton-16': 'C1', 'triton-32': 'C1', 'triton-64': 'C1', 'sparta': 'C2'}

    xticks, xlabels = [0], [0]
    while xlabels[-1] < df['sparsity'].max():
        xticks.append(xticks[-1] + 1)
        xlabels.append(1 - (1 - xlabels[-1]) * 0.1)

    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3)
    for i, granularity in enumerate(GRANULARITY_LIST):
        for j, direction in enumerate(['forward', 'backward']):
            ax = plt.subplot(len(GRANULARITY_LIST) * 100 + 21 + i * 2 + j)
            for method in methods:
                sub_df = df[(df['method'] == method) & (df['granularity'] == granularity)]
                ax.plot(
                    -np.log10(1 - sub_df['sparsity']),
                    sub_df[direction],
                    alpha=0.5,
                    color=colors[method],
                    marker=markers[method],
                    label=method,
                )
            ax.set_title(f'Granularity = {granularity}, {direction}')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel('Sparsity')
            ax.set_ylabel(f'Latency / ms')
            ax.set_ylim(bottom=0)
            ax.legend()
    plt.suptitle(f'{M}x{K}x{N} MatMul Latency')

    fig_path = os.path.join(os.path.dirname(log_path), 'latency.png')
    plt.savefig(fig_path)
    print(f'Figure saved to: {fig_path}')


def main():
    log_path = os.path.join(WORK_DIR, 'latency.csv')
    profile_all(log_path, device='cuda')
    plot_latency(log_path)


if __name__ == '__main__':
    main()
