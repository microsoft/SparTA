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

from sparta.nn import SparseAttention
from sparta.testing import block_mask, profile, sparse_multi_head_attention_reference


Ns, Nt, E = 4096, 3072, 768
GRANULARITY_LIST = [1, 8, 32]
SPARSITY_LIST = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
WORK_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_WARMUPS = 100
NUM_ITERS = 100


def profile_attention(
    func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data: Dict[str, torch.Tensor],
    check_results: bool = True,
):
    if check_results:
        forward_targets = [data['out']]
        backward_targets = [data['out'], data['grad_query'], data['grad_key'], data['grad_value']]
    else:
        forward_targets = None
        backward_targets = None

    forward_latency = profile(
        func=func,
        inputs=[data['query'], data['key'], data['value']],
        target_outputs=forward_targets,
        num_warmups=NUM_WARMUPS,
        num_iters=NUM_ITERS,
    )

    def matmul_forward_backward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        out = func(query, key, value)
        out.backward(grad_out)
        return out, query.grad, key.grad, value.grad

    data['query'].requires_grad = True
    data['key'].requires_grad = True
    data['value'].requires_grad = True
    total_latency = profile(
        func=matmul_forward_backward,
        inputs=[data['query'], data['key'], data['value'], data['grad_out']],
        target_outputs=backward_targets,
        num_warmups=NUM_WARMUPS,
        num_iters=NUM_ITERS,
    )

    return forward_latency, total_latency - forward_latency


def prepare_data(
    Ns: int,
    Nt: int,
    E: int,
    granularity: Tuple[int, int],
    sparsity: float,
    ndim: int = 3,
    seed: int = 2022,
    device: Any = 'cuda',
):
    torch.manual_seed(seed)
    data: Dict[str, torch.Tensor] = {}
    data['query'] = torch.rand(size=(1, Nt, E), device=device)
    data['key'] = torch.rand(size=(1, Ns, E), device=device)
    data['value'] = torch.rand(size=(1, Ns, E), device=device)
    data['grad_out'] = torch.rand(size=(1, Nt, E), device=device)
    mask = block_mask((Nt, Ns), granularity, sparsity=sparsity, device=device)
    data['query'].requires_grad = True
    data['key'].requires_grad = True
    data['value'].requires_grad = True
    inputs = [data['query'], data['key'], data['value']]
    data['out'] = sparse_multi_head_attention_reference(*inputs, mask)
    data['out'].backward(data['grad_out'])
    data['grad_query'] = data['query'].grad
    data['grad_key'] = data['key'].grad
    data['grad_value'] = data['value'].grad
    data['query'].requires_grad = False
    data['key'].requires_grad = False
    data['value'].requires_grad = False
    for _ in range(ndim - 3):
        data = {k: v.unsqueeze(0) for k, v in data.items()}
    return data, mask


def profile_triton_attention(
    block_size: int,
    Ns: int,
    Nt: int,
    E: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(Ns, Nt, E, granularity, sparsity, ndim=4, device=device)
    if mask.sum() == 0:
        return 0., 0.

    layout = mask.reshape(Nt // block_size, block_size, Ns // block_size, block_size)
    layout = layout.swapaxes(1, 2).any(-1).any(-1).unsqueeze(0).to(torch.int32).cpu()

    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(
        layout=layout,
        block=block_size,
        mode="sdd",
        trans_a=False,
        trans_b=True,
        device=device,
    )
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(
        layout=layout,
        block=block_size,
        mode="dsd",
        trans_a=False,
        trans_b=False,
        device=device,
    )
    sparse_softmax = triton.ops.blocksparse.softmax(
        layout=layout,
        block=block_size,
        device=device
    )
    scale = 1 / np.sqrt(E)

    def triton_attention(query: torch.Tensor,  key: torch.Tensor, value: torch.Tensor):
        w = sparse_dot_sdd_nt(query, key)
        w = sparse_softmax(w, scale=scale, is_causal=True)
        a = sparse_dot_dsd_nn(w, value)
        return a

    # Because triton does not support masked sparse softmax, we will not check its correctness.
    return profile_attention(triton_attention, data, check_results=False)


def profile_sparta_attention(
    config: Dict[str, Any],
    Ns: int,
    Nt: int,
    E: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(Ns, Nt, E, granularity, sparsity, ndim=3, device=device)
    if mask.sum() == 0:
        return 0., 0.

    sparse_attention = SparseAttention(mask=mask)
    sparse_attention.build(config, sample_inputs=[data['query'], data['key'], data['value']])

    return profile_attention(sparse_attention, data)


def profile_dense_attention(
    Ns: int,
    Nt: int,
    E: int,
    granularity: Tuple[int, int],
    sparsity: float,
    device: Any = 'cuda',
):
    data, mask = prepare_data(Ns, Nt, E, granularity, sparsity, ndim=3, device=device)
    if mask.sum() == 0:
        return 0., 0.

    def dense_attention(query, key, value):
        return sparse_multi_head_attention_reference(query, key, value, mask)

    return profile_attention(dense_attention, data)


def load_sparta_config(device: Any = 'cuda'):
    device_name = torch.cuda.get_device_name(device)
    device_cfg_path = os.path.join(WORK_DIR, 'params', f'{device_name}.csv')
    default_cfg_path = os.path.join(WORK_DIR, 'params', 'default.csv')
    if os.path.exists(device_cfg_path):
        return pd.read_csv(device_cfg_path)
    else:
        return pd.read_csv(default_cfg_path)


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
    cols = ['method', 'Ns', 'Nt', 'E', 'granularity', 'sparsity', 'forward', 'backward']
    with open(log_path, 'w') as f:
        f.write(','.join(cols) + '\n')
    sparta_configs = load_sparta_config(device)
    for g in GRANULARITY_LIST:
        for s in SPARSITY_LIST:
            print(f'========== Granularuty: {g} Sparsity: {s} ==========')
            latency: Dict[str, Tuple[int, int]] = {}
            latency['dense'] = profile_dense_attention(Ns, Nt, E, (g, g), s, device)
            config = get_sparta_config(sparta_configs, g, s)
            latency['sparta'] = profile_sparta_attention(config, Ns, Nt, E, (g, g), s, device)
            for block in [16, 32, 64]:
                latency[f'triton-{block}'] = profile_triton_attention(block, Ns, Nt, E, (g, g), s, device)
            with open(log_path, 'a') as f:
                for method, (lat_f, lat_b) in latency.items():
                    f.write(f'{method},{Ns},{Nt},{E},{g},{s},{lat_f},{lat_b}\n')
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
    plt.suptitle(f'{Nt}x{Ns}x{E} Attention Latency')

    fig_path = os.path.join(os.path.dirname(log_path), 'latency.png')
    plt.savefig(fig_path)
    print(f'Figure saved to: {fig_path}')


def main():
    log_path = os.path.join(WORK_DIR, 'latency.csv')
    profile_all(log_path, device='cuda')
    plot_latency(log_path)


if __name__ == '__main__':
    main()
