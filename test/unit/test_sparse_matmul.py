# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Type, Optional

import torch
import pytest

from sparta.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.operators import SparseLinear, SparseMatMul, SparseBatchMatMul
from sparta.tesa import BCSIndexes
from sparta.testing import block_mask


def prepare_data(
    batch: Optional[int] = 4,
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
    mask: Optional[torch.Tensor] = None,
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


def get_params(impl: str):
    if impl == 'sparta':
        return {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': 32,
            'BLOCK_SIZE_K_VALUE': 32,
            'BLOCK_SIZE_N_VALUE': 32,
        }
    else:
        return {'_impl': impl}


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


@pytest.mark.parametrize("impl", ['sparta', 'openai'])
@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("batch", [None, 4])
def test_sparse_matmul_kernel(
    impl: str,
    mode: str,
    biased: bool,
    compressed: bool,
    trans_A: bool,
    trans_B: bool,
    batch: Optional[int],
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(
        batch, M, K, N,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        False,
    )

    kernelClass: Type[SparseMatMulKernel] = {
        'sparta': SparTASparseMatMulKernel,
        'openai': OpenAISparseMatMulKernel,
    }[impl]
    batched = batch is not None
    kernel = kernelClass(
        mode=mode,
        biased=biased,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
        batched=batched,
    )
    kernel.attr.set_mask(mask)
    batch = 1 if batch is None else batch
    kernel.compile(get_params(impl), (batch, M, K, N))

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]

    if compressed:
        data, mask = compress_data(kernel.attr.indexes, sparse_port, data, mask, False)

    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    input_data = [data[f'input_{x}'] for x in inputs]

    data['output_C'] = kernel(*input_data)
    add_mask(data, mask, sparse_port, 'output')
    check_results(data)


@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("batch", [None, 4])
def test_sparse_matmul_operator(
    mode: str,
    biased: bool,
    compressed: bool,
    trans_A: bool,
    trans_B: bool,
    batch: Optional[int],
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(
        batch, M, K, N,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        True,
    )

    if batch is None:
        sparse_matmul = SparseMatMul(mode, trans_A, trans_B, biased, compressed)
    else:
        sparse_matmul = SparseBatchMatMul(mode, trans_A, trans_B, biased, compressed)
    sparse_matmul.set_mask(mask)

    kernel_names = ['forward', 'backward:A', 'backward:B']
    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    sparse_matmul.build(
        config={kernel_name: get_params('sparta') for kernel_name in kernel_names},
        sample_inputs=[data[f'input_{x}'] for x in inputs]
    )

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]

    def run_test():
        nonlocal sparse_matmul, data, mask
        if compressed:
            indexes = sparse_matmul.get_sparse_indexes()
            data, cmask = compress_data(indexes, sparse_port, data, mask, True)
        else:
            cmask = mask
        data['output_C'] = sparse_matmul(*[data[f'input_{x}'] for x in inputs])
        data['output_C'].backward(data['input_grad_C'])
        for x in inputs:
            data[f'output_grad_{x}'] = data[f'input_{x}'].grad
        add_mask(data, cmask, sparse_port, 'output')
        check_results(data)

    run_test()

    # Dynamic mask
    data, mask = prepare_data(
        batch, M, K, N,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        True, random_seed=2023,
    )
    sparse_matmul.set_mask(mask)
    run_test()

    # Dynamic Dim
    if sparse_port == 'A':
        N = 1024
    elif sparse_port == 'B':
        M = 1024
    elif sparse_port == 'C':
        K = 1024
    data, mask = prepare_data(
        batch, M, K, N,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        True, mask,
    )
    run_test()


@pytest.mark.parametrize('mode', ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize('biased', [False, True])
def test_sparse_linear_operator(
    mode: str,
    biased: bool,
    batch: int = 128,
    in_dims: int = 256,
    out_dims: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, mask = prepare_data(
        None, batch, in_dims, out_dims,
        granularity, sparsity,
        mode, False, True, biased,
        True,
    )

    dense_linear = torch.nn.Linear(in_dims, out_dims, bias=biased, device='cuda')
    if biased:
        dense_linear.load_state_dict({'weight': data['input_B'], 'bias': data['input_bias']})
    else:
        dense_linear.load_state_dict({'weight': data['input_B']})

    sparse_linear = SparseLinear(dense_linear, mode)
    sparse_linear.set_mask(mask)

    kernel_names = ['forward', 'backward:A', 'backward:B']
    sparse_linear.build(
        config={kernel_name: get_params('sparta') for kernel_name in kernel_names},
        sample_inputs=[data['input_A']],
    )

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]

    def run_test():
        nonlocal sparse_linear, data, mask
        if mode == 'dsd':
            indexes = sparse_linear.get_sparse_indexes()
            data, cmask = compress_data(indexes, sparse_port, data, mask, True)
        else:
            cmask = mask
        data['output_C'] = sparse_linear(data['input_A'])
        data['output_C'].backward(data['input_grad_C'])
        data[f'output_grad_A'] = data[f'input_A'].grad
        data[f'output_grad_B'] = sparse_linear.weight.grad
        if biased:
            data[f'output_grad_bias'] = sparse_linear.bias.grad
        add_mask(data, cmask, sparse_port, 'output')
        check_results(data)

    run_test()

    # Dynamic mask
    data, mask = prepare_data(
        None, batch, in_dims, out_dims,
        granularity, sparsity,
        mode, False, True, biased,
        True, random_seed=2023
    )
    sparse_linear.ports['B'].sample_data = data['input_B']
    if biased:
        sparse_linear.bias = torch.nn.Parameter(data['input_bias'])
    sparse_linear.set_mask(mask)
    run_test()

    # Dynamic Dim
    if sparse_port == 'A':
        N = 1024
    elif sparse_port == 'B':
        M = 1024
    elif sparse_port == 'C':
        K = 1024
    data, mask = prepare_data(
        None, batch, in_dims, out_dims,
        granularity, sparsity,
        mode, False, True, biased,
        True, mask,
    )
    weight = data['input_B']
    if mode == 'dsd':
        weight = sparse_linear.get_sparse_indexes().convert(weight.detach())
    sparse_linear.weight = torch.nn.Parameter(weight)
    if biased:
        sparse_linear.bias = torch.nn.Parameter(data['input_bias'])
    run_test()
