# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Type, Optional

import torch
import pytest

from sparta.specializer.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.specializer.functional import SparsityAttr, SparseMatMul, SparseBatchMatMul
# from sparta.nn import SparseBatchMatMul, SparseLinear
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
    mask = block_mask(shapes[sparse_port], block=granularity, sparsity=sparsity, device='cuda')
    add_mask(data, mask, sparse_port, 'input')

    if requires_grad:
        for x in inputs:
            data[f'input_{x}'].requires_grad = True

    if batch is None:
        input_A = data['input_A'].T if trans_A else data['input_A']
        input_B = data['input_B'].T if trans_B else data['input_B']
        data['target_C'] = torch.mm(input_A, input_B)
        if biased:
            data['target_C'] += data['input_bias']
    else:
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

    add_mask(data, mask, sparse_port, 'target')

    return data, mask


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
            torch.testing.assert_close(val, data[name.replace('target', 'output')], msg=name)


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
    data, mask = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, biased, False)

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
    shape = (batch, M, K, N)
    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    BCSR = {
        'A': not trans_A,
        'B': trans_B,
        'C': True,
    }[sparse_port]
    BCSC = not BCSR
    attr = SparsityAttr(BCSR, BCSC)
    attr.set_mask(mask)
    kernel.set_parameter('BCSR', BCSR)
    kernel.set_parameter('BCSC', BCSC)
    kernel.compile(get_params(impl), shape, attr)

    sparse_axis = {
        'A': ['K', 'M'] if trans_A else ['M', 'K'],
        'B': ['N', 'K'] if trans_B else ['K', 'N'],
        'C': ['M', 'N'],
    }[sparse_port]
    attr.BCSR = BCSR
    attr.BCSC = BCSC
    attr.set_block_size(*[
        kernel.get_parameter(f'BLOCK_SIZE_{i}_VALUE')
        for i in sparse_axis
    ])

    if compressed:
        data, mask = compress_data(attr.indexes, sparse_port, data, mask, False)

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
def test_sparse_matmul_function(
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
    data, mask = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, biased, True)

    if batch is None:
        func = SparseMatMul(mode, trans_A, trans_B, biased, compressed)
    else:
        func = SparseBatchMatMul(mode, trans_A, trans_B, biased, compressed)

    sparse_attr = func.get_sparse_attr()
    sparse_attr.set_mask(mask)

    kernel_names = ['forward', 'backward:A', 'backward:B']
    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    func.build(
        config={kernel_name: get_params('sparta') for kernel_name in kernel_names},
        sample_inputs=[data[f'input_{x}'] for x in inputs]
    )

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    if compressed:
        data, mask = compress_data(sparse_attr.indexes, sparse_port, data, mask, True)

    data['output_C'] = func(*[data[f'input_{x}'] for x in inputs])
    data['output_C'].backward(data['input_grad_C'])
    for x in inputs:
        data[f'output_grad_{x}'] = data[f'input_{x}'].grad
    add_mask(data, mask, sparse_port, 'output')
    check_results(data)


# @pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
# @pytest.mark.parametrize("trans_A", [False, True])
# @pytest.mark.parametrize("trans_B", [False, True])
# @pytest.mark.parametrize("compressed", [False, True])
# def test_sparse_matmul_operator(
#     mode: str,
#     compressed: bool,
#     trans_A: bool,
#     trans_B: bool,
#     batch: int = 4,
#     M: int = 128,
#     K: int = 256,
#     N: int = 192,
#     granularity: Tuple[int, int] = (8, 8),
#     sparsity: float = 0.9,
# ):
#     data, masks = prepare_data(
#         batch, M, K, N, granularity, sparsity,
#         mode, trans_A, trans_B, False, True,
#     )

#     sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
#     sparse_matmul = SparseBatchMatMul(
#         **{f'{name}_mask': val for name, val in masks.items()},
#         transpose_A=trans_A,
#         transpose_B=trans_B,
#         compressed=compressed,
#     )
#     sparse_matmul.build(
#         config={
#             kernel_name: get_params('sparta')
#             for kernel_name in sparse_matmul.get_kernel_placeholders(backward=True)
#         },
#         sample_inputs=[data['input_A'], data['input_B']],
#     )

#     for random_seed in range(3):  # Test dynamic sparse
#         if compressed:
#             compress_data(sparse_matmul.get_sparse_indexes(sparse_port), sparse_port, data, masks)

#         data['output_C'] = sparse_matmul.forward(data['input_A'], data['input_B'])
#         data['output_C'].backward(data['input_grad_C'])
#         for x in ['A', 'B']:
#             data[f'output_grad_{x}'] = data[f'input_{x}'].grad

#         add_mask(data, masks, sparse_port, 'output')

#         check_results(data)

#         data, masks = prepare_data(
#             batch, M, K, N, granularity, sparsity,
#             mode, trans_A, trans_B, False, True, random_seed,
#         )
#         sparse_matmul.update_mask(**{f'{name}_mask': val for name, val in masks.items()})


# @pytest.mark.parametrize('mode', ['sdd', 'dsd', 'dds'])
# @pytest.mark.parametrize('biased', [False, True])
# def test_sparse_linear_operator(
#     mode: str,
#     biased: bool,
#     batch: int = 128,
#     in_dims: int = 256,
#     out_dims: int = 192,
#     granularity: Tuple[int, int] = (8, 8),
#     sparsity: float = 0.9,
# ):
#     data, masks = prepare_data(
#         -1, batch, in_dims, out_dims, granularity, sparsity,
#         mode, False, True, biased, True,
#     )

#     dense_linear = torch.nn.Linear(in_dims, out_dims, bias=biased, device='cuda')
#     if biased:
#         dense_linear.load_state_dict({'weight': data['input_B'], 'bias': data['input_bias']})
#     else:
#         dense_linear.load_state_dict({'weight': data['input_B']})

#     sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
#     mask_name = {'sdd': 'input_mask', 'dsd': 'weight_mask', 'dds': 'output_mask'}[mode]
#     sparse_linear = SparseLinear(dense_linear, **{mask_name: masks[sparse_port]})
#     sparse_linear.build(
#         config={
#             kernel_name: get_params('sparta')
#             for kernel_name in sparse_linear.get_kernel_placeholders(backward=True)
#         },
#         sample_inputs=[data['input_A']],
#     )

#     for random_seed in range(3):  # Test dynamic sparse
#         if mode == 'dsd':
#             compress_data(sparse_linear.get_sparse_indexes('B'), 'B', data, masks)

#         data['output_C'] = sparse_linear.forward(data['input_A'])
#         data['output_C'].backward(data['input_grad_C'])
#         data[f'output_grad_A'] = data[f'input_A'].grad
#         data[f'output_grad_B'] = sparse_linear.weight.grad
#         if biased:
#             data[f'output_grad_bias'] = sparse_linear.bias.grad

#         add_mask(data, masks, sparse_port, 'output')

#         check_results(data)

#         data, masks = prepare_data(
#             -1, batch, in_dims, out_dims, granularity, sparsity,
#             mode, False, True, biased, True, random_seed,
#         )
#         if biased:
#             sparse_linear.bias = torch.nn.Parameter(data['input_bias'])
#         sparse_linear._raw_weight = data['input_B']
#         sparse_linear.update_mask(**{mask_name: masks[sparse_port]})
