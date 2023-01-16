# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Type

import torch
import pytest

from sparta.specializer.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.specializer.funtional import SparseBatchMatMulCtx, SparseBatchMatMulFunc
from sparta.nn import SparseBatchMatMul, SparseLinear
from sparta.tesa import BCSIndexes
from sparta.testing import block_mask


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


def get_params(impl: str):
    if impl == 'sparta':
        return {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': 32,
            'BLOCK_SIZE_K_VALUE': 32,
            'BLOCK_SIZE_N_VALUE': 32,
            'THREAD_SIZE_M_VALUE': 4,
            'THREAD_SIZE_K_VALUE': 4,
            'THREAD_SIZE_N_VALUE': 4,
        }
    else:
        return {'_impl': impl}


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


@pytest.mark.parametrize("impl", ['sparta', 'openai'])
@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_kernel(
    impl: str,
    mode: str,
    biased: bool,
    compressed: bool,
    trans_A: bool,
    trans_B: bool,
    batch: int = 4,
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, masks = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, biased, False)

    kernelClass: Type[SparseMatMulKernel] = {
        'sparta': SparTASparseMatMulKernel,
        'openai': OpenAISparseMatMulKernel,
    }[impl]
    kernel = kernelClass(
        mode=mode,
        biased=biased,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
    )

    for sparse_port, mask in masks.items():
        kernel.ports[sparse_port].set_mask(mask)
    kernel.set_shape(batch, M, K, N)
    kernel.compile(get_params(impl))

    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    input_data = [data[f'input_{x}'] for x in inputs]
    kernel.test(input_data, num_warmups=0, num_iters=1, cuda=False)


@pytest.mark.parametrize("mode", ['sdd', 'dsd', 'dds'])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_sparse_matmul_function(
    mode: str,
    biased: bool,
    compressed: bool,
    trans_A: bool,
    trans_B: bool,
    batch: int = 4,
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
):
    data, masks = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, biased, True)

    sparse_ctx = SparseBatchMatMulCtx(mode, trans_A, trans_B, biased, compressed)
    kernel_names = sparse_ctx.get_kernel_placeholders(backward=True).keys()
    sparse_ctx.select_impls({
        kernel_name: 'sparta'
        for kernel_name in kernel_names
    })
    for port_name, ports in sparse_ctx.sparse_ports.items():
        for port in ports:
            port.set_mask(masks[port_name])
    sparse_ctx.set_shape(batch, M, K, N)
    sparse_ctx.build({
        kernel_name: get_params('sparta')
        for kernel_name in kernel_names
    })

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    if compressed:
        compress_data(sparse_ctx.sparse_ports[sparse_port][0].indexes, sparse_port, data, masks)

    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    input_data = [data[f'input_{x}'] for x in inputs]
    data['output_C'] = SparseBatchMatMulFunc.apply(sparse_ctx, *input_data)
    data['output_C'].backward(data['input_grad_C'])
    for x in inputs:
        data[f'output_grad_{x}'] = data[f'input_{x}'].grad

    add_mask(data, masks, sparse_port, 'output')

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
#     data, masks = prepare_data(batch, M, K, N, granularity, sparsity, mode, trans_A, trans_B, False, True)

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

#     sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
#     if compressed:
#         compress_data(sparse_matmul.get_converter('forward:C', sparse_port), sparse_port, data, masks)

#     data['output_C'] = sparse_matmul.forward(data['input_A'], data['input_B'])
#     data['output_C'].backward(data['input_grad_C'])
#     for x in ['A', 'B']:
#         data[f'output_grad_{x}'] = data[f'input_{x}'].grad

#     add_mask(data, masks, sparse_port, 'output')

#     check_results(data)


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
#     data, masks = prepare_data(1, batch, in_dims, out_dims, granularity, sparsity, mode, False, True, biased, True)

#     for name, val in data.items():
#         if val.requires_grad:
#             val = val.detach().squeeze(0)
#             val.requires_grad = True
#         else:
#             val = val.squeeze(0)
#         data[name] = val

#     dense_linear = torch.nn.Linear(in_dims, out_dims, bias=biased, device='cuda')
#     if biased:
#         dense_linear.load_state_dict({'weight': data['input_B'], 'bias': data['input_bias']})
#     else:
#         dense_linear.load_state_dict({'weight': data['input_B']})

#     sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
#     mask_name = {'sdd': 'input_mask', 'dsd': 'weight_mask', 'dds': 'output_mask'}[mode]
#     sparse_linear = SparseLinear(dense_linear, **{mask_name: masks[sparse_port]})
#     sparse_linear.build(
#         params={
#             kernel_name: get_params('sparta')
#             for kernel_name in sparse_linear.get_kernel_placeholders(backward=True)
#         },
#         sample_inputs=[data['input_A']],
#     )

#     if mode == 'dsd':
#         compress_data(sparse_linear.get_converter('forward:C', 'B'), 'B', data, masks)

#     data['output_C'] = sparse_linear.forward(data['input_A'])
#     data['output_C'].backward(data['input_grad_C'])
#     data[f'output_grad_A'] = data[f'input_A'].grad
#     data[f'output_grad_B'] = sparse_linear.weight.grad
#     if biased:
#         data[f'output_grad_bias'] = sparse_linear.bias.grad

#     add_mask(data, masks, sparse_port, 'output')

#     check_results(data)
