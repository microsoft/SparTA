# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import math
from typing import List, Tuple, Callable

import torch
import triton
import pytest
import pandas as pd

from sparta.nn import SparseAttention
from sparta.testing import block_mask, profile, sparse_multi_head_attention_reference


BACKWARD = False
FILE_SUFFIX = '-backward' if BACKWARD else '-forward'

DATA_PATH = os.path.join('benchmark', 'tmp', 'attention', 'latency')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
DENSE_FILE = f'dense{FILE_SUFFIX}.csv'
TRITON_FILE = f'triton{FILE_SUFFIX}-v{triton.__version__}.csv'
SPARTA_FILE = f'sparta{FILE_SUFFIX}.csv'
DATA_COLUMNS = [
    'METHOD', 'BATCH_SIZE', 'Ns', 'Nt', 'E',
    'BLOCK_SIZE', 'SPARSITY', 'LATENCY', 'CUDA_LATENCY'
]
for file_name in [DENSE_FILE, TRITON_FILE, SPARTA_FILE]:
    with open(os.path.join(DATA_PATH, file_name), 'w') as f:
        f.write(','.join(DATA_COLUMNS) + '\n')

SPARTA_PARAMS_PATH = os.path.join('benchmark', 'tmp', 'attention', 'params')
SPARTA_PARAMS_FILE = f'sparta-attention-params-2080-4096x3072x768.csv'
SPARTA_PARAMS_FILE = os.path.join(SPARTA_PARAMS_PATH, SPARTA_PARAMS_FILE)
SPARTA_PARAMS = None
if os.path.exists(SPARTA_PARAMS_FILE):
    SPARTA_PARAMS = pd.read_csv(SPARTA_PARAMS_FILE)

BATCH_SIZE, Ns, Nt, E = 1, 4096, 3072, 768
SHAPE = f'{BATCH_SIZE},{Ns},{Nt},{E}'
BLOCK_SIZE = [1, 8, 32]
SPARSITY = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

device = torch.device(f'cuda:0')


def profile_func(
    method: str, func_call: Callable, targets: List[torch.Tensor],
    block_size: int, sparsity: float
):
    latency = profile(func_call, inputs=[], target_outputs=targets, num_warmups=100, num_iters=100)
    cuda_latency = profile(func_call, inputs=[], cuda=True)
    if method.startswith('tri'):
        file_name = TRITON_FILE
    elif method.startswith('sparta'):
        file_name = SPARTA_FILE
    else:
        file_name = DENSE_FILE
    with open(os.path.join(DATA_PATH, file_name), 'a') as f:
        f.write(f'{method},{SHAPE},{block_size},{sparsity},{latency},{cuda_latency}\n')


def prepare_data(block: Tuple[int, int], sparsity: float, seed: int = 2022):
    torch.manual_seed(seed)
    mask = block_mask((Nt, Ns), block=block, sparsity=sparsity).cuda()
    query = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()
    key = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    value = torch.rand(size=(BATCH_SIZE, Ns, E)).cuda()
    grad_out = torch.rand(size=(BATCH_SIZE, Nt, E)).cuda()
    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True
    out = sparse_multi_head_attention_reference(query, key, value, mask)
    out.backward(grad_out)
    grad_query = query.grad
    grad_key = key.grad
    grad_value = value.grad
    query.grad = None
    key.grad = None
    value.grad = None
    return mask, (query, key, value, out), (grad_query, grad_key, grad_value, grad_out)


def triton_attention(
    layout,
    block: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = 1 / math.sqrt(E),
):
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=device)
    sparse_dot_dsd_nn = triton.ops.blocksparse.matmul(layout, block, "dsd", trans_a=False, trans_b=False, device=device)
    sparse_softmax = triton.ops.blocksparse.softmax(layout, block, device=device)

    w = sparse_dot_sdd_nt(query, key)
    w = sparse_softmax(w, scale=scale, is_causal=True)
    a = sparse_dot_dsd_nn(w, value)
    return a


@pytest.mark.parametrize("tri_block_size", [16, 32, 64])
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("sparsity", SPARSITY)
def test_triton(tri_block_size: int, block_size: int, sparsity: float):
    mask, tensors, grads = prepare_data((block_size, block_size), sparsity)
    # query, key, value, out = tensors
    # grad_query, grad_key, grad_value, grad_out = grads
    query, key, value, out = [x.unsqueeze(0).detach() for x in tensors]
    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True
    grad_query, grad_key, grad_value, grad_out = [x.unsqueeze(0) for x in grads]

    # layout = torch.tril(torch.ones([BATCH_SIZE, n_blocks, n_blocks], dtype=torch.long))
    layout = mask.reshape(Nt // tri_block_size, tri_block_size, Ns // tri_block_size, tri_block_size)
    layout = layout.swapaxes(1, 2).any(-1).any(-1).unsqueeze(0).to(torch.int32).cpu()

    if BACKWARD:
        def func_call():
            out_hat = triton_attention(layout, tri_block_size, query, key, value)
            out_hat.backward(grad_out)
            grad_query_hat = query.grad
            grad_key_hat = key.grad
            grad_value_hat = value.grad
            return out_hat, grad_query_hat, grad_key_hat, grad_value_hat
        targets = [out, grad_query, grad_key, grad_value]
    else:
        def func_call():
            out_hat = triton_attention(layout, tri_block_size, query, key, value)
            return out_hat
        targets = [out]

    # profile_func(f'tri_{tri_block_size}', func_call, targets, block_size, sparsity)
    profile_func(f'tri_{tri_block_size}', func_call, None, block_size, sparsity)


@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("sparsity", SPARSITY)
def test_sparta(block_size: int, sparsity: float):
    mask, tensors, grads = prepare_data((block_size, block_size), sparsity)
    query, key, value, out = tensors
    grad_query, grad_key, grad_value, grad_out = grads
    if mask.sum() == 0:
        return

    sparse_attention = SparseAttention(mask=mask)
    filt = (SPARTA_PARAMS['BLOCK_SIZE'] == block_size) & (SPARTA_PARAMS['SPARSITY'] == sparsity)
    params = {}
    for param_name, param_val in SPARTA_PARAMS[filt].iloc[0, 2:].to_dict().items():
        kernel_name, param_name = param_name.split(';')
        if kernel_name not in params:
            params[kernel_name] = {}
        if param_name.startswith('THREAD'):
            if params[kernel_name]['_impl'] == 'openai':
                continue
        if type(param_val) is not str:
            param_val = int(param_val)
        params[kernel_name][param_name] = param_val
    sparse_attention.build(params, sample_inputs=[query, key, value])

    if BACKWARD:
        def func_call():
            out_hat = sparse_attention.forward(query, key, value)
            out_hat.backward(grad_out)
            return out_hat, query.grad, key.grad, value.grad
        targets = [out, grad_query, grad_key, grad_value]
    else:
        def func_call():
            out_hat = sparse_attention.forward(query, key, value)
            return out_hat
        targets = [out]

    profile_func('sparta', func_call, targets, block_size, sparsity)


@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("sparsity", SPARSITY)
def test_dense(block_size: int, sparsity: float):
    mask, tensors, grads = prepare_data((block_size, block_size), sparsity)
    query, key, value, out = tensors
    grad_query, grad_key, grad_value, grad_out = grads

    if BACKWARD:
        def func_call():
            out_hat = sparse_multi_head_attention_reference(query, key, value, mask)
            out_hat.backward(grad_out)
            return out_hat, query.grad, key.grad, value.grad
        targets = [out, grad_query, grad_key, grad_value]
    else:
        def func_call():
            out_hat = sparse_multi_head_attention_reference(query, key, value, mask)
            return out_hat
        targets = [out]

    profile_func('dense', func_call, targets, block_size, sparsity)
