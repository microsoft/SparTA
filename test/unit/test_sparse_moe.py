# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
import pytest

from sparta.nn import DynamicSparseMoE


def check_dtype(dtype: torch.dtype):
    major, minor = torch.cuda.get_device_capability()
    return major >= 7 or dtype is torch.float32


def moe_reference(
    exp_modules: List[torch.nn.Linear],
    data: torch.Tensor,
    exp_ids: torch.Tensor,
    out_dims: int,
):
    n_exp = len(exp_modules)
    out = torch.zeros(data.size(0), out_dims).to(data.device)
    for eid in range(n_exp):
        out[exp_ids == eid] = exp_modules[eid](data[exp_ids == eid])
    return out


@pytest.mark.parametrize("batch", [32])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("num_exps", [8])
@pytest.mark.parametrize("in_dims", [3072])
@pytest.mark.parametrize("out_dims", [768])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sparse_moe(
    batch: int,
    seq_len: int,
    num_exps: int,
    in_dims: int,
    out_dims: int,
    dtype: torch.dtype,
):
    if not check_dtype(dtype):
        return

    torch.manual_seed(2022)
    exp_modules = [
        torch.nn.Linear(in_dims, out_dims, bias=False, dtype=dtype, device='cuda')
        for _ in range(num_exps)
    ]
    moe = DynamicSparseMoE(exp_modules)

    for random_seed in range(3):  # Test dynamic sparse
        torch.manual_seed(random_seed)
        data = torch.rand(batch * seq_len, in_dims, dtype=dtype, device='cuda')
        exp_ids = torch.randint(0, num_exps, (batch * seq_len, ), dtype=torch.int32, device='cuda')

        out = moe(data, exp_ids)
        target_out = moe_reference(exp_modules, data, exp_ids, out_dims)

        assert torch.allclose(out, target_out, atol=1e-4, rtol=1e-8)
