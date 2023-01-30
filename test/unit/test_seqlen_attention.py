# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.nn import SeqlenDynamicSparseAttention
from sparta.testing import sparse_multi_head_attention_reference


def random_seqlens(B: int, N: int):
    seqlens = torch.randint(1, N, size=(B, ), dtype=torch.int32, device='cuda')
    mask = torch.zeros(size=(B, 1, N, N), dtype=torch.int32, device='cuda')
    for batch in range(B):
        mask[batch, 0, :seqlens[batch], :seqlens[batch]] = 1
    return seqlens, mask


@pytest.mark.parametrize("B", [12])
@pytest.mark.parametrize("H", [20])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("E", [64])
@pytest.mark.parametrize("global_mode", [False, True])
def test_seqlen_attention_operator(B: int, H: int, N: int, E: int, global_mode: bool):
    seqlen_attention = SeqlenDynamicSparseAttention(global_mode)

    if global_mode:
        torch.manual_seed(2022)
        seqlens, mask = random_seqlens(B, N)
        SeqlenDynamicSparseAttention.set_global_seqlens(seqlens)

    for random_seed in range(3):  # Test dynamic sparse
        print(random_seed)
        torch.manual_seed(random_seed)
        if not global_mode:
            seqlens, mask = random_seqlens(B, N)
        query = torch.rand(size=(B, H, N, E), dtype=torch.float32, device='cuda')
        key = torch.rand(size=(B, H, N, E), dtype=torch.float32, device='cuda')
        value = torch.rand(size=(B, H, N, E), dtype=torch.float32, device='cuda')

        target_out = sparse_multi_head_attention_reference(
            query=query.reshape((-1, N, E)),
            key=key.reshape((-1, N, E)),
            value=value.reshape((-1, N, E)),
            mask=mask.tile((1, H, 1, 1)).reshape(-1, N, N),
            temperature=1.0,
        ).reshape((B, H, N, E))

        if global_mode:
            out = seqlen_attention.forward(query, key, value)
        else:
            out = seqlen_attention.forward(query, key, value, seqlens)

        torch.testing.assert_close(out, target_out, atol=1e-4, rtol=1e-8)
