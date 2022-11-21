# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import pytest

from sparta.nn import SparseLinear
from sparta.testing import block_mask
from sparta.nn import tune


BATCH_SIZE, IN_DIMS, OUT_DIMS = 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


def test_grid_search_tuner():
    dense_linear = torch.nn.Linear(IN_DIMS, OUT_DIMS, bias=True).cuda()

    torch.manual_seed(2022)
    sample_input = torch.rand((BATCH_SIZE, IN_DIMS), dtype=torch.float32).cuda()
    sample_grad = torch.rand((BATCH_SIZE, OUT_DIMS), dtype=torch.float32).cuda()

    mask = block_mask((OUT_DIMS, IN_DIMS), block=BLOCK, sparsity=SPARSITY).cuda()

    sparse_linear = SparseLinear(dense_linear, weight_mask=mask)

    tune(sparse_linear, [sample_input], [sample_grad], algo='rand', max_trials=1, backward_weight=1)


if __name__ == '__main__':
    test_grid_search_tuner()
