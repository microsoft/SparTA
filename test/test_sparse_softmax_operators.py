# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

import sparta


H, W = 1024, 512
SPARSITY = 0.95
BLOCK = (8, 8)
SHAPE_CONFIG = {
    'GLOBAL_H_VALUE': H,
    'GLOBAL_W_VALUE': W,
}
TILE_CONFIG = {
    'BLOCK_SIZE_H_VALUE': 32,
    'BLOCK_SIZE_W_VALUE': 32,
    'ROW_TILE_VALUE': 4,
}


def sparse_matmul_reference(dense_input: torch.Tensor, mask: torch.Tensor):
    C_max = dense_input.max(axis=-1).values.reshape((-1, 1))
    C_exp = torch.exp(dense_input - C_max) * mask
    C_exp_sum = C_exp.sum(axis=-1).reshape((-1, 1)) + 1e-10
    return C_exp / C_exp_sum


class TestSparseSoftmaxOperators(unittest.TestCase):

    def test_sparse_softmax(self):
        print('==================== Testing Sparse Softmax Operators ====================')
        print('---------- Sparse Softmax (Implementation: sparta) ----------')
        torch.manual_seed(2022)
        mask = sparta.testing.block_mask(shape=(H, W), block=BLOCK, sparsity=SPARSITY).cuda()
        dense_input = torch.rand((H, W)).cuda()
        dense_op = torch.nn.Softmax(dim=-1).cuda()
        sparse_op = sparta.nn.SparseSoftmax(dense_op, mask=mask)
        sparse_op.build(dict(implement='sparta', config=TILE_CONFIG))
        torch.testing.assert_close(
            sparse_op(dense_input),
            sparse_matmul_reference(dense_input, mask)
        )
        print('PASS')


if __name__ == '__main__':
    unittest.main()
