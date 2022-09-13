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
    'BLOCK_SIZE_H_VALUE': 16,
    'BLOCK_SIZE_W_VALUE': 64,
    'ROW_TILE_VALUE': 4,
}


class TestSparseSoftmaxOperators(unittest.TestCase):

    def test_sparse_softmax(self):
        print('==================== Testing Sparse Softmax Operators ====================')
        print('---------- Sparse Softmax (Implementation: sparta) ----------')
        torch.manual_seed(2022)
        mask = sparta.testing.block_mask(shape=(H, W), block=BLOCK, sparsity=SPARSITY).cuda()
        dense_input = torch.rand((H, W)).cuda()
        dense_op = torch.nn.Softmax(dim=-1).cuda()
        sparse_op = sparta.nn.SparseSoftmax(dense_op, mask=mask)
        sparse_op.build(dict(_name='sparta', **TILE_CONFIG), sample_inputs=[dense_input])
        torch.testing.assert_close(
            sparse_op(dense_input),
            sparta.testing.sparse_softmax_reference(dense_input, mask)
        )
        print('PASS')


if __name__ == '__main__':
    unittest.main()
