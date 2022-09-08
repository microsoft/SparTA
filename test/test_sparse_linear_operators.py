# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

import sparta


M, K, N = 1024, 256, 512
SPARSITY = 0.8
BLOCK = (8, 8)
SHAPE_CONFIG = {
    'GLOBAL_M_VALUE': M,
    'GLOBAL_K_VALUE': K,
    'GLOBAL_N_VALUE': N,
}
TILE_CONFIG = {
    'BLOCK_SIZE_M_VALUE': 16,
    'BLOCK_SIZE_K_VALUE': 16,
    'BLOCK_SIZE_N_VALUE': 16,
    'THREAD_SIZE_M_VALUE': 4,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 4,
}


def test_sparse_linear_sdd(impl: str):
    print(f'---------- Sparse Linear SDD (Implementation: {impl}) ----------')
    torch.manual_seed(2022)
    input_mask = sparta.testing.block_mask(shape=(M, K), block=BLOCK, sparsity=SPARSITY).cuda()
    sparse_input = torch.rand((M, K)).cuda() * input_mask
    dense_op = torch.nn.Linear(K, N).cuda()
    sparse_op = sparta.nn.SparseLinear(dense_op, input_mask=input_mask)
    sparse_op.build(impl, dict(SHAPE_CONFIG, **TILE_CONFIG) if impl == 'sparta' else SHAPE_CONFIG)
    torch.testing.assert_close(sparse_op(sparse_input), dense_op(sparse_input))
    print('PASS')

def test_sparse_linear_dsd(impl: str):
    print(f'---------- Sparse Linear DSD (Implementation: {impl}) ----------')
    torch.manual_seed(2022)
    weight_mask = sparta.testing.block_mask(shape=(N, K), block=BLOCK, sparsity=SPARSITY).cuda()
    dense_input = torch.rand((M, K)).cuda()
    dense_op = torch.nn.Linear(K, N).cuda()
    dense_op.weight = torch.nn.Parameter(dense_op.weight.detach() * weight_mask)
    sparse_op = sparta.nn.SparseLinear(dense_op, weight_mask=weight_mask)
    sparse_op.build(impl, dict(SHAPE_CONFIG, **TILE_CONFIG) if impl == 'sparta' else SHAPE_CONFIG)
    torch.testing.assert_close(sparse_op(dense_input), dense_op(dense_input))
    print('PASS')

def test_sparse_linear_dds(impl: str):
    print(f'---------- Sparse Linear DDS (Implementation: {impl}) ----------')
    torch.manual_seed(2022)
    output_mask = sparta.testing.block_mask(shape=(M, N), block=BLOCK, sparsity=SPARSITY).cuda()
    dense_input = torch.rand((M, K)).cuda()
    dense_op = torch.nn.Linear(K, N).cuda()
    sparse_op = sparta.nn.SparseLinear(dense_op, output_mask=output_mask)
    sparse_op.build(impl, dict(SHAPE_CONFIG, **TILE_CONFIG) if impl == 'sparta' else SHAPE_CONFIG)
    torch.testing.assert_close(
        sparse_op(dense_input) * output_mask,
        dense_op(dense_input) * output_mask
    )
    print('PASS')


class TestSparseLinearOperators(unittest.TestCase):

    def test_sparse_linear(self):
        print('==================== Testing Sparse Linear Operators ====================')
        test_sparse_linear_sdd('sparta')
        test_sparse_linear_dsd('sparta')
        test_sparse_linear_sdd('openai')
        test_sparse_linear_dsd('openai')
        test_sparse_linear_dds('openai')


if __name__ == '__main__':
    unittest.main()
