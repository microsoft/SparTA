# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

import sparta
from sparta.testing import block_mask


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


def test_sparse_linear_sdd(params: dict, has_bias: bool):
    print(f'---------- Sparse Linear SDD ----------')
    print(f'params={params}, bias={has_bias}')
    torch.manual_seed(2022)
    A = torch.rand((M,K), dtype=torch.float32).cuda()
    B = torch.rand((N,K), dtype=torch.float32).cuda()
    bias = torch.rand((N,),dtype=torch.float32).cuda()
    # generate and apply mask
    A_mask = block_mask(A.shape, block=BLOCK, sparsity=SPARSITY).cuda()
    A = torch.mul(A, A_mask)
    # spase and dense operators
    linear = torch.nn.Linear(K, N, bias=has_bias)
    linear.load_state_dict(dict(weight=B, bias=bias) if has_bias else dict(weight=B) )
    linear.cuda()
    splinear = sparta.nn.SparseLinear(linear, input_mask=A_mask)
    splinear.build(params, sample_inputs=[A])
    torch.testing.assert_close(splinear(A), linear(A))
    print('PASS')

def test_sparse_linear_dsd(params, has_bias: bool):
    print(f'---------- Sparse Linear DSD ----------')
    print(f'params={params}, bias={has_bias}')
    torch.manual_seed(2022)
    A = torch.rand((M,K), dtype=torch.float32).cuda()
    B = torch.rand((N,K), dtype=torch.float32).cuda()
    bias = torch.rand((N,),dtype=torch.float32).cuda()
    # generate and apply mask
    B_mask = block_mask(B.shape, block=BLOCK, sparsity=SPARSITY).cuda()
    B = torch.mul(B, B_mask)
    # spase and dense operators
    linear = torch.nn.Linear(K, N, bias=has_bias)
    linear.load_state_dict(dict(weight=B, bias=bias) if has_bias else dict(weight=B) )
    linear.cuda()
    splinear = sparta.nn.SparseLinear(linear, weight_mask=B_mask)
    splinear.build(params, sample_inputs=[A])
    torch.testing.assert_close(splinear(A), linear(A))
    print('PASS')

def test_sparse_linear_dds(params: dict, has_bias: bool):
    print(f'---------- Sparse Linear DDS ----------')
    print(f'params={params}, bias={has_bias}')
    torch.manual_seed(2022)
    A = torch.rand((M,K), dtype=torch.float32).cuda()
    B = torch.rand((N,K), dtype=torch.float32).cuda()
    bias = torch.rand((N,),dtype=torch.float32).cuda()
    # generate and apply mask
    C_mask = block_mask((M,N), block=BLOCK, sparsity=SPARSITY).cuda()
    # spase and dense operators
    linear = torch.nn.Linear(K, N, bias=has_bias)
    linear.load_state_dict(dict(weight=B, bias=bias) if has_bias else dict(weight=B) )
    linear.cuda()
    splinear = sparta.nn.SparseLinear(linear, output_mask=C_mask)
    splinear.build(params, sample_inputs=[A])
    torch.testing.assert_close(torch.mul(splinear(A), C_mask), torch.mul(linear(A), C_mask))
    print('PASS')


class TestSparseLinearOperators(unittest.TestCase):

    def test_sparse_linear(self):
        print('==================== Testing Sparse Linear Operators ====================')
        test_sparse_linear_dsd(dict(_name='sparta', **TILE_CONFIG), True)
        test_sparse_linear_dsd(dict(_name='sparta', **TILE_CONFIG), False)
        test_sparse_linear_dsd(dict(_name='openai'), True)
        test_sparse_linear_dsd(dict(_name='openai'), False)

        test_sparse_linear_sdd(dict(_name='sparta', **TILE_CONFIG), True)
        test_sparse_linear_sdd(dict(_name='sparta', **TILE_CONFIG), False)
        test_sparse_linear_sdd(dict(_name='openai'), True)
        test_sparse_linear_sdd(dict(_name='openai'), False)

        test_sparse_linear_dds(dict(_name='openai'), True)
        test_sparse_linear_dds(dict(_name='openai'), False)


if __name__ == '__main__':
    unittest.main()
