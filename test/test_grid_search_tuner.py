# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

import sparta


M, K, N = 1024, 1024, 1024
SEARCH_SPACE = {
    'sparta': {
        'BLOCK_SIZE_M_VALUE': [32, 64],
        'BLOCK_SIZE_K_VALUE': [32, 64],
        'BLOCK_SIZE_N_VALUE': [32, 64],
        'THREAD_SIZE_M_VALUE': [4],
        'THREAD_SIZE_K_VALUE': [4],
        'THREAD_SIZE_N_VALUE': [4],
    }
}


class TestGridSearchTuner(unittest.TestCase):

    def test_tune_sparse_linear_dsd(self):
        print('==================== Testing Grid Search Tuner ====================')
        dense_input = torch.rand((M, K)).cuda()
        weight_mask = sparta.testing.block_mask(shape=(N, K)).cuda()
        dense_op = torch.nn.Linear(K, N).cuda()
        dense_op.weight = torch.nn.Parameter(dense_op.weight.detach() * weight_mask)
        sparse_op = sparta.nn.SparseLinear(dense_op, weight_mask=weight_mask)
        sparse_op.set_search_space(SEARCH_SPACE)
        impl, config = sparse_op.tune(
            sample_inputs=[dense_input],
            algo='grid'
        )
        self.assertIsNotNone(impl)
        self.assertIsNotNone(config)
        sparse_op.build(impl, config)
        sparse_op.cuda()
        torch.testing.assert_allclose(sparse_op(dense_input), dense_op(dense_input))
        print(f'PASS')


if __name__ == '__main__':
    unittest.main()
