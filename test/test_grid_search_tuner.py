# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch

from sparta.nn import SparseLinear
from sparta.testing import block_mask


BATCH_SIZE, IN_DIMS, OUT_DIMS = 1024, 256, 512
BM, BK, BN, TM, TK, TN = 32, 32, 32, 4, 4, 4
BLOCK = (8, 8)
SPARSITY = 0.95


class TestGridSearchTuner(unittest.TestCase):

    def test_grid_search_tuner(self):
        print('==================== Testing Grid Search Tuner ====================')
        dense_linear = torch.nn.Linear(IN_DIMS, OUT_DIMS, bias=True).cuda()

        torch.manual_seed(2022)
        sample_input = torch.rand((BATCH_SIZE, IN_DIMS), dtype=torch.float32).cuda()
        sample_grad = torch.rand((BATCH_SIZE, OUT_DIMS), dtype=torch.float32).cuda()

        mask = block_mask((OUT_DIMS, IN_DIMS), block=BLOCK, sparsity=SPARSITY).cuda()

        sparse_linear = SparseLinear(dense_linear, weight_mask=mask)
        kernel_names = ['forward:C', 'backward:A', 'backward:B']
        kernel_config = {
            'BLOCK_SIZE_M_VALUE': BM,
            'BLOCK_SIZE_K_VALUE': BK,
            'BLOCK_SIZE_N_VALUE': BN,
            'THREAD_SIZE_M_VALUE': TM,
            'THREAD_SIZE_K_VALUE': TK,
            'THREAD_SIZE_N_VALUE': TN,
        }
        sparse_linear.build(
            params=dict(
                _impl=';'.join([f'{kernel_name}=sparta' for kernel_name in kernel_names]),
                **{
                    f'{kernel_name};{param_name}': param_value
                    for param_name, param_value in kernel_config.items()
                    for kernel_name in kernel_names
                }
            ),
            sample_inputs=[sample_input]
        )

        latency = sparse_linear.test(kernel_names, [sample_input], sample_grad)
        print(latency)


if __name__ == '__main__':
    unittest.main()
