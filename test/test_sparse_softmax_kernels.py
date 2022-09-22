# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import numpy as np

from sparta.specializer import kernels


SHAPE_CONFIG = {
    'GLOBAL_H_VALUE': 1024,
    'GLOBAL_W_VALUE': 512,
}
TILE_CONFIG = {
    'BLOCK_SIZE_H_VALUE': 16,
    'BLOCK_SIZE_W_VALUE': 64,
    'ROW_TILE_VALUE': 4,
}


def test_softmax_kernel(compressed, cfg):
    np.random.seed(2022)
    kernel = kernels.SparTATemplateSparseSoftmaxKernel(batch_size=4, compressed=compressed)
    print(f'{kernel.get_kernel_name()}: {kernel.test(cfg, num_iters=1000)} ms')


class TestSparseSoftmaxKernels(unittest.TestCase):

    def test_sparta_sparse_softmax(self):
        print('==================== Testing SparTA Sparse Softmax Kernels ====================')
        kernels.SparTATemplateSparseSoftmaxKernel()
        for compressed in [False, True]:
            test_softmax_kernel(compressed, dict(SHAPE_CONFIG, **TILE_CONFIG))


if __name__ == '__main__':
    unittest.main()
