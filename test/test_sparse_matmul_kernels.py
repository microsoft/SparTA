# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from typing import Type

import numpy as np

from sparta.specializer import kernels


SHAPE_CONFIG = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}
TILE_CONFIG = {
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 32,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16,
}


def test_matmul_kernel(kernel_class: Type[kernels.KernelBase], s, b, t, c, cfg):
    np.random.seed(2022)
    kernel = kernel_class(batch_size=4, sparse_type=s, biased=b, transpose=t, compressed=c)
    print(f'{kernel.get_kernel_name()}: {kernel.test(cfg, num_iters=1000)} ms')


class TestSparseMatmulKernels(unittest.TestCase):

    def test_sparta_sparse_matmul(self):
        print('==================== Testing SparTA Sparse Matmul Kernels ====================')
        for stype in ['sdd', 'dsd', 'dds']:
            for biased in [False, True]:
                for transpose in [False, True]:
                    for compressed in [False, True]:
                        test_matmul_kernel(
                            kernels.SparTATemplateSparseMatMulKernel,
                            stype, biased, transpose, compressed,
                            dict(SHAPE_CONFIG, **TILE_CONFIG)
                        )

    def test_openai_sparse_matmul(self):
        print('==================== Testing OpenAI Sparse Matmul Kernels ====================')
        for stype in ['sdd', 'dsd', 'dds']:
            for biased in [False, True]:
                for transpose in [False, True]:
                    for compressed in [False, True]:
                        test_matmul_kernel(
                            kernels.OpenAITemplateSparseMatMulKernel,
                            stype, biased, transpose, compressed,
                            SHAPE_CONFIG
                        )


if __name__ == '__main__':
    unittest.main()
