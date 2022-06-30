# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
os.sys.path.append(os.getcwd())

import numpy as np

from SparTA.OPs.Emitter.BlockSparseEmmiter import BlockSparseEmitter

if __name__ == '__main__':
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    np.random.seed(2022)

    M, K, N = 1024, 1024, 1024
    A = np.random.randn(M, K)
    W = np.random.randn(K, N) * (np.random.rand(K, N) < 0.001)
    bias = np.random.randn(N)

    emitter = BlockSparseEmitter(M, K, N, tmp_dir)
    tmp_config = {
        'BLOCK_SIZE_M_VALUE': 64,
        'BLOCK_SIZE_K_VALUE': 8,
        'BLOCK_SIZE_N_VALUE': 128,
        'THREAD_SIZE_M_VALUE': 8,
        'THREAD_SIZE_K_VALUE': 4,
        'THREAD_SIZE_N_VALUE': 8
    }
    latency = emitter.measure_trail_latency(A, W, bias, tmp_config)
    print(f'Latency: {latency} s')
