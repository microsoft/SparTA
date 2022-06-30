# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import logging
os.sys.path.append(os.getcwd())

import numpy as np

from SparTA.OPs.Emitter.BlockSparseEmmiter import BlockSparseEmitter
from SparTA.OPs.Tunner.GridSearchTunner import GridSearchTunner

if __name__ == '__main__':
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    log_file = os.path.join(tmp_dir, 'tunning_log.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()

    np.random.seed(2022)

    M, K, N = 1024, 1024, 1024
    A = np.random.randn(M, K)
    W = np.random.randn(K, N) * (np.random.rand(K, N) < 0.001)
    bias = np.random.randn(N)

    emitter = BlockSparseEmitter(M, K, N, tmp_dir)
    tunner = GridSearchTunner(emitter, emitter.template_cfg['space'], logger)

    print(f'Tunning... View log at: {log_file}')
    best_cfg = tunner.tunning_kernel_cfg(A, W, bias)
    print(f'Best Config: {best_cfg}')
