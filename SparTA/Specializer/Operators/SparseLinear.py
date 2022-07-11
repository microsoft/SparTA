# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from OpBase import FactoryBase

class SparseLinearFactory(FactoryBase):

    def _set_attrs(self):
        self._operator_name = 'sparse_linear'
        self._kernel_name = 'sparse_matmul'
        self._shape_features = ['M', 'N', 'K']
        self._inputs = {
            'A': {'type': 'float', 'shape': ['M', 'K'], 'sparsity': None},
            'W_val': {'type': 'float', 'shape': ['K', 'N'], 'sparsity': ['W_row', 'W_col']},
            'W_row': {'type': 'int', 'shape': ['K'], 'sparsity': None},
            'W_col': {'type': 'int', 'shape': ['N'], 'sparsity': None},
            'bias': {'type': 'float', 'shape': ['N'], 'sparsity': None}
        }
        self._outputs = {
            'C': {'type': 'float', 'shape': ['M', 'N'], 'sparsity': None}
        }
        self._tiles = {
            'block': ['BLOCK_SIZE_N_VALUE // THREAD_SIZE_N_VALUE', 'BLOCK_SIZE_M_VALUE // THREAD_SIZE_M_VALUE'],
            'grid': ['GLOBAL_N_VALUE // BLOCK_SIZE_N_VALUE', 'GLOBAL_M_VALUE // BLOCK_SIZE_M_VALUE']
        }

    def test(self):
        return


M, K, N = 1024, 256, 512
BM, BK, BN = 64, 8, 128
TM, TK, TN = 8, 4, 8

cfg = {
    # 'TYPE': 'float',
    'GLOBAL_M_VALUE': M,
    'GLOBAL_K_VALUE': K,
    'GLOBAL_N_VALUE': N,
    'BLOCK_SIZE_M_VALUE': BM,
    'BLOCK_SIZE_K_VALUE': BK,
    'BLOCK_SIZE_N_VALUE': BN,
    'THREAD_SIZE_M_VALUE': TM,
    'THREAD_SIZE_K_VALUE': TK,
    'THREAD_SIZE_N_VALUE': TN
}

A = np.random.randn(M, K)
W_dense = np.random.randn(N, K)

block_size_k = BK
block_size_n = BN
block_num_k = K // BK
block_num_n = N // BN
W_row = np.zeros(block_num_k + 1)
W_col = np.array([])
W_val = np.array([])
for block_i in range(block_num_n):
    block_start_n = block_i * block_size_n
    block_end_n = block_start_n + block_size_n
    for block_j in range(block_num_k):
        block_start_k = block_j * block_size_k
        block_end_k = block_start_k + block_size_k
        block = W_dense[block_start_n:block_end_n, block_start_k:block_end_k]
        if np.random.random() < 0.5:
            W_col = np.append(W_col, block_j)
            W_val = np.concatenate([W_val, block.T.flatten()])
        else:
            W_dense[block_start_n:block_end_n, block_start_k:block_end_k] = 0
    W_row[block_i + 1] = len(W_col)

bias = np.random.randn(N)

C = A @ W_dense.T + bias

f = SparseLinearFactory().get_test_function(cfg)
f.save_test_data(
    input_data={'A': A, 'W_val': W_val, 'W_col': W_col, 'W_row': W_row, 'bias': bias},
    output_data={'C': C}
)
print(f())