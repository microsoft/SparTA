# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
import shutil

from jinja2 import Template
import pycuda.autoinit
from pycuda.compiler import SourceModule


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


s = SparseLinearFactory().get_test_function(**{
    # 'TYPE': 'float',
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 1024,
    'GLOBAL_N_VALUE': 1024,
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 8,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 8
})