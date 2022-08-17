import os

import numpy as np

os.sys.path.append(os.getcwd())

from sparta.specializer import specializer, tuners

np.random.seed(2022)

space = {
    'BLOCK_SIZE_M_VALUE': [32, 64],
    'BLOCK_SIZE_K_VALUE': [32, 64],
    'BLOCK_SIZE_N_VALUE': [32, 64],
    'THREAD_SIZE_M_VALUE': [2, 4],
    'THREAD_SIZE_K_VALUE': [2, 4],
    'THREAD_SIZE_N_VALUE': [2, 4],
}

M, K, N = 1024, 256, 512
shape = {
    'GLOBAL_M_VALUE': M,
    'GLOBAL_K_VALUE': K,
    'GLOBAL_N_VALUE': N,
}

B_mask = np.random.uniform(size=(K, N)) < 0.01

linear_specializer = specializer.Specializer(op_name='sparse_linear_dsd', **shape)
best_config = linear_specializer.find_best_config(search_space=space, mask={'B': B_mask})
if best_config is not None:
    our_linear = linear_specializer.get_module(best_config, mask={'B': B_mask})
