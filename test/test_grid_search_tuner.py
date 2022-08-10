import os

import numpy as np

os.sys.path.append(os.getcwd())

from sparta.specializer import specializer, tuners

np.random.seed(2022)
shape = {
    'GLOBAL_M_VALUE': 4096,
    'GLOBAL_K_VALUE': 768,
    'GLOBAL_N_VALUE': 3072,
}
space = {
    'BLOCK_SIZE_M_VALUE': [32, 64],
    'BLOCK_SIZE_K_VALUE': [32, 64],
    'BLOCK_SIZE_N_VALUE': [32, 64],
    'THREAD_SIZE_M_VALUE': [2, 4],
    'THREAD_SIZE_K_VALUE': [2, 4],
    'THREAD_SIZE_N_VALUE': [2, 4],
}
mask = np.random.uniform(size=(shape['GLOBAL_M_VALUE'], shape['GLOBAL_K_VALUE'])) < 0.01

cfg = tuners.GridSearchTunner(
    specializer=specializer.Specializer(op_name='sparse_linear_sdd', **shape),
    search_space=space
).find_best_config(mask={'A': mask})
