import os
import time

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.common import tesa
from sparta.specializer import specializer

np.random.seed(2022)

cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 8,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16
}

for kernel_cfg in ['', '_b', '_t', '_b_t']:
    kernel_name = f'sparse_linear_sdd{kernel_cfg}'
    factory = specializer.get_factory(kernel_name)
    print(f'{kernel_name}: {factory.get_test_func(cfg)(num_iters=1000)} ms')

exit()
