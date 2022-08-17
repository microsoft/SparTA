import os

import numpy as np

os.sys.path.append(os.getcwd())

from sparta.specializer import specializer

np.random.seed(2022)

shape_cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}
tile_cfg = {
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 8,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16,
}

def test_kernel(kernel_name, cfg):
    factory = specializer.get_factory(kernel_name)
    print(f'{kernel_name}: {factory.get_test_func(cfg)(num_iters=1000)} ms')

for kernel_type in ['sdd', 'dsd']:
    for kernel_cfg in ['', '_b', '_t', '_b_t']:
        kernel_name = f'sparse_linear_{kernel_type}{kernel_cfg}'
        test_kernel(kernel_name, shape_cfg | tile_cfg)

for kernel_type in ['sdd', 'dsd', 'dds']:
    for kernel_cfg in ['', '_t']:
        openai_kernel_name = f'sparse_linear_openai_{kernel_type}{kernel_cfg}'
        test_kernel(openai_kernel_name, shape_cfg)
