import os

import numpy as np

os.sys.path.append(os.getcwd())

from sparta.specializer import kernels

np.random.seed(2022)

shape_cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}
tile_cfg = {
    'BLOCK_SIZE_M_VALUE': 64,
    'BLOCK_SIZE_K_VALUE': 32,
    'BLOCK_SIZE_N_VALUE': 128,
    'THREAD_SIZE_M_VALUE': 8,
    'THREAD_SIZE_K_VALUE': 4,
    'THREAD_SIZE_N_VALUE': 16,
}

def test_kernel(kernel_class: type[kernels.KernelBase], s, b, t, c, cfg):
    kernel = kernel_class(sparse_type=s, biased=b, transpose=t, compressed=c)
    print(f'{kernel.get_kernel_name()}: {kernel.test(cfg, num_iters=1000)} ms')

for stype in ['sdd', 'dsd']:
    for biased in [False, True]:
        for transpose in [False, True]:
            for compressed in [False, True]:
                test_kernel(
                    kernels.OurTemplateSparseMatMulKernel,
                    stype, biased, transpose, compressed,
                    shape_cfg | tile_cfg
                )

for stype in ['sdd', 'dsd', 'dds']:
    for biased in [False, True]:
        for transpose in [False, True]:
            for compressed in [True]:
                test_kernel(
                    kernels.OpenAITemplateSparseMatMulKernel,
                    stype, biased, transpose, compressed,
                    shape_cfg
                )
