import torch
from sparta.specializer.jit import kernels as jk
import numpy as np
import time

from sparta.common import tesa
from sparta import specializer

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule

def prepare_data(cfg):
    A = np.random.uniform(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    B = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    B_mask = np.random.uniform(size=(
        cfg['GLOBAL_N_VALUE'] // cfg['BLOCK_SIZE_N_VALUE'],
        cfg['GLOBAL_K_VALUE'] // cfg['BLOCK_SIZE_K_VALUE'],
    )) < 0.2
    B_tesa = tesa.BCSR(
        dense=B,
        mask=B_mask,
        block_size=(cfg['BLOCK_SIZE_N_VALUE'], cfg['BLOCK_SIZE_K_VALUE'])
    ).sparse
    B_val = B_tesa['val']
    bias = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], )).astype(np.float32)

    B_mask_tiled = np.zeros((cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE']))
    for row_idx in range(B_mask.shape[0]):
        for col_idx in range(B_mask.shape[1]):
            row_start = row_idx * cfg['BLOCK_SIZE_N_VALUE']
            row_end = row_start + cfg['BLOCK_SIZE_N_VALUE']
            col_start = col_idx * cfg['BLOCK_SIZE_K_VALUE']
            col_end = col_start + cfg['BLOCK_SIZE_K_VALUE']
            B_mask_tiled[row_start:row_end, col_start:col_end] = B_mask[row_idx, col_idx]

    B *= B_mask_tiled
    C_tgt = A @ B.T 
    return A, B, B_tesa, B_mask, bias, C_tgt


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

A, B, B_tesa, B_mask, bias, C_tgt = prepare_data(cfg)
TG = torch.from_numpy(C_tgt).cuda()
AG = torch.from_numpy(A).cuda()
BG = {k:torch.from_numpy(v).cuda() for k,v in B_tesa.items()}
VALG = torch.from_numpy(B_tesa['val']).cuda()
PTRG = torch.from_numpy(B_tesa['row_ptr']).cuda()
IDXG = torch.from_numpy(B_tesa['col_idx']).cuda()
CG = torch.empty(C_tgt.shape).cuda()

s = jk.SparseMatMul('dsd', transpose=True)
s.set_parameters(cfg)
s.compile()

s.matmul(AG, VALG, PTRG, IDXG, CG, bias)
torch.cuda.synchronize()
torch.testing.assert_close(CG,TG)

for _ in range(10):
    s.matmul(AG, VALG, PTRG, IDXG, CG, bias)

torch.cuda.synchronize()


start = time.time()
for _ in range(1000):
    s.matmul(AG, VALG, PTRG, IDXG, CG, bias)
torch.cuda.synchronize()
print(f'PyCuda Latency: {(time.time() - start)} ms')

