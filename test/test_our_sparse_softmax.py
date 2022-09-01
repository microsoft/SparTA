import os
import time

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.common import tesa
from sparta.specializer import kernels

np.random.seed(2022)

cfg = {
    'GLOBAL_H_VALUE': 1024,
    'GLOBAL_W_VALUE': 512,
    'BLOCK_SIZE_H_VALUE': 32,
    'BLOCK_SIZE_W_VALUE': 32,
    'ROW_TILE_VALUE': 4,
}

kernel = kernels.OurTemplateSparseSoftmaxKernel()

# Prepare Data
def prepare_data():
    C = np.random.uniform(size=(cfg['GLOBAL_H_VALUE'], cfg['GLOBAL_W_VALUE'])).astype(np.float32)
    C_mask = np.random.uniform(size=(
        cfg['GLOBAL_H_VALUE'] // cfg['BLOCK_SIZE_H_VALUE'],
        cfg['GLOBAL_W_VALUE'] // cfg['BLOCK_SIZE_W_VALUE'],
    )) < 0.2
    C_tesa = tesa.BCSR(
        dense=C,
        mask=C_mask,
        block_size=(cfg['BLOCK_SIZE_H_VALUE'], cfg['BLOCK_SIZE_W_VALUE']),
        mode='H',
    ).sparse
    C_val = C_tesa['val']

    C_mask_tiled = np.zeros((cfg['GLOBAL_H_VALUE'], cfg['GLOBAL_W_VALUE']))
    for row_idx in range(C_mask.shape[0]):
        for col_idx in range(C_mask.shape[1]):
            row_start = row_idx * cfg['BLOCK_SIZE_H_VALUE']
            row_end = row_start + cfg['BLOCK_SIZE_H_VALUE']
            col_start = col_idx * cfg['BLOCK_SIZE_W_VALUE']
            col_end = col_start + cfg['BLOCK_SIZE_W_VALUE']
            C_mask_tiled[row_start:row_end, col_start:col_end] = C_mask[row_idx, col_idx]

    C *= C_mask_tiled
    C_max = C.max(axis=-1).reshape((-1, 1))
    C_exp = np.exp(C - C_max) * C_mask_tiled
    C_exp_sum = C_exp.sum(axis=-1).reshape((-1, 1)) + 1e-10
    C_tgt = C_exp / C_exp_sum
    C_tgt = C_tgt.astype(np.float32)
    return C, C_val, C_mask_tiled, C_tgt

C, C_val, C_mask, C_tgt = prepare_data()

# Test Function
test_latency = kernel.test(cfg, mask={'C_in': C_mask}, inputs={'C_in': C}, num_iters=1000)
print(f'NVCC Latency: {test_latency} ms')

# PyTorch Module
f = kernel.compile(cfg, mask={'C_in': C_mask}).forward

device = torch.device(f'cuda')
C_in = torch.from_numpy(C_val).to(device)

for _ in range(10):
    C_out = f(C_in)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C_out = f(C_in)
torch.cuda.synchronize()
print(f'PyTorch Latency: {(time.time() - start)} ms')

C_out = C_out.cpu().numpy()
print(f'Sum_C: {C_out.sum()}')
print(f'Sum_C_tgt: {C_tgt.sum()}')
print(f'Error: {np.sum(np.abs(C_out - C_tgt))}')
