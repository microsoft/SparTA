import os
import time

import torch
import numpy as np

os.sys.path.append(os.getcwd())

from sparta.common import tesa
from sparta.specializer import kernels

np.random.seed(2022)

cfg = {
    'GLOBAL_M_VALUE': 1024,
    'GLOBAL_K_VALUE': 256,
    'GLOBAL_N_VALUE': 512,
}

kernel = kernels.OpenAITemplateSparseMatMulKernel('dds', biased=False, transpose=True)

# Prepare Data
def prepare_data():
    A = np.random.uniform(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    B = np.random.uniform(size=(cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)

    C_tgt = (A @ B.T).astype(np.float32)
    C_mask = np.random.uniform(size=(
        cfg['GLOBAL_M_VALUE'] // 32,
        cfg['GLOBAL_N_VALUE'] // 32,
    )) < 0.2
    C_tesa = tesa.BCSR(
        dense=C_tgt,
        mask=C_mask,
        block_size=(32, 32),
        mode='X',
    ).sparse
    C_val = C_tesa['val']

    C_mask = np.tile(C_mask.reshape(C_mask.shape + (1, 1)), [1, 1, 32, 32])
    C_mask = C_mask.swapaxes(1, 2).reshape((cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_N_VALUE']))

    return A, B, C_val, C_mask, C_tgt

A, B, C_val, C_mask, C_tgt = prepare_data()

# Test Function
test_latency = kernel.test(cfg, mask={'C': C_mask}, inputs={'A': A, 'B': B}, num_iters=1000)
print(f'NVCC Latency: {test_latency} ms')

# PyTorch Module
f = kernel.compile(cfg, mask={'C': C_mask}).forward

device = torch.device(f'cuda:3')
A = torch.from_numpy(A).to(device)
B = torch.from_numpy(B).to(device)

for _ in range(10):
    C = f(A, B)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C = f(A, B)
torch.cuda.synchronize()
print(f'PyTorch Latency: {(time.time() - start)} ms')

C = C.cpu().numpy()
print(f'Sum_C: {C.sum()}')
print(f'Sum_C_tgt: {C_val.sum()}')
print(f'Error: {np.sum(np.abs(C - C_val))}')
