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

factory = specializer.get_factory('sparse_linear_sdd_b_t')

# Prepare Data
def prepare_data():
    W = np.random.normal(size=(cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    W_mask = np.random.uniform(size=(
        cfg['GLOBAL_M_VALUE'] // cfg['BLOCK_SIZE_M_VALUE'],
        cfg['GLOBAL_K_VALUE'] // cfg['BLOCK_SIZE_K_VALUE'],
    )) < 0.2
    W_tesa = tesa.BCSR(
        dense=W,
        mask=W_mask,
        block_size=(cfg['BLOCK_SIZE_M_VALUE'], cfg['BLOCK_SIZE_K_VALUE'])
    ).sparse
    W_val = W_tesa['val']
    B = np.random.normal(size=(cfg['GLOBAL_N_VALUE'], cfg['GLOBAL_K_VALUE'])).astype(np.float32)
    bias = np.random.normal(size=(cfg['GLOBAL_N_VALUE'], )).astype(np.float32)

    W_mask_tiled = np.zeros((cfg['GLOBAL_M_VALUE'], cfg['GLOBAL_K_VALUE']))
    for row_idx in range(W_mask.shape[0]):
        for col_idx in range(W_mask.shape[1]):
            row_start = row_idx * cfg['BLOCK_SIZE_M_VALUE']
            row_end = row_start + cfg['BLOCK_SIZE_M_VALUE']
            col_start = col_idx * cfg['BLOCK_SIZE_K_VALUE']
            col_end = col_start + cfg['BLOCK_SIZE_K_VALUE']
            W_mask_tiled[row_start:row_end, col_start:col_end] = W_mask[row_idx, col_idx]

    W *= W_mask_tiled
    C_tgt = W @ B.T + bias
    return W, W_val, W_mask, B, bias, C_tgt

W, W_val, W_mask, B, bias, C_tgt = prepare_data()

# Test Function
test_func = factory.get_test_func(cfg, mask={'W': W_mask})
print(f'NVCC Latency: {test_func(inputs={"W": W, "B": B, "bias": bias}, num_iters=1000)} ms')

# PyTorch Module
module_code = factory.get_module_code(cfg, mask={'W': W_mask})
with open('./test/module.cu', 'w') as f:
    f.write(module_code)

f = factory.get_module(cfg, mask={'W': W_mask}).forward

device = torch.device(f'cuda:3')
W_val = torch.from_numpy(W_val).to(device)
B = torch.from_numpy(B).to(device)
bias = torch.from_numpy(bias).to(device)

for _ in range(10):
    C = f(W_val, B, bias)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    C = f(W_val, B, bias)
torch.cuda.synchronize()
print(f'PyTorch Latency: {(time.time() - start)} ms')

C = C.cpu().numpy()
print(f'Sum_C: {C.sum()}')
print(f'Sum_C_tgt: {C_tgt.sum()}')
print(f'Error: {np.sum(np.abs(C - C_tgt))}')
